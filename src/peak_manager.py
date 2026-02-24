"""
PeakManager
===========
Manages peak detection and energy assignments per channel.

Two-phase workflow:
  Phase 1 — Detection: TSpectrum finds raw peak positions per channel (no energy yet).
             Stored in detected_positions[ch_id] = list[float]
  Phase 2 — Assignment: user assigns known energies to detected positions.
             An energy assignment on one channel propagates to all other channels
             that have a detected peak within ± window ADC counts of the same position.

Per-channel behaviour:
  - excluded_channels  : channels skipped during energy propagation
  - channel_peaks      : explicit peak list for a channel (overrides global)
  - global_peaks       : fallback used when no per-channel list exists
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from scipy.ndimage import gaussian_filter1d


@dataclass
class Peak:
    adc_position:  float
    known_energy:  float
    label:         str  = ""
    auto_detected: bool = False

@dataclass
class RefinedPeakResult:
    adc: float
    sigma: float
    amplitude: float
    chi2_ndf: float
    success: bool
    reason: str = ""

# Known internal emission lines for crystal types (energy in keV)
CRYSTAL_KNOWN_LINES = {
    "lyso": [
        {"energy": 88.34,  "label": "Lu-176 88 keV",  "color": "#7b1fa2"},
        {"energy": 201.83, "label": "Lu-176 202 keV",  "color": "#7b1fa2"},
        {"energy": 306.78, "label": "Lu-176 307 keV",  "color": "#7b1fa2"},
    ],
    "gagg": [],
    "generic": [],
}


class PeakManager:
    """
    detected_positions : dict[int, list[float]]   — raw ADC positions per channel (no energy)
    channel_peaks      : dict[int, list[Peak]]    — assigned (energy-labelled) peaks per channel
    global_peaks       : list[Peak]               — fallback for channels with no assignment
    excluded_channels  : set[int]                 — channels excluded from energy propagation
    """

    def __init__(self):
        self.detected_positions: dict[int, list[float]] = {}
        self.global_peaks:       list[Peak]             = []
        self.channel_peaks:      dict[int, list[Peak]]  = {}
        self.excluded_channels:  set[int]               = set()

    # ------------------------------------------------------------------ #
    # Detected positions (Phase 1)
    # ------------------------------------------------------------------ #

    def set_detected(self, channel_id: int, positions: list[float]):
        """Store raw detected ADC positions for a channel."""
        self.detected_positions[channel_id] = sorted(positions)

    def get_detected(self, channel_id: int) -> list[float]:
        return self.detected_positions.get(channel_id, [])

    def clear_detected(self, channel_id: int = None):
        if channel_id is None:
            self.detected_positions.clear()
        else:
            self.detected_positions.pop(channel_id, None)

    # ------------------------------------------------------------------ #
    # Exclusion (per-channel override — skip propagation)
    # ------------------------------------------------------------------ #

    def set_excluded(self, channel_id: int, excluded: bool):
        if excluded:
            self.excluded_channels.add(channel_id)
        else:
            self.excluded_channels.discard(channel_id)

    def is_excluded(self, channel_id: int) -> bool:
        return channel_id in self.excluded_channels

    # ------------------------------------------------------------------ #
    # Global peaks
    # ------------------------------------------------------------------ #

    def add_global_peak(self, adc_position: float, known_energy: float,
                         label: str = "", auto: bool = False):
        self.global_peaks.append(
            Peak(adc_position, known_energy, label, auto))

    def remove_global_peak(self, index: int):
        if 0 <= index < len(self.global_peaks):
            self.global_peaks.pop(index)

    def clear_global_peaks(self):
        self.global_peaks.clear()

    # ------------------------------------------------------------------ #
    # Per-channel assigned peaks
    # ------------------------------------------------------------------ #

    def set_channel_peaks(self, channel_id: int, peaks: list[Peak]):
        self.channel_peaks[channel_id] = peaks

    def add_channel_peak(self, channel_id: int, adc_position: float,
                          known_energy: float, label: str = ""):
        if channel_id not in self.channel_peaks:
            self.channel_peaks[channel_id] = [
                Peak(p.adc_position, p.known_energy, p.label, p.auto_detected)
                for p in self.global_peaks
            ]
        self.channel_peaks[channel_id].append(
            Peak(adc_position, known_energy, label))

    def remove_channel_peak(self, channel_id: int, index: int):
        peaks = self.channel_peaks.get(channel_id, [])
        if 0 <= index < len(peaks):
            peaks.pop(index)

    def reset_channel(self, channel_id: int):
        """Remove per-channel peak assignment; revert to global."""
        self.channel_peaks.pop(channel_id, None)

    def has_channel_peaks(self, channel_id: int) -> bool:
        return channel_id in self.channel_peaks

    # ------------------------------------------------------------------ #
    # Effective peaks (for calibration)
    # ------------------------------------------------------------------ #

    def get_peaks(self, channel_id: int) -> list[Peak]:
        if channel_id in self.channel_peaks:
            return self.channel_peaks[channel_id]
        return self.global_peaks

    def get_calibration_points(self, channel_id: int
                                ) -> tuple[np.ndarray, np.ndarray]:
        peaks = self.get_peaks(channel_id)
        if not peaks:
            return np.array([]), np.array([])
        adc = np.array([p.adc_position for p in peaks])
        eng = np.array([p.known_energy  for p in peaks])
        return adc, eng

    def n_calibration_points(self, channel_id: int) -> int:
        return len(self.get_peaks(channel_id))

    # ------------------------------------------------------------------ #
    # Energy propagation (Phase 2 helper)
    # ------------------------------------------------------------------ #

    def propagate_energy(self,
                          source_adc:   float,
                          known_energy: float,
                          label:        str,
                          window:       float,
                          all_channels: list[int],
                          source_ch:    int) -> list[int]:
        """
        For every channel (except source_ch and excluded channels) that has a
        detected peak within source_adc ± window, assign known_energy to that
        detected position.

        Returns list of channel IDs that received the assignment.
        """
        updated = []
        for ch_id in all_channels:
            if ch_id == source_ch:
                continue
            if ch_id in self.excluded_channels:
                continue
            detected = self.detected_positions.get(ch_id, [])
            if not detected:
                continue
            # Find the closest detected peak within window
            dists = [(abs(p - source_adc), p) for p in detected]
            dists.sort()
            if dists and dists[0][0] <= window:
                best_adc = dists[0][1]
                # Add or update this energy assignment in channel peaks
                if ch_id not in self.channel_peaks:
                    self.channel_peaks[ch_id] = []
                # Remove any existing assignment at this ADC (within tiny tolerance)
                tol = window * 0.1
                self.channel_peaks[ch_id] = [
                    p for p in self.channel_peaks[ch_id]
                    if abs(p.adc_position - best_adc) > tol
                ]
                self.channel_peaks[ch_id].append(
                    Peak(adc_position=best_adc,
                         known_energy=known_energy,
                         label=label,
                         auto_detected=True))
                updated.append(ch_id)
        return updated

    def refine_detected_peaks_sliding_gauss(self,
                                        channel_id: int,
                                        bin_centers: np.ndarray,
                                        counts: np.ndarray,
                                        window_adc: float = 30.0,
                                        max_chi2_ndf: float = 10.0,
                                        dedup_tol: float = 5.0
                                        ) -> list[RefinedPeakResult]:
        """
        Refine detected peaks for one channel using:
        sliding window → local max → Gaussian fit

        Updates detected_positions[channel_id] in-place.
    
        Returns list of refinement results (good + rejected).
        """

        raw = self.detected_positions.get(channel_id, [])
        if not raw:
            return []

        refined = []
        accepted_adc = []

        for adc0 in raw:
            # Sliding max
            mask = (bin_centers >= adc0 - window_adc) & \
                   (bin_centers <= adc0 + window_adc)
            if not np.any(mask):
                continue

            local_x = bin_centers[mask]
            local_y = counts[mask]
            peak_adc = local_x[np.argmax(local_y)]

            res = self._fit_gaussian(
                bin_centers, counts, peak_adc, window_adc)

            if not res.success:
                refined.append(res)
                continue

            if np.isfinite(res.chi2_ndf) and res.chi2_ndf > max_chi2_ndf:
                res.success = False
                res.reason  = "Bad χ²/NDF"
                refined.append(res)
                continue

            # Deduplicate
            if any(abs(res.adc - a) < dedup_tol for a in accepted_adc):
                continue

            accepted_adc.append(res.adc)
            refined.append(res)

        # Replace detected positions with refined good peaks
        self.detected_positions[channel_id] = sorted(accepted_adc)

        return refined

    # ------------------------------------------------------------------ #
    # Peak detection — TSpectrum (PyROOT)
    # ------------------------------------------------------------------ #

    @staticmethod
    def detect_peaks_tspectrum(bin_centers: np.ndarray,
                                counts: np.ndarray,
                                sigma: float = 2.0,
                                threshold: float = 0.05,
                                max_peaks: int = 10,
                                pedestal_cut: float = 0.0,
                                tspec_mode: str = "highres",
                                iterations: int = 10,
                                bg_subtract: bool = True,
                                bg_iterations: int = 20,
                                ) -> tuple[list[float], np.ndarray | None]:
        """
        Detect peaks using ROOT TSpectrum.

        Returns (positions, bg_counts) where bg_counts is the estimated
        SNIP background array (same length as counts after pedestal cut),
        or None if bg_subtract is False or ROOT unavailable.

        bg_subtract    : run TSpectrum::Background() (SNIP) first to
                         estimate and remove the continuum before searching.
                         Critical for LYSO where the Lu-176 beta continuum
                         causes many false peaks if left unsubtracted.
        bg_iterations  : SNIP smoothing width — larger = broader background
                         estimate. Try 20–40 for LYSO.

        tspec_mode == "standard"  : TSpectrum::Search()
        tspec_mode == "highres"   : TSpectrum::SearchHighRes() — deconvolution,
                                    resolves closely-spaced peaks.
        """
        import os, array as arr
        try:
            import ROOT

            if pedestal_cut > 0:
                mask        = bin_centers >= pedestal_cut
                bin_centers = bin_centers[mask]
                counts      = counts[mask]

            if len(counts) == 0:
                return [], None

            n         = len(counts)
            counts_f  = counts.astype(float)

            # ── Suppress ROOT stderr ──────────────────────────────────────
            devnull_fd   = os.open(os.devnull, os.O_WRONLY)
            saved_stderr = os.dup(2)
            os.dup2(devnull_fd, 2)
            os.close(devnull_fd)

            bg_array = None   # will hold estimated background

            try:
                import threading
                hname = f"_tspec_{threading.get_ident()}_{id(bin_centers)}"
                h = ROOT.TH1F(hname, "", n,
                               float(bin_centers[0]),
                               float(bin_centers[-1]))
                h.SetDirectory(ROOT.nullptr)
                ROOT.SetOwnership(h, True)
                for i, c in enumerate(counts_f):
                    h.SetBinContent(i + 1, c)

                sp = ROOT.TSpectrum(max_peaks)

                # ── SNIP background estimate ──────────────────────────────
                if bg_subtract:
                    hbg = sp.Background(h, int(bg_iterations),
                                        "BackDecreasing BackSmoothing3")
                    ROOT.SetOwnership(hbg, True)
                    bg_array = np.array([hbg.GetBinContent(i + 1)
                                         for i in range(n)], dtype=float)
                    # Subtract from working array for peak search
                    counts_sub = np.maximum(counts_f - bg_array, 0.0)
                    # Reload histogram with subtracted counts
                    for i, c in enumerate(counts_sub):
                        h.SetBinContent(i + 1, c)
                    try:
                        hbg.Delete(); del hbg
                    except Exception:
                        pass
                else:
                    counts_sub = counts_f

                # ── Peak search ───────────────────────────────────────────
                if tspec_mode == "highres":
                    src_arr  = arr.array("d", [float(c) for c in counts_sub])
                    dest_arr = arr.array("d", [0.0] * n)
                    n_found  = sp.SearchHighRes(
                        src_arr, dest_arr, n,
                        float(sigma), float(threshold),
                        False,            # backgroundRemove — already done
                        int(iterations),
                        False, 3)
                    positions = []
                    px = sp.GetPositionX()
                    for i in range(n_found):
                        bin_idx = max(0, min(n - 1, int(round(float(px[i])))))
                        positions.append(float(bin_centers[bin_idx]))
                else:
                    n_found = sp.Search(h, sigma, "nobackground nodraw", threshold)
                    positions = []
                    px = sp.GetPositionX()
                    for i in range(n_found):
                        positions.append(float(px[i]))

                result = sorted(positions)

            finally:
                os.dup2(saved_stderr, 2)
                os.close(saved_stderr)
                try:
                    del sp
                    h.Delete(); del h
                except Exception:
                    pass

            return result, bg_array

        except Exception:
            positions = PeakManager.detect_peaks_scipy(
                bin_centers, counts, sigma, threshold, max_peaks,
                pedestal_cut=0.0)
            return positions, None

    # ------------------------------------------------------------------ #
    # Peak detection — scipy fallback
    # ------------------------------------------------------------------ #

    @staticmethod
    def detect_peaks_scipy(bin_centers: np.ndarray,
                            counts: np.ndarray,
                            sigma: float = 2.0,
                            threshold: float = 0.05,
                            max_peaks: int = 10,
                            pedestal_cut: float = 0.0
                            ) -> list[float]:
        """Detect peaks using scipy.signal.find_peaks."""
        from scipy.signal import find_peaks as sp_find_peaks

        if pedestal_cut > 0:
            mask        = bin_centers >= pedestal_cut
            bin_centers = bin_centers[mask]
            counts      = counts[mask]

        if len(counts) == 0 or counts.sum() == 0:
            return []

        smoothed       = gaussian_filter1d(counts.astype(float), sigma=sigma)
        min_prominence = max(1.0, counts.max() * threshold)
        min_distance   = max(1, int(len(counts) * 0.02))

        indices, props = sp_find_peaks(smoothed,
                                        prominence=min_prominence,
                                        distance=min_distance)
        if len(indices) == 0:
            return []

        order = np.argsort(props["prominences"])[::-1]
        top   = indices[order[:max_peaks]]

        positions = []
        window_w = max(3, int(len(counts) * 0.01))
        for idx in top:
            lo = max(0, idx - window_w)
            hi = min(len(counts), idx + window_w + 1)
            w  = counts[lo:hi]
            c  = bin_centers[lo:hi]
            centroid = float(np.sum(c * w) / w.sum()) if w.sum() > 0 \
                       else float(bin_centers[idx])
            positions.append(centroid)

        return sorted(positions)

    # ------------------------------------------------------------------ #
    # Fit peaks with gaussian for refinement
    # ------------------------------------------------------------------ #

    @staticmethod
    def _fit_gaussian(bin_centers, counts, center, window):
        """
        Local Gaussian fit around 'center'.
        Returns RefinedPeakResult.
        """
        mask = (bin_centers >= center - window) & \
               (bin_centers <= center + window)

        x = bin_centers[mask]
        y = counts[mask]

        if len(x) < 5 or y.max() <= 0:
            return RefinedPeakResult(center, np.nan, 0, np.nan,
                                 False, "Insufficient data")

        # Initial guesses
        amp0 = y.max()
        mu0  = x[y.argmax()]
        sig0 = window / 4

        # Try ROOT first if available
        try:
            import ROOT
            h = ROOT.TH1F("tmp", "", len(x), float(x[0]), float(x[-1]))
            h.SetDirectory(ROOT.nullptr)
            for i, c in enumerate(y):
                h.SetBinContent(i + 1, float(c))

            f = ROOT.TF1("g", "gaus", float(x[0]), float(x[-1]))
            f.SetParameters(amp0, mu0, sig0)

            status = h.Fit(f, "RQ0")
            if status != 0:
                raise RuntimeError("ROOT fit failed")

            chi2 = f.GetChisquare()
            ndf  = f.GetNDF()
            chi2_ndf = chi2 / ndf if ndf > 0 else np.nan

            return RefinedPeakResult(
                adc=f.GetParameter(1),
                sigma=abs(f.GetParameter(2)),
                amplitude=f.GetParameter(0),
                chi2_ndf=chi2_ndf,
                success=True
            )

        except Exception:
            # scipy fallback
            from scipy.optimize import curve_fit

            def gaus(x, A, mu, sig):
                return A * np.exp(-(x - mu) ** 2 / (2 * sig ** 2))

            try:
                popt, pcov = curve_fit(
                    gaus, x, y, p0=[amp0, mu0, sig0], maxfev=5000)
                residuals = y - gaus(x, *popt)
                chi2 = np.sum(residuals ** 2)
                ndf  = len(y) - 3
                chi2_ndf = chi2 / ndf if ndf > 0 else np.nan

                return RefinedPeakResult(
                    adc=popt[1],
                    sigma=abs(popt[2]),
                    amplitude=popt[0],
                    chi2_ndf=chi2_ndf,
                    success=True
                )

            except Exception:
                return RefinedPeakResult(center, np.nan, 0, np.nan,
                                     False, "Fit failed")
    

    # ------------------------------------------------------------------ #
    # Auto-detect dispatcher
    # ------------------------------------------------------------------ #

    @staticmethod
    def detect_peaks(bin_centers: np.ndarray,
                  counts: np.ndarray,
                  sigma: float = 2.0,
                  threshold: float = 0.05,
                  max_peaks: int = 10,
                  backend: str = "pyroot",
                  pedestal_cut: float = 0.0,
                  tspec_mode: str = "highres",
                  iterations: int = 10,
                  bg_subtract: bool = True,
                  bg_iterations: int = 20,
                  refine: bool = False,
                  refine_window_adc: float = 30.0,
                  max_chi2_ndf: float = 10.0,
                  ):
        """
        Peak detection dispatcher.
    
        Phase 1:
            - TSpectrum (standard / highres) or scipy
            - returns raw ADC peak positions

        Optional Phase 1b (refine=True):
        - sliding window local maximum
        - Gaussian fit
        - reject bad fits

        Returns:
        positions : list[float]
        bg_array  : np.ndarray | None
        refine_results : list[RefinedPeakResult] | None
        """

        # ── Phase 1: coarse detection ───────────────────────────────
        if backend == "pyroot":
            positions, bg_array = PeakManager.detect_peaks_tspectrum(
                bin_centers, counts, sigma, threshold, max_peaks,
                pedestal_cut=pedestal_cut,
                tspec_mode=tspec_mode,
                iterations=iterations,
                bg_subtract=bg_subtract,
                bg_iterations=bg_iterations)
        else:
            positions = PeakManager.detect_peaks_scipy(
                bin_centers, counts, sigma, threshold, max_peaks,
                pedestal_cut=pedestal_cut)
            bg_array = None

        if not refine or not positions:
            return positions, bg_array, None

        # ── Phase 1b: refinement ────────────────────────────────────
        refine_results = []
        refined_positions = []

        for adc0 in positions:
            # sliding window
            mask = (bin_centers >= adc0 - refine_window_adc) & \
                   (bin_centers <= adc0 + refine_window_adc)
            if not np.any(mask):
                continue

            local_x = bin_centers[mask]
            local_y = counts[mask]
            peak_adc = local_x[np.argmax(local_y)]

            res = PeakManager._fit_gaussian(
                bin_centers, counts, peak_adc, refine_window_adc)

            if not res.success:
                refine_results.append(res)
                continue

            if np.isfinite(res.chi2_ndf) and res.chi2_ndf > max_chi2_ndf:
                res.success = False
                res.reason  = "Bad χ²/NDF"
                refine_results.append(res)
                continue

            refined_positions.append(res.adc)
            refine_results.append(res)

        # de-duplicate
        refined_positions = sorted(set(refined_positions))

        return refined_positions, bg_array, refine_results
