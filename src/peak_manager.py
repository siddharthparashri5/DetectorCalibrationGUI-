"""
PeakManager
===========
Manages peak detection and energy assignments per channel.

Detection algorithms:
  1. TSpectrum (PyROOT) — SearchHighRes or Search
  2. scipy      — find_peaks fallback
  3. Sliding window — user-defined window scans spectrum for local maxima

All detectors optionally confirm peaks with a Gaussian fit
(returns refined centroid + sigma, rejects non-peak-like shapes).

Two-phase workflow:
  Phase 1 — Detection: algorithm finds raw peak positions per channel (no energy yet).
  Phase 2 — Assignment: user assigns known energies to detected positions.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit


@dataclass
class Peak:
    adc_position:  float
    known_energy:  float
    label:         str  = ""
    auto_detected: bool = False


@dataclass
class GaussianFitInfo:
    """Result of a Gaussian confirmation fit on a single detected peak."""
    adc:       float          # input candidate position
    success:   bool
    centroid:  float = 0.0   # refined centroid from Gaussian fit
    sigma:     float = 0.0
    amplitude: float = 0.0
    chi2_ndf:  float = 0.0
    reason:    str   = ""    # failure reason if not success


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


# ─────────────────────────────────────────────────────────────────────────── #
# Gaussian helpers
# ─────────────────────────────────────────────────────────────────────────── #

def _gauss(x, A, mu, sigma):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def _gauss_linear(x, A, mu, sigma, B, C):
    return _gauss(x, A, mu, sigma) + B + C * x


def fit_gaussian_to_peak(bin_centers: np.ndarray,
                          counts: np.ndarray,
                          peak_adc: float,
                          window_adc: float = 20.0,
                          min_points: int = 6
                          ) -> GaussianFitInfo:
    """
    Fit a Gaussian + linear background to the region around peak_adc.

    Parameters
    ----------
    bin_centers : ADC bin center array
    counts      : count array
    peak_adc    : candidate peak position (ADC)
    window_adc  : half-width of fit window (ADC units)
    min_points  : minimum bins in window for a valid fit

    Returns
    -------
    GaussianFitInfo with refined centroid (or failure info)
    """
    mask = (bin_centers >= peak_adc - window_adc) & \
           (bin_centers <= peak_adc + window_adc)
    x = bin_centers[mask]
    y = counts[mask].astype(float)

    if len(x) < min_points:
        return GaussianFitInfo(adc=peak_adc, success=False,
                               reason=f"Only {len(x)} bins in window (need {min_points})")

    A0    = float(y.max())
    mu0   = float(x[np.argmax(y)])
    sig0  = window_adc / 3.0
    B0    = float(np.percentile(y, 10))

    try:
        popt, pcov = curve_fit(
            _gauss_linear, x, y,
            p0=[A0, mu0, sig0, max(B0, 0.0), 0.0],
            bounds=([0, x.min(), 0.5, -np.inf, -np.inf],
                    [np.inf, x.max(), window_adc * 2, np.inf, np.inf]),
            maxfev=5000, ftol=1e-9, xtol=1e-9)
        perr = np.sqrt(np.abs(np.diag(pcov)))
    except Exception as e:
        return GaussianFitInfo(adc=peak_adc, success=False,
                               reason=f"Fit failed: {e}")

    A, mu, sigma = popt[0], popt[1], popt[2]

    # Sanity checks
    if np.isnan(sigma) or sigma <= 0 or np.any(np.isnan(perr)):
        return GaussianFitInfo(adc=peak_adc, success=False,
                               reason="Invalid sigma or covariance NaN")
    if abs(mu - peak_adc) > window_adc:
        return GaussianFitInfo(adc=peak_adc, success=False,
                               reason=f"Centroid drifted too far: {mu:.1f} vs {peak_adc:.1f}")

    fitted   = _gauss_linear(x, *popt)
    residuals = y - fitted
    ndf      = max(1, len(x) - len(popt))
    chi2_ndf = float(np.sum(residuals ** 2) / ndf)

    return GaussianFitInfo(
        adc=peak_adc, success=True,
        centroid=float(mu), sigma=float(sigma),
        amplitude=float(A), chi2_ndf=chi2_ndf)


# ─────────────────────────────────────────────────────────────────────────── #
# PeakManager
# ─────────────────────────────────────────────────────────────────────────── #

class PeakManager:
    """
    detected_positions : dict[int, list[float]]    — raw ADC positions per channel
    channel_peaks      : dict[int, list[Peak]]     — assigned (energy-labelled) peaks
    global_peaks       : list[Peak]                — fallback for channels with no assignment
    excluded_channels  : set[int]                  — excluded from energy propagation
    """

    def __init__(self):
        self.detected_positions: dict[int, list[float]] = {}
        self.global_peaks:       list[Peak]             = []
        self.channel_peaks:      dict[int, list[Peak]]  = {}
        self.excluded_channels:  set[int]               = set()

    # ------------------------------------------------------------------ #
    # Detected positions
    # ------------------------------------------------------------------ #

    def set_detected(self, channel_id: int, positions: list[float]):
        self.detected_positions[channel_id] = sorted(positions)

    def get_detected(self, channel_id: int) -> list[float]:
        return self.detected_positions.get(channel_id, [])

    def clear_detected(self, channel_id: int = None):
        if channel_id is None:
            self.detected_positions.clear()
        else:
            self.detected_positions.pop(channel_id, None)

    # ------------------------------------------------------------------ #
    # Exclusion
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
        self.global_peaks.append(Peak(adc_position, known_energy, label, auto))

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
                for p in self.global_peaks]
        self.channel_peaks[channel_id].append(Peak(adc_position, known_energy, label))

    def remove_channel_peak(self, channel_id: int, index: int):
        peaks = self.channel_peaks.get(channel_id, [])
        if 0 <= index < len(peaks):
            peaks.pop(index)

    def reset_channel(self, channel_id: int):
        self.channel_peaks.pop(channel_id, None)

    def has_channel_peaks(self, channel_id: int) -> bool:
        return channel_id in self.channel_peaks

    def has_override(self, channel_id: int) -> bool:
        return channel_id in self.channel_peaks

    def reset_channel_to_global(self, channel_id: int):
        self.channel_peaks.pop(channel_id, None)

    # ------------------------------------------------------------------ #
    # Effective peaks
    # ------------------------------------------------------------------ #

    def get_peaks(self, channel_id: int) -> list[Peak]:
        if channel_id in self.channel_peaks:
            return self.channel_peaks[channel_id]
        return self.global_peaks

    def get_calibration_points(self, channel_id: int) -> tuple[np.ndarray, np.ndarray]:
        peaks = self.get_peaks(channel_id)
        if not peaks:
            return np.array([]), np.array([])
        adc = np.array([p.adc_position for p in peaks])
        eng = np.array([p.known_energy  for p in peaks])
        return adc, eng

    def n_calibration_points(self, channel_id: int) -> int:
        return len(self.get_peaks(channel_id))

    # ------------------------------------------------------------------ #
    # Energy propagation
    # ------------------------------------------------------------------ #

    def propagate_energy(self, source_adc, known_energy, label, window,
                          all_channels, source_ch) -> list[int]:
        updated = []
        for ch_id in all_channels:
            if ch_id == source_ch or ch_id in self.excluded_channels:
                continue
            detected = self.detected_positions.get(ch_id, [])
            if not detected:
                continue
            dists = sorted((abs(p - source_adc), p) for p in detected)
            if dists and dists[0][0] <= window:
                best_adc = dists[0][1]
                if ch_id not in self.channel_peaks:
                    self.channel_peaks[ch_id] = []
                tol = window * 0.1
                self.channel_peaks[ch_id] = [
                    p for p in self.channel_peaks[ch_id]
                    if abs(p.adc_position - best_adc) > tol]
                self.channel_peaks[ch_id].append(
                    Peak(adc_position=best_adc, known_energy=known_energy,
                         label=label, auto_detected=True))
                updated.append(ch_id)
        return updated

    # ================================================================== #
    # DETECTION ALGORITHMS
    # ================================================================== #

    # ------------------------------------------------------------------ #
    # 1. TSpectrum (PyROOT)
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
        Detect peaks via ROOT TSpectrum.
        Returns (positions, bg_array).
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

            n        = len(counts)
            counts_f = counts.astype(float)

            devnull_fd   = os.open(os.devnull, os.O_WRONLY)
            saved_stderr = os.dup(2)
            os.dup2(devnull_fd, 2)
            os.close(devnull_fd)

            bg_array = None
            try:
                import threading
                hname = f"_tspec_{threading.get_ident()}_{id(bin_centers)}"
                h = ROOT.TH1F(hname, "", n,
                               float(bin_centers[0]), float(bin_centers[-1]))
                h.SetDirectory(ROOT.nullptr)
                ROOT.SetOwnership(h, True)
                for i, c in enumerate(counts_f):
                    h.SetBinContent(i + 1, c)

                sp = ROOT.TSpectrum(max_peaks)

                if bg_subtract:
                    hbg = sp.Background(h, int(bg_iterations),
                                        "BackDecreasing BackSmoothing3")
                    ROOT.SetOwnership(hbg, True)
                    bg_array = np.array([hbg.GetBinContent(i + 1)
                                         for i in range(n)], dtype=float)
                    counts_sub = np.maximum(counts_f - bg_array, 0.0)
                    for i, c in enumerate(counts_sub):
                        h.SetBinContent(i + 1, c)
                    try:
                        hbg.Delete(); del hbg
                    except Exception:
                        pass
                else:
                    counts_sub = counts_f

                if tspec_mode == "highres":
                    src_arr  = arr.array("d", [float(c) for c in counts_sub])
                    dest_arr = arr.array("d", [0.0] * n)
                    n_found  = sp.SearchHighRes(
                        src_arr, dest_arr, n,
                        float(sigma), float(threshold),
                        False, int(iterations), False, 3)
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
    # 2. scipy fallback
    # ------------------------------------------------------------------ #

    @staticmethod
    def detect_peaks_scipy(bin_centers: np.ndarray,
                            counts: np.ndarray,
                            sigma: float = 2.0,
                            threshold: float = 0.05,
                            max_peaks: int = 10,
                            pedestal_cut: float = 0.0,
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

        positions  = []
        window_w   = max(3, int(len(counts) * 0.01))
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
    # 3. Sliding window peak detection (NEW)
    # ------------------------------------------------------------------ #

    @staticmethod
    def detect_peaks_sliding_window(bin_centers: np.ndarray,
                                     counts: np.ndarray,
                                     window_adc: float = 50.0,
                                     threshold: float = 0.05,
                                     max_peaks: int = 10,
                                     pedestal_cut: float = 0.0,
                                     min_prominence_frac: float = 0.02,
                                     smooth_sigma: float = 1.5,
                                     ) -> list[float]:
        """
        Sliding-window local-maximum peak detector.

        Algorithm
        ---------
        1. Optionally smooth the spectrum with a Gaussian kernel.
        2. Slide a window of width `window_adc` ADC units across the spectrum.
        3. For each window position, the centre bin is a candidate peak if:
               counts[centre] == max(counts in window)          (local max)
           AND counts[centre] >= threshold * global_max          (height cut)
           AND counts[centre] - min(counts in window) >= min_prominence_frac
                                * global_max                     (prominence)
        4. Merge candidates closer than window_adc/2 (keep the taller one).
        5. Return up to max_peaks by prominence, sorted by ADC.

        Parameters
        ----------
        window_adc           : full width of the sliding window in ADC units
        threshold            : minimum height as a fraction of the global max
        min_prominence_frac  : minimum local prominence as a fraction of the global max
        smooth_sigma         : Gaussian smoothing sigma (bins); 0 = no smoothing
        """
        if pedestal_cut > 0:
            mask        = bin_centers >= pedestal_cut
            bin_centers = bin_centers[mask]
            counts      = counts[mask]

        if len(counts) < 3 or counts.sum() == 0:
            return []

        y = counts.astype(float)
        if smooth_sigma > 0:
            y = gaussian_filter1d(y, sigma=smooth_sigma)

        global_max      = float(y.max())
        if global_max <= 0:
            return []

        height_cut      = threshold * global_max
        prom_cut        = min_prominence_frac * global_max

        # ADC step (assume roughly uniform bins)
        adc_step = float(np.median(np.diff(bin_centers))) if len(bin_centers) > 1 else 1.0
        half_win = max(1, int(round(window_adc / 2.0 / adc_step)))

        candidates = []   # (adc, prominence)
        n = len(y)
        for i in range(n):
            lo = max(0, i - half_win)
            hi = min(n, i + half_win + 1)
            window_vals = y[lo:hi]

            if y[i] < height_cut:
                continue
            if y[i] != window_vals.max():   # not the local maximum
                continue

            prominence = float(y[i] - window_vals.min())
            if prominence < prom_cut:
                continue

            candidates.append((float(bin_centers[i]), prominence))

        if not candidates:
            return []

        # Merge candidates within window_adc/2 of each other (keep tallest)
        candidates.sort(key=lambda c: c[1], reverse=True)  # sort by prominence
        merged = []
        used   = set()
        for idx, (adc, prom) in enumerate(candidates):
            if idx in used:
                continue
            cluster = [(adc, prom)]
            for j, (adc2, prom2) in enumerate(candidates):
                if j != idx and j not in used and abs(adc2 - adc) < window_adc / 2:
                    cluster.append((adc2, prom2))
                    used.add(j)
            used.add(idx)
            # keep the highest-prominence peak in the cluster
            best = max(cluster, key=lambda x: x[1])
            merged.append(best)

        # Keep top max_peaks by prominence
        merged.sort(key=lambda x: x[1], reverse=True)
        top = merged[:max_peaks]

        return sorted(adc for adc, _ in top)

    # ------------------------------------------------------------------ #
    # 4. Gaussian confirmation of any detected peaks (NEW)
    # ------------------------------------------------------------------ #

    @staticmethod
    def confirm_peaks_gaussian(bin_centers: np.ndarray,
                                counts: np.ndarray,
                                positions: list[float],
                                window_adc: float = 20.0,
                                ) -> tuple[list[float], list[GaussianFitInfo]]:
        """
        For each candidate position in `positions`, fit a Gaussian + linear
        background in a ±window_adc window.

        Returns
        -------
        accepted  : list of refined centroids (only successful fits)
        fit_infos : list of GaussianFitInfo for every candidate (success or not)
        """
        fit_infos: list[GaussianFitInfo] = []
        accepted: list[float] = []

        for adc in positions:
            info = fit_gaussian_to_peak(bin_centers, counts, adc,
                                         window_adc=window_adc)
            fit_infos.append(info)
            if info.success:
                accepted.append(info.centroid)

        return sorted(accepted), fit_infos

    # ------------------------------------------------------------------ #
    # Main dispatcher
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
                      # Sliding window options
                      detection_method: str = "tspectrum",
                      sw_window_adc: float = 50.0,
                      sw_min_prominence: float = 0.02,
                      sw_smooth_sigma: float = 1.5,
                      # Gaussian confirmation
                      gauss_confirm: bool = False,
                      gauss_window_adc: float = 20.0,
                      ) -> tuple[list[float], np.ndarray | None, list]:
        """
        Unified peak detection dispatcher.

        Parameters
        ----------
        detection_method : "tspectrum" | "sliding_window"
        sw_window_adc    : sliding window full width (ADC units)
        sw_min_prominence: minimum local prominence (fraction of global max)
        sw_smooth_sigma  : pre-smoothing for sliding window (bins)
        gauss_confirm    : if True, confirm every detected peak with a Gaussian fit
        gauss_window_adc : half-width of the Gaussian fit window (ADC units)

        Returns
        -------
        (positions, bg_array, gauss_infos)
          positions   : list[float]         — accepted peak ADC positions
          bg_array    : np.ndarray | None   — SNIP background (TSpectrum only)
          gauss_infos : list[GaussianFitInfo] — one per candidate (empty if
                        gauss_confirm=False)
        """
        bg_array: np.ndarray | None = None
        gauss_infos: list = []

        if detection_method == "sliding_window":
            positions = PeakManager.detect_peaks_sliding_window(
                bin_centers, counts,
                window_adc=sw_window_adc,
                threshold=threshold,
                max_peaks=max_peaks,
                pedestal_cut=pedestal_cut,
                min_prominence_frac=sw_min_prominence,
                smooth_sigma=sw_smooth_sigma)
        else:
            # TSpectrum or scipy fallback
            if backend == "pyroot":
                positions, bg_array = PeakManager.detect_peaks_tspectrum(
                    bin_centers, counts, sigma, threshold, max_peaks,
                    pedestal_cut=pedestal_cut,
                    tspec_mode=tspec_mode, iterations=iterations,
                    bg_subtract=bg_subtract, bg_iterations=bg_iterations)
            else:
                positions = PeakManager.detect_peaks_scipy(
                    bin_centers, counts, sigma, threshold, max_peaks,
                    pedestal_cut=pedestal_cut)

        # Optional Gaussian confirmation
        if gauss_confirm and len(positions) > 0:
            positions, gauss_infos = PeakManager.confirm_peaks_gaussian(
                bin_centers, counts, positions,
                window_adc=gauss_window_adc)

        return positions, bg_array, gauss_infos