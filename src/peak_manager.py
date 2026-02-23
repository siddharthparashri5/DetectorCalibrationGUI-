"""
PeakManager
===========
Manages peak detection and assignments per channel.

Peak detection:
  - PyROOT backend : ROOT TSpectrum (user-triggered, threshold adjustable)
  - uproot backend : scipy find_peaks (fallback)

Peak assignment:
  - Detected peaks shown on spectrum, user assigns known energies
  - Per-channel override supported
  - Global peaks as default fallback
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional
from scipy.ndimage import gaussian_filter1d


@dataclass
class Peak:
    adc_position:  float
    known_energy:  float
    label:         str  = ""
    auto_detected: bool = False


class PeakManager:
    """
    Stores peak assignments.
    global_peaks  : list[Peak]            — applied to all channels by default
    channel_peaks : dict[int, list[Peak]] — per-channel overrides
    """

    def __init__(self):
        self.global_peaks:  list[Peak]            = []
        self.channel_peaks: dict[int, list[Peak]] = {}

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
    # Per-channel overrides
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

    def reset_channel_to_global(self, channel_id: int):
        self.channel_peaks.pop(channel_id, None)

    def has_override(self, channel_id: int) -> bool:
        return channel_id in self.channel_peaks

    # ------------------------------------------------------------------ #
    # Effective peaks
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
    # Peak detection — TSpectrum (PyROOT)
    # ------------------------------------------------------------------ #

    @staticmethod
    def detect_peaks_tspectrum(bin_centers: np.ndarray,
                                counts: np.ndarray,
                                sigma: float = 2.0,
                                threshold: float = 0.05,
                                max_peaks: int = 10,
                                pedestal_cut: float = 0.0
                                ) -> list[float]:
        """
        Detect peaks using ROOT TSpectrum.
        pedestal_cut : ignore all ADC bins below this value (excludes pedestal).
                       Slices the arrays rather than zeroing, so TSpectrum's
                       relative threshold is computed only over physics peaks.
        """
        try:
            import ROOT
            import array as arr

            # Apply pedestal cut by slicing arrays
            if pedestal_cut > 0:
                mask = bin_centers >= pedestal_cut
                bin_centers = bin_centers[mask]
                counts      = counts[mask]

            if len(counts) == 0:
                return []

            n = len(counts)
            h = ROOT.TH1F("_tspec_tmp", "", n,
                           float(bin_centers[0]),
                           float(bin_centers[-1]))
            for i, c in enumerate(counts):
                h.SetBinContent(i + 1, float(c))

            sp      = ROOT.TSpectrum(max_peaks)
            n_found = sp.Search(h, sigma, "nobackground nodraw", threshold)

            positions = []
            px = sp.GetPositionX()
            for i in range(n_found):
                positions.append(float(px[i]))

            ROOT.gDirectory.Delete("_tspec_tmp")
            return sorted(positions)

        except Exception:
            return PeakManager.detect_peaks_scipy(
                bin_centers, counts, sigma, threshold, max_peaks,
                pedestal_cut=0.0)  # already sliced above

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
        """
        Detect peaks using scipy.signal.find_peaks.
        pedestal_cut : ignore ADC bins below this value.
        """
        from scipy.signal import find_peaks as sp_find_peaks

        # Apply pedestal cut by slicing
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
        window = max(3, int(len(counts) * 0.01))
        for idx in top:
            lo = max(0, idx - window)
            hi = min(len(counts), idx + window + 1)
            w  = counts[lo:hi]
            c  = bin_centers[lo:hi]
            centroid = float(np.sum(c * w) / w.sum()) if w.sum() > 0 \
                       else float(bin_centers[idx])
            positions.append(centroid)

        return sorted(positions)

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
                      pedestal_cut: float = 0.0
                      ) -> list[float]:
        """Dispatch to TSpectrum or scipy depending on active backend."""
        if backend == "pyroot":
            return PeakManager.detect_peaks_tspectrum(
                bin_centers, counts, sigma, threshold, max_peaks,
                pedestal_cut=pedestal_cut)
        else:
            return PeakManager.detect_peaks_scipy(
                bin_centers, counts, sigma, threshold, max_peaks,
                pedestal_cut=pedestal_cut)

    # ------------------------------------------------------------------ #
    # find_peaks_in_windows — used by propagate
    # ------------------------------------------------------------------ #

    @staticmethod
    def find_peaks_in_windows(bin_centers: np.ndarray,
                               counts:      np.ndarray,
                               reference_peaks: list,
                               window:      float,
                               sigma:       float = 2.0,
                               threshold:   float = 0.05,
                               backend:     str   = "pyroot",
                               pedestal_cut: float = 0.0
                               ) -> list:
        """
        For each reference peak, search inside [ref_adc - window, ref_adc + window]
        for the best matching local maximum in this channel's spectrum.

        Returns a list[Peak] with same known_energy as reference peaks but
        ADC positions refined to this channel. Every reference peak gets a
        result — falls back to centroid or reference position if no peak found.
        """
        result = []
        smooth = gaussian_filter1d(counts.astype(float), sigma=max(1, sigma))

        for ref_peak in reference_peaks:
            ref_adc = ref_peak.adc_position
            lo      = ref_adc - window
            hi      = ref_adc + window

            # Slice spectrum to window
            mask = (bin_centers >= lo) & (bin_centers <= hi)
            if mask.sum() < 3:
                # Window too narrow / outside spectrum — keep reference position
                result.append(Peak(
                    adc_position  = ref_adc,
                    known_energy  = ref_peak.known_energy,
                    label         = ref_peak.label,
                    auto_detected = True))
                continue

            win_centers = bin_centers[mask]
            win_counts  = smooth[mask]
            raw_counts  = counts[mask]

            # Try peak detection inside the window
            found_pos = None
            try:
                if backend == "pyroot":
                    candidates = PeakManager.detect_peaks_tspectrum(
                        win_centers, raw_counts,
                        sigma=sigma, threshold=0.01, max_peaks=5,
                        pedestal_cut=0.0)
                else:
                    candidates = PeakManager.detect_peaks_scipy(
                        win_centers, raw_counts,
                        sigma=sigma, threshold=0.01, max_peaks=5,
                        pedestal_cut=0.0)

                if candidates:
                    # Take candidate closest to reference position
                    found_pos = min(candidates, key=lambda x: abs(x - ref_adc))
            except Exception:
                pass

            if found_pos is None:
                # Fallback: centroid around local maximum in window
                idx_max = int(np.argmax(win_counts))
                half_w  = max(2, int(len(win_counts) * 0.1))
                lo_i    = max(0, idx_max - half_w)
                hi_i    = min(len(win_counts), idx_max + half_w + 1)
                w_slice = win_counts[lo_i:hi_i]
                c_slice = win_centers[lo_i:hi_i]
                if w_slice.sum() > 0:
                    found_pos = float(np.sum(c_slice * w_slice) / w_slice.sum())
                else:
                    found_pos = ref_adc

            result.append(Peak(
                adc_position  = found_pos,
                known_energy  = ref_peak.known_energy,
                label         = ref_peak.label,
                auto_detected = True))

        return result
