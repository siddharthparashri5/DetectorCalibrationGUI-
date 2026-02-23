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
        pedestal_cut : ignore all ADC bins below this value.

        Ownership fix: SetDirectory(0) detaches the histogram from gDirectory
        so ROOT never double-deletes it when Python GC runs.  ROOT stderr is
        suppressed during the call to silence residual TList warnings.
        """
        import sys, os
        try:
            import ROOT

            if pedestal_cut > 0:
                mask        = bin_centers >= pedestal_cut
                bin_centers = bin_centers[mask]
                counts      = counts[mask]

            if len(counts) == 0:
                return []

            n = len(counts)

            # Suppress ROOT stderr noise (TList::Clear warnings) ──────────
            devnull_fd  = os.open(os.devnull, os.O_WRONLY)
            saved_stderr = os.dup(2)
            os.dup2(devnull_fd, 2)
            os.close(devnull_fd)

            try:
                # Use a unique name to avoid gDirectory collisions
                import threading
                hname = f"_tspec_{threading.get_ident()}_{id(bin_centers)}"
                h = ROOT.TH1F(hname, "", n,
                               float(bin_centers[0]),
                               float(bin_centers[-1]))
                # Detach from gDirectory immediately — Python owns this object
                h.SetDirectory(ROOT.nullptr)
                ROOT.SetOwnership(h, True)

                for i, c in enumerate(counts):
                    h.SetBinContent(i + 1, float(c))

                sp      = ROOT.TSpectrum(max_peaks)
                n_found = sp.Search(h, sigma, "nobackground nodraw", threshold)

                positions = []
                px = sp.GetPositionX()
                for i in range(n_found):
                    positions.append(float(px[i]))

                # Copy results before deleting
                result = sorted(positions)

            finally:
                # Restore stderr
                os.dup2(saved_stderr, 2)
                os.close(saved_stderr)
                # Explicitly delete ROOT objects to avoid TList race
                try:
                    del sp
                    h.Delete()
                    del h
                except Exception:
                    pass

            return result

        except Exception:
            return PeakManager.detect_peaks_scipy(
                bin_centers, counts, sigma, threshold, max_peaks,
                pedestal_cut=0.0)

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
