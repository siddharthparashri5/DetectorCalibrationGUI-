"""
CalibSpectrum
=============
Applies energy calibration to raw ADC spectra and produces
calibrated energy spectra (x-axis in keV).

Sources of calibration:
  1. Current session memory  (FitResult dict from CalibrationFitter)
  2. calibration_coeffs.txt  (exported by OutputWriter)
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional
from src.calib_fitter import MODELS, model_linear, model_nonlinear


@dataclass
class CalibratedSpectrum:
    channel_id:   int
    energy_centers: np.ndarray   # keV
    counts:         np.ndarray
    model:          str
    params:         np.ndarray
    source:         str          # "memory" or filename


class CalibSpectrumEngine:
    """
    Converts raw ADC spectra → calibrated energy spectra
    using stored calibration parameters.
    """

    def __init__(self):
        self.calib_params: dict[int, dict] = {}
        # {ch_id: {"model": str, "params": np.ndarray, "param_names": list}}
        self.source: str = ""

    # ------------------------------------------------------------------ #
    # Load calibration
    # ------------------------------------------------------------------ #

    def load_from_memory(self, fit_results: dict):
        """Load calibration from current session FitResult dict."""
        self.calib_params.clear()
        self.source = "session memory"
        for ch_id, r in fit_results.items():
            if r.success and not r.bad_channel and len(r.params) > 0:
                self.calib_params[ch_id] = {
                    "model":       r.model,
                    "params":      r.params.copy(),
                    "param_names": r.param_names,
                }

    def load_from_file(self, filepath: str):
        """
        Load calibration_coeffs.txt produced by OutputWriter.
        Parses the compact table: Channel  P0  P1  [P2  P3]  Chi2/NDF  Status
        """
        self.calib_params.clear()
        self.source = filepath

        param_names: list[str] = []
        model = "linear"

        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Parse header comment lines
                if line.startswith("#"):
                    # Detect model
                    if "Model" in line and "E =" in line:
                        if "P2" in line or "nonlinear" in line.lower():
                            model = "nonlinear"
                        else:
                            model = "linear"
                    # Parse column header to get param names
                    if "Channel" in line and ("P0" in line or "p0" in line):
                        parts = line.lstrip("#").split()
                        # parts: Channel P0 P1 [P2 P3] Chi2/NDF Status
                        param_names = []
                        for p in parts[1:]:
                            if p in ("Chi2/NDF", "Status", "Chi2_NDF"):
                                break
                            param_names.append(p)
                    continue

                # Data lines
                parts = line.split()
                if len(parts) < 3:
                    continue
                try:
                    ch_id = int(parts[0])
                except ValueError:
                    continue

                # Skip bad channels
                if "BAD" in line or "nan" in parts[1].lower():
                    continue

                try:
                    n_params = len(param_names) if param_names else \
                               (4 if model == "nonlinear" else 2)
                    params = np.array([float(parts[i + 1])
                                       for i in range(n_params)])
                    self.calib_params[ch_id] = {
                        "model":       model,
                        "params":      params,
                        "param_names": param_names or
                                       ["P0", "P1", "P2", "P3"][:n_params],
                    }
                except (ValueError, IndexError):
                    continue

    # ------------------------------------------------------------------ #
    # Apply calibration
    # ------------------------------------------------------------------ #

    def apply(self, channel_id: int,
               bin_centers: np.ndarray,
               counts: np.ndarray,
               out_bins: int = 1024,
               e_min: float = 0.0,
               e_max: float = 0.0) -> Optional[CalibratedSpectrum]:
        """
        Apply calibration to a raw ADC spectrum.
        Returns a CalibratedSpectrum with energy axis, or None if no
        calibration available for this channel.
        """
        if channel_id not in self.calib_params:
            return None

        info   = self.calib_params[channel_id]
        model  = info["model"]
        params = info["params"]

        # Convert all bin centers to energy
        if model == "linear":
            energy_centers = model_linear(bin_centers, *params)
        elif model == "nonlinear":
            energy_centers = model_nonlinear(bin_centers, *params)
        else:
            return None

        # Keep only physical (positive) energies
        mask = energy_centers > 0
        if mask.sum() < 2:
            return None
        ec = energy_centers[mask]
        ct = counts[mask]

        # Re-bin onto uniform energy grid
        e_lo = e_min if e_min > 0 else float(ec.min())
        e_hi = e_max if e_max > 0 else float(ec.max())
        if e_lo >= e_hi:
            e_lo, e_hi = float(ec.min()), float(ec.max())

        edges      = np.linspace(e_lo, e_hi, out_bins + 1)
        bin_idx    = np.searchsorted(edges, ec) - 1
        new_counts = np.zeros(out_bins)
        for i, idx in enumerate(bin_idx):
            if 0 <= idx < out_bins:
                new_counts[idx] += ct[i]

        new_centers = 0.5 * (edges[:-1] + edges[1:])

        return CalibratedSpectrum(
            channel_id    = channel_id,
            energy_centers= new_centers,
            counts        = new_counts,
            model         = model,
            params        = params,
            source        = self.source,
        )

    def calibrated_channels(self) -> list[int]:
        return sorted(self.calib_params.keys())

    def has_calibration(self, channel_id: int) -> bool:
        return channel_id in self.calib_params
