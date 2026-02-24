"""
CalibSpectrum
=============
Applies energy calibration to raw ADC spectra to produce
calibrated energy spectra (x-axis in keV).

Key fix: for nonlinear models, ADC → energy is evaluated via
FitResult.energy_at() which clamps at adc_max (the last assigned
peak), preventing the super-exponential blow-up that makes the
energy axis garbage beyond the calibration range.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional
from src.calib_fitter import MODELS, model_linear, model_nonlinear, model_nonlinear_3pt


@dataclass
class CalibratedSpectrum:
    channel_id:     int
    energy_centers: np.ndarray
    counts:         np.ndarray
    model:          str
    params:         np.ndarray
    source:         str


class CalibSpectrumEngine:
    def __init__(self):
        self.calib_params: dict = {}   # ch_id -> param dict
        self._fit_results:  dict = {}  # ch_id -> FitResult (kept for energy_at)
        self.source: str = ""

    def load_from_memory(self, fit_results: dict):
        """Load calibration from live FitResult objects.
        Stores the full FitResult so energy_at() (with adc_max clamping)
        can be used during apply()."""
        self.calib_params.clear()
        self._fit_results.clear()
        self.source = "session memory"
        for ch_id, r in fit_results.items():
            if r.success and not r.bad_channel and len(r.params) > 0:
                self.calib_params[ch_id] = {
                    "model":       r.model,
                    "params":      r.params.copy(),
                    "param_names": r.param_names,
                    "adc_max":     float(r.adc_max),
                }
                self._fit_results[ch_id] = r   # keep reference for energy_at

    def load_from_file(self, filepath: str):
        """Load calibration from exported text file (no FitResult available,
        so adc_max is unknown — fall back to direct model evaluation)."""
        self.calib_params.clear()
        self._fit_results.clear()
        self.source = filepath
        param_names = []
        model = "linear"
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("#"):
                    if "Model" in line and "E =" in line:
                        if "P2" in line or "nonlinear" in line.lower():
                            model = "nonlinear"
                        else:
                            model = "linear"
                    if "Channel" in line and ("P0" in line or "p0" in line):
                        parts = line.lstrip("#").split()
                        param_names = []
                        for p in parts[1:]:
                            if p in ("Chi2/NDF", "Status", "Chi2_NDF"):
                                break
                            param_names.append(p)
                    continue
                parts = line.split()
                if len(parts) < 3:
                    continue
                try:
                    ch_id = int(parts[0])
                except ValueError:
                    continue
                if "BAD" in line or "nan" in parts[1].lower():
                    continue
                try:
                    n_params = len(param_names) if param_names else (4 if model == "nonlinear" else 2)
                    params = np.array([float(parts[i + 1]) for i in range(n_params)])
                    self.calib_params[ch_id] = {
                        "model":       model,
                        "params":      params,
                        "param_names": param_names or ["P0","P1","P2","P3"][:n_params],
                        "adc_max":     np.inf,   # unknown from file
                    }
                except (ValueError, IndexError):
                    continue

    def apply(self, channel_id: int,
               bin_centers: np.ndarray,
               counts: np.ndarray,
               out_bins: int = 200,
               e_min: float = 0.0,
               e_max: float = 0.0,
               ) -> Optional[CalibratedSpectrum]:

        if channel_id not in self.calib_params:
            return None

        info    = self.calib_params[channel_id]
        model   = info["model"]
        params  = info["params"]
        adc_max = info.get("adc_max", np.inf)

        # ── Map ADC → energy ─────────────────────────────────────────── #
        # Prefer FitResult.energy_at() which clamps at adc_max and returns
        # NaN beyond it, preventing nonlinear blow-up.
        fit = self._fit_results.get(channel_id)
        if fit is not None:
            energy_centers = fit.energy_at(bin_centers)   # NaN beyond adc_max
        else:
            # File-loaded calibration: evaluate directly but clip manually
            adc_clipped = np.where(bin_centers <= adc_max, bin_centers, np.nan)
            if model == "linear":
                energy_centers = model_linear(adc_clipped, *params)
            elif model == "nonlinear":
                energy_centers = model_nonlinear(adc_clipped, *params)
            elif model == "nonlinear_3pt":
                energy_centers = model_nonlinear_3pt(adc_clipped, *params)
            else:
                return None

        # ── Keep only bins with valid, positive energies ─────────────── #
        mask = np.isfinite(energy_centers) & (energy_centers > 0)
        if mask.sum() < 2:
            return None

        ec = energy_centers[mask]
        ct = counts[mask].astype(float)

        e_lo = e_min if e_min > 0 else float(ec.min())
        e_hi = e_max if e_max > 0 else float(ec.max())
        if e_lo >= e_hi:
            e_lo, e_hi = float(ec.min()), float(ec.max())

        # ── Rebin into uniform energy grid ───────────────────────────── #
        edges      = np.linspace(e_lo, e_hi, out_bins + 1)
        bin_idx    = np.searchsorted(edges, ec) - 1
        new_counts = np.zeros(out_bins)
        for i, idx in enumerate(bin_idx):
            if 0 <= idx < out_bins:
                new_counts[idx] += ct[i]
        new_centers = 0.5 * (edges[:-1] + edges[1:])

        return CalibratedSpectrum(
            channel_id     = channel_id,
            energy_centers = new_centers,
            counts         = new_counts,
            model          = model,
            params         = params,
            source         = self.source,
        )

    def calibrated_channels(self) -> list:
        return sorted(self.calib_params.keys())

    def has_calibration(self, channel_id: int) -> bool:
        return channel_id in self.calib_params
