"""
CalibSpectrum
=============
Applies energy calibration to raw ADC spectra to produce
calibrated energy spectra (x-axis in keV).

Fix: the calibrated spectrum now shows the FULL spectrum range,
not clipped at adc_max. The adc_max concept was originally used
to prevent nonlinear model blow-up beyond the last calibration
point, but this cut the spectrum short in the display. We now
use a gentler approach: evaluate energy_at() for the full ADC
range and only drop bins where the energy value is physically
invalid (NaN, negative, or wildly extrapolated by the nonlinear
model).
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
        self._fit_results:  dict = {}  # ch_id -> FitResult
        self.source: str = ""

    def load_from_memory(self, fit_results: dict):
        """Load calibration from live FitResult objects."""
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
                self._fit_results[ch_id] = r

    def load_from_file(self, filepath: str):
        """Load calibration from exported text file."""
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
                        "adc_max":     np.inf,
                    }
                except (ValueError, IndexError):
                    continue

    def apply(self, channel_id: int,
               bin_centers: np.ndarray,
               counts: np.ndarray,
               out_bins: int = 1024,
               e_min: float = 0.0,
               e_max: float = 0.0,
               ) -> Optional[CalibratedSpectrum]:

        if channel_id not in self.calib_params:
            return None

        info   = self.calib_params[channel_id]
        model  = info["model"]
        params = info["params"]

        # ── Map ADC → energy over the FULL ADC range ─────────────────── #
        # We evaluate the model for ALL bins, not just up to adc_max.
        # For nonlinear models beyond adc_max the exponential can blow up,
        # so we cap energy values at a physically reasonable upper limit
        # (5× the energy at adc_max) rather than blanking them entirely.
        fit = self._fit_results.get(channel_id)
        if fit is not None:
            # Evaluate model directly (not energy_at which clips at adc_max)
            adc_max = float(fit.adc_max)
            Q = bin_centers.astype(float)

            if fit.model == "linear":
                energy_centers = model_linear(Q, *fit.params)
            elif fit.model == "nonlinear":
                energy_centers = model_nonlinear(Q, *fit.params)
            elif fit.model == "nonlinear_3pt":
                energy_centers = model_nonlinear_3pt(Q, *fit.params)
            elif fit.model == "custom" and hasattr(fit, "_custom_func"):
                try:
                    energy_centers = fit._custom_func(Q, *fit.params)
                except Exception:
                    energy_centers = np.full_like(Q, np.nan)
            else:
                energy_centers = np.full_like(Q, np.nan)

            # For nonlinear models: cap extrapolation beyond adc_max to
            # avoid super-exponential blow-up ruining the display.
            # Cap at 3× the energy value at adc_max.
            if fit.model in ("nonlinear", "nonlinear_3pt") and np.isfinite(adc_max):
                e_at_max = fit.energy_at(np.array([adc_max]))[0]
                if np.isfinite(e_at_max) and e_at_max > 0:
                    energy_cap = 3.0 * e_at_max
                    energy_centers = np.where(
                        energy_centers <= energy_cap,
                        energy_centers, np.nan)

        else:
            # File-loaded calibration: evaluate directly, no capping
            Q = bin_centers.astype(float)
            if model == "linear":
                energy_centers = model_linear(Q, *params)
            elif model == "nonlinear":
                energy_centers = model_nonlinear(Q, *params)
            elif model == "nonlinear_3pt":
                energy_centers = model_nonlinear_3pt(Q, *params)
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