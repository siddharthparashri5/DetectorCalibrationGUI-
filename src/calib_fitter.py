"""
CalibrationFitter
=================
Supported models:

  linear       (default):  E = P0 + P1·Q
  nonlinear:               E = P0·(P1^Q)^P2 + P3·Q − P0
                           Initial guess: (P0,P1,P2,P3) = (20, 1.10, 1.05, 8.4)
  nonlinear_3pt:           E = P0·P1^Q + P3·Q − P0  (P2=1, for 3-point fits)
                           Initial guess: (P0,P1,P3) = (20, 1.10, 8.4)
  custom:                  user-defined expression, e.g.  a*x**2 + b*x + c

Where Q = raw ADC value.

Bad channel flagging:
  - Fewer calibration points than model parameters  → bad (not enough data)
  - Fit did not converge                            → bad
  - NDF > 0  AND  chi2/NDF > threshold             → bad
  - NDF == 0 (exact interpolation, e.g. 2pts/linear) → NOT bad, just noted
  - Uncertainty > UNC_FRAC_THRESHOLD * |param|     → bad
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable
from scipy.optimize import curve_fit


# ── Models ─────────────────────────────────────────────────────────────── #

def model_linear(Q, P0, P1):
    return P0 + P1 * Q


def model_nonlinear(Q, P0, P1, P2, P3):
    Q    = np.asarray(Q, dtype=float)
    S    = np.power(np.abs(Q), np.clip(P2, 0.1, 10.0))
    base = np.clip(np.abs(P1), 1e-10, 2.0)
    return P0 * np.power(base, S) + P3 * Q - P0


def model_nonlinear_3pt(Q, P0, P1, P3):
    """Constrained nonlinear with P2=1: E = P0*P1^Q + P3*Q - P0
    Used when only 3 calibration points are available."""
    Q    = np.asarray(Q, dtype=float)
    base = np.clip(np.abs(P1), 1e-10, 2.0)
    return P0 * np.power(base, Q) + P3 * Q - P0


MODELS = {
    "linear": {
        "func":        model_linear,
        "param_names": ["P0", "P1"],
        "n_params":    2,
        "label":       "E = P0 + P1·Q",
        "bounds":      (-np.inf, np.inf),
    },
    "nonlinear": {
        "func":        model_nonlinear,
        "param_names": ["P0", "P1", "P2", "P3"],
        "n_params":    4,
        "label":       "E = P0·(P1^Q)^P2 + P3·Q − P0",
        "bounds": (
            [-np.inf, 0.0,  0.1,   0.0   ],
            [ np.inf, 2.0,  10.0,  np.inf]
        ),
    },
    "nonlinear_3pt": {
        "func":        model_nonlinear_3pt,
        "param_names": ["P0", "P1", "P3"],
        "n_params":    3,
        "label":       "E = P0·P1^Q + P3·Q − P0  (P2=1, 3-point)",
        "bounds": (
            [-np.inf, 0.0,   0.0   ],
            [ np.inf, 2.0,   np.inf]
        ),
    },
}


# ── Result ─────────────────────────────────────────────────────────────── #

@dataclass
class FitResult:
    channel_id:    int
    model:         str
    model_label:   str
    params:        np.ndarray
    uncertainties: np.ndarray
    param_names:   list
    chi2:          float
    ndf:           int
    chi2_ndf:      float
    adc_points:    np.ndarray
    energy_points: np.ndarray
    residuals:     np.ndarray
    success:       bool
    bad_channel:   bool = False
    bad_reason:    str  = ""
    note:          str  = ""

    def energy_at(self, Q):
        """Evaluate the calibration model at ADC value(s) Q."""
        Q = np.asarray(Q, dtype=float)
        if self.model == "linear":
            return model_linear(Q, *self.params)
        elif self.model == "nonlinear":
            return model_nonlinear(Q, *self.params)
        elif self.model == "nonlinear_3pt":
            return model_nonlinear_3pt(Q, *self.params)
        elif self.model == "custom" and hasattr(self, "_custom_func"):
            return self._custom_func(Q, *self.params)
        return np.full_like(Q, np.nan)

    def __str__(self):
        lines = [f"Channel {self.channel_id} | {self.model_label}"]
        lines.append(f"Chi2/NDF = {self.chi2_ndf:.4f}  "
                     f"(Chi2={self.chi2:.4f}, NDF={self.ndf})")
        for n, v, u in zip(self.param_names, self.params, self.uncertainties):
            lines.append(f"  {n} = {v:+.8e}  ±  {u:.3e}")
        for adc, eng, res in zip(self.adc_points,
                                  self.energy_points, self.residuals):
            lines.append(f"  ADC={adc:.2f}  E={eng:.3f} keV  "
                         f"res={res:+.4f} keV")
        return "\n".join(lines)


# ── Fitter ─────────────────────────────────────────────────────────────── #

class CalibrationFitter:

    CHI2_NDF_THRESHOLD = 10.0
    UNC_FRAC_THRESHOLD = 5.0

    def __init__(self):
        self.results: dict[int, FitResult] = {}

    # ------------------------------------------------------------------ #
    # Single channel
    # ------------------------------------------------------------------ #

    def fit_channel(self, channel_id: int,
                     adc_points:    np.ndarray,
                     energy_points: np.ndarray,
                     model: str = "linear",
                     custom_expr: str = "") -> FitResult:

        adc_points    = np.asarray(adc_points,    dtype=float)
        energy_points = np.asarray(energy_points, dtype=float)
        n_pts         = len(adc_points)

        # Auto-downgrade nonlinear to 3-point variant when only 3 points
        auto_downgraded = False
        if model == "nonlinear" and n_pts == 3:
            model           = "nonlinear_3pt"
            auto_downgraded = True

        # ── resolve model ────────────────────────────────────────────── #
        if model in MODELS:
            info     = MODELS[model]
            func     = info["func"]
            names    = info["param_names"]
            n_params = info["n_params"]
            label    = info["label"]
            bounds   = info["bounds"]
            custom_f = None
        elif model == "custom":
            if not custom_expr.strip():
                return self._bad(channel_id, "custom",
                                 "Custom expression", "Expression is empty.")
            try:
                func, names, custom_f = self._parse_custom(custom_expr)
                n_params = len(names)
                label    = f"Custom: {custom_expr}"
                bounds   = (-np.inf, np.inf)
            except ValueError as e:
                return self._bad(channel_id, "custom",
                                 f"Custom: {custom_expr}", str(e))
        else:
            return self._bad(channel_id, model, model,
                             f"Unknown model '{model}'.")

        # ── check point count ────────────────────────────────────────── #
        if n_pts < n_params:
            return self._bad(
                channel_id, model, label,
                f"Need ≥{n_params} calibration points for '{model}', "
                f"got {n_pts}.")

        # ── initial guess ────────────────────────────────────────────── #
        p0 = self._initial_guess(model, adc_points, energy_points, n_params)

        # ── fit ──────────────────────────────────────────────────────── #
        try:
            popt, pcov = curve_fit(func, adc_points, energy_points,
                                    p0=p0, bounds=bounds,
                                    maxfev=100000, ftol=1e-12, xtol=1e-12)
            perr = np.sqrt(np.abs(np.diag(pcov)))
        except Exception as e:
            return self._bad(channel_id, model, label,
                             f"Fit failed: {e}")

        # ── residuals & chi2 ─────────────────────────────────────────── #
        fitted    = func(adc_points, *popt)
        residuals = energy_points - fitted
        chi2      = float(np.sum(residuals ** 2))
        ndf       = n_pts - n_params

        chi2_ndf = chi2 / ndf if ndf > 0 else float("nan")

        bad, reason, note = False, "", ""

        if ndf == 0:
            note = (f"Exact interpolation ({n_pts} pts = {n_params} params). "
                    "Add more peaks for a χ²/NDF estimate.")
        elif np.any(np.isnan(perr)):
            bad    = True
            reason = "Covariance matrix has NaN (fit unstable — try a different model or more peaks)"

        if auto_downgraded:
            downgrade_note = ("Auto-selected 3-point nonlinear (P2=1) — "
                              "add a 4th peak for full nonlinear fit.")
            note = (note + "  " + downgrade_note).strip()

        # ── build result ─────────────────────────────────────────────── #
        result = FitResult(
            channel_id=channel_id, model=model, model_label=label,
            params=popt, uncertainties=perr, param_names=list(names),
            chi2=chi2, ndf=ndf, chi2_ndf=chi2_ndf,
            adc_points=adc_points, energy_points=energy_points,
            residuals=residuals,
            success=True, bad_channel=bad, bad_reason=reason, note=note
        )

        if model == "custom" and custom_f is not None:
            result._custom_func = custom_f

        self.results[channel_id] = result
        return result

    # ------------------------------------------------------------------ #
    # Batch
    # ------------------------------------------------------------------ #

    def fit_all(self, channel_ids: list,
                 peak_manager,
                 model:       str = "linear",
                 custom_expr: str = "",
                 progress_callback=None) -> dict:
        total = len(channel_ids)
        for i, ch_id in enumerate(channel_ids):
            adc, eng = peak_manager.get_calibration_points(ch_id)
            if len(adc) == 0:
                self.results[ch_id] = self._bad(
                    ch_id, model,
                    MODELS.get(model, {}).get("label", model),
                    "No calibration peaks defined.")
            else:
                self.fit_channel(ch_id, adc, eng, model, custom_expr)
            if progress_callback:
                progress_callback(i + 1, total)
        return self.results

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _initial_guess(model, adc, eng, n_params):
        dQ    = float(adc[-1] - adc[0]) + 1e-9
        dE    = float(eng[-1] - eng[0])
        slope = dE / dQ
        inter = float(eng[0]) - slope * float(adc[0])

        if model == "linear":
            return [inter, max(slope, 1e-6)]
        elif model == "nonlinear":
            # Best-known starting point for GAGG/LYSO detectors:
            # E = P0*(P1^Q)^P2 + P3*Q - P0
            # (P0, P1, P2, P3) = (20, 1.10, 1.05, 8.4)
            return [20.0, 1.10, 1.05, 8.4]
        elif model == "nonlinear_3pt":
            # 3-point variant with P2=1: E = P0*P1^Q + P3*Q - P0
            return [20.0, 1.10, 8.4]
        else:
            p0     = [1.0] * n_params
            p0[-1] = slope
            return p0

    @staticmethod
    def _parse_custom(expr: str):
        import re
        names = sorted(set(re.findall(r'\b([a-wyzA-WYZ])\b', expr)))
        if not names:
            raise ValueError(
                "No parameters found. Use single letters (not x) "
                "e.g.  a*x**2 + b*x + c")

        def func(x, *args):
            local = {"x": x, "np": np, "exp": np.exp,
                     "log": np.log, "sqrt": np.sqrt,
                     "abs": np.abs, "sin": np.sin, "cos": np.cos}
            for name, val in zip(names, args):
                local[name] = val
            return eval(expr, {"__builtins__": {}}, local)

        func.__name__ = "custom"
        return func, names, func

    @staticmethod
    def _bad(channel_id, model, label, reason) -> FitResult:
        return FitResult(
            channel_id=channel_id, model=model, model_label=label,
            params=np.array([]), uncertainties=np.array([]),
            param_names=MODELS.get(model, {}).get("param_names", []),
            chi2=float("nan"), ndf=0, chi2_ndf=float("nan"),
            adc_points=np.array([]), energy_points=np.array([]),
            residuals=np.array([]),
            success=False, bad_channel=True, bad_reason=reason, note=""
        )

    def bad_channels(self) -> list:
        return [ch for ch, r in self.results.items() if r.bad_channel]

    def good_channels(self) -> list:
        return [ch for ch, r in self.results.items()
                if not r.bad_channel and r.success]
