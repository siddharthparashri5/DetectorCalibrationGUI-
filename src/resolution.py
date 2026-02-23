"""
ResolutionCalculator
====================
Background models (physically motivated for scintillator spectra):
  linear      : A·G + B + C·E
  quadratic   : A·G + B + C·E + D·E²
  exponential : A·G + B·exp(C·E)       (C ≤ 0 — Compton tail)
  step        : A·G + B·erfc + C + D·E (Compton step — most physical)
  none        : Gaussian only
  auto        : try all, pick lowest chi²/NDF
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from scipy.optimize import curve_fit
from scipy.special import erfc


def _gauss(E, A, mu, sigma):
    return A * np.exp(-0.5 * ((E - mu) / sigma) ** 2)

def gaussian_only(E, A, mu, sigma):
    return _gauss(E, A, mu, sigma)

def gaussian_linear_bg(E, A, mu, sigma, B, C):
    return _gauss(E, A, mu, sigma) + B + C * E

def gaussian_quadratic_bg(E, A, mu, sigma, B, C, D):
    return _gauss(E, A, mu, sigma) + B + C * E + D * E**2

def gaussian_exp_bg(E, A, mu, sigma, B, C):
    return _gauss(E, A, mu, sigma) + B * np.exp(C * E)

def gaussian_step_bg(E, A, mu, sigma, B, C, D):
    """Gaussian + Compton step (erfc) + linear offset.
    B*erfc((E-mu)/(sqrt(2)*sigma)) models the background from
    multiply-scattered photons under the photopeak."""
    step = B * erfc((E - mu) / (np.sqrt(2.0) * (np.abs(sigma) + 1e-9)))
    return _gauss(E, A, mu, sigma) + step + C + D * E


# Model registry: p0_extra, blo_extra, bhi_extra are lambdas(A0,mu0,sig0,B0,dE)
# and lambdas(e_lo, e_hi) respectively for the extra BG params only.
BG_MODELS = {
    "none": dict(
        func      = gaussian_only,
        label     = "Gauss only",
        p0_extra  = lambda A0, mu0, sig0, B0, dE: [],
        blo_extra = lambda lo, hi: [],
        bhi_extra = lambda lo, hi: [],
    ),
    "linear": dict(
        func      = gaussian_linear_bg,
        label     = "Gauss + linear (B + C·E)",
        p0_extra  = lambda A0, mu0, sig0, B0, dE: [max(B0, 0.0), 0.0],
        blo_extra = lambda lo, hi: [0.0,     -np.inf],
        bhi_extra = lambda lo, hi: [np.inf,   np.inf],
    ),
    "quadratic": dict(
        func      = gaussian_quadratic_bg,
        label     = "Gauss + quadratic (B + C·E + D·E²)",
        p0_extra  = lambda A0, mu0, sig0, B0, dE: [max(B0, 0.0), 0.0, 0.0],
        blo_extra = lambda lo, hi: [0.0,     -np.inf, -np.inf],
        bhi_extra = lambda lo, hi: [np.inf,   np.inf,  np.inf],
    ),
    "exponential": dict(
        func      = gaussian_exp_bg,
        label     = "Gauss + exponential (B·exp(C·E))",
        p0_extra  = lambda A0, mu0, sig0, B0, dE: [max(B0, 1.0),
                                                    -1.0 / max(mu0, 1.0)],
        blo_extra = lambda lo, hi: [0.0,    -np.inf],
        bhi_extra = lambda lo, hi: [np.inf,  0.0  ],   # C ≤ 0 (decay)
    ),
    "step": dict(
        func      = gaussian_step_bg,
        label     = "Gauss + Compton step (B·erfc + C + D·E)",
        p0_extra  = lambda A0, mu0, sig0, B0, dE: [A0 * 0.05,
                                                    max(B0 * 0.5, 0.0),
                                                    0.0],
        blo_extra = lambda lo, hi: [0.0,    -np.inf, -np.inf],
        bhi_extra = lambda lo, hi: [np.inf,  np.inf,  np.inf],
    ),
}

_AUTO_ORDER = ("step", "exponential", "quadratic", "linear", "none")


@dataclass
class ResolutionResult:
    channel_id:     int
    peak_energy:    float
    fit_range:      tuple
    mu:             float
    sigma:          float
    mu_err:         float
    sigma_err:      float
    amplitude:      float
    fwhm:           float
    fwhm_err:       float
    resolution:     float
    resolution_err: float
    chi2_ndf:       float
    success:        bool
    fail_reason:    str   = ""
    bg_model:       str   = "step"
    bg_params:      tuple = field(default_factory=tuple)

    @property
    def fwhm_keV(self):
        return f"{self.fwhm:.3f} \u00b1 {self.fwhm_err:.3f} keV"

    @property
    def resolution_str(self):
        return f"{self.resolution:.2f} \u00b1 {self.resolution_err:.2f} %"


class ResolutionCalculator:

    def __init__(self):
        self.results: dict = {}

    def fit_peak(self, channel_id, peak_label, energy_arr, counts_arr,
                  e_lo, e_hi, bg_model: str = "step"):
        mask = (energy_arr >= e_lo) & (energy_arr <= e_hi)
        E    = energy_arr[mask]
        C    = counts_arr[mask].astype(float)

        def _bad(reason):
            return ResolutionResult(
                channel_id=channel_id, peak_energy=peak_label,
                fit_range=(e_lo, e_hi),
                mu=float("nan"), sigma=float("nan"),
                mu_err=float("nan"), sigma_err=float("nan"),
                amplitude=float("nan"),
                fwhm=float("nan"), fwhm_err=float("nan"),
                resolution=float("nan"), resolution_err=float("nan"),
                chi2_ndf=float("nan"), success=False,
                fail_reason=reason, bg_model=bg_model)

        if len(E) < 6:
            return _bad("Fit range has fewer than 6 bins.")

        A0   = float(C.max())
        mu0  = float(E[np.argmax(C)])
        sig0 = (e_hi - e_lo) / 6.0
        B0   = float(np.percentile(C, 10))
        dE   = e_hi - e_lo

        if bg_model == "auto":
            return self._fit_auto(channel_id, peak_label, E, C,
                                   e_lo, e_hi, A0, mu0, sig0, B0, dE, _bad)
        return self._fit_model(channel_id, peak_label, E, C,
                                e_lo, e_hi, A0, mu0, sig0, B0, dE,
                                bg_model, _bad)

    def _fit_model(self, channel_id, peak_label, E, C,
                    e_lo, e_hi, A0, mu0, sig0, B0, dE, bg_model, _bad):
        if bg_model not in BG_MODELS:
            return _bad(f"Unknown model '{bg_model}'.")
        m    = BG_MODELS[bg_model]
        func = m["func"]
        p0   = [A0, mu0, sig0] + m["p0_extra"](A0, mu0, sig0, B0, dE)
        blo  = [0.0, e_lo, 1e-9] + m["blo_extra"](e_lo, e_hi)
        bhi  = [np.inf, e_hi, (e_hi - e_lo)] + m["bhi_extra"](e_lo, e_hi)

        try:
            popt, pcov = curve_fit(func, E, C, p0=p0, bounds=(blo, bhi),
                                    maxfev=50000, ftol=1e-12, xtol=1e-12)
            perr = np.sqrt(np.abs(np.diag(pcov)))
        except Exception as exc:
            return _bad(f"[{bg_model}] {exc}")

        A, mu, sigma   = popt[0], popt[1], popt[2]
        mu_err, sig_err = perr[1], perr[2]

        if np.isnan(sigma) or sigma <= 0:
            return _bad(f"Invalid sigma from [{bg_model}].")
        if np.any(np.isnan(perr)):
            return _bad(f"Covariance NaN in [{bg_model}] — fit unstable.")

        fitted    = func(E, *popt)
        residuals = C - fitted
        ndf       = max(1, len(E) - len(popt))
        chi2_ndf  = float(np.sum(residuals**2) / ndf)
        fwhm      = 2.3548 * abs(sigma)
        fwhm_err  = 2.3548 * sig_err
        res       = fwhm / abs(mu) * 100.0 if abs(mu) > 1e-9 else float("nan")
        res_err   = (res * np.sqrt((fwhm_err/fwhm)**2 + (mu_err/mu)**2)
                     if (abs(mu) > 1e-9 and fwhm > 1e-9) else float("nan"))

        result = ResolutionResult(
            channel_id=channel_id, peak_energy=peak_label,
            fit_range=(e_lo, e_hi),
            mu=float(mu), sigma=float(abs(sigma)),
            mu_err=float(mu_err), sigma_err=float(sig_err),
            amplitude=float(A),
            fwhm=fwhm, fwhm_err=fwhm_err,
            resolution=res, resolution_err=res_err,
            chi2_ndf=chi2_ndf, success=True,
            bg_model=bg_model, bg_params=tuple(float(v) for v in popt[3:]),
        )
        self.results[(channel_id, peak_label)] = result
        return result

    def _fit_auto(self, channel_id, peak_label, E, C,
                   e_lo, e_hi, A0, mu0, sig0, B0, dE, _bad):
        best, best_chi2 = None, np.inf
        for name in _AUTO_ORDER:
            r = self._fit_model(channel_id, peak_label, E, C,
                                 e_lo, e_hi, A0, mu0, sig0, B0, dE, name, _bad)
            if r.success and not np.isnan(r.chi2_ndf) and r.chi2_ndf < best_chi2:
                best_chi2, best = r.chi2_ndf, r
        if best is None:
            return _bad("All background models failed.")
        self.results[(channel_id, peak_label)] = best
        return best

    def fwhm_trend(self, peak_label):
        rows = sorted(
            [(ch, r) for (ch, pk), r in self.results.items()
             if pk == peak_label and r.success and not np.isnan(r.fwhm)],
            key=lambda x: x[0])
        if not rows:
            return np.array([]), np.array([]), np.array([])
        return (np.array([r[0] for r in rows]),
                np.array([r[1].fwhm     for r in rows]),
                np.array([r[1].fwhm_err for r in rows]))

    def resolution_trend(self, peak_label):
        rows = sorted(
            [(ch, r) for (ch, pk), r in self.results.items()
             if pk == peak_label and r.success and not np.isnan(r.resolution)],
            key=lambda x: x[0])
        if not rows:
            return np.array([]), np.array([]), np.array([])
        return (np.array([r[0] for r in rows]),
                np.array([r[1].resolution     for r in rows]),
                np.array([r[1].resolution_err for r in rows]))

    def peak_labels(self):
        return sorted(set(pk for _, pk in self.results.keys()))

    def export(self, filepath, source_file=""):
        from datetime import datetime
        with open(filepath, "w") as f:
            f.write("# ============================================================\n")
            f.write("# DetectorCalibGUI — Energy Resolution Results\n")
            f.write(f"# Date   : {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}\n")
            if source_file:
                f.write(f"# Source : {source_file}\n")
            f.write("# FWHM = 2.3548 * sigma  |  R% = FWHM/centroid * 100\n#\n")
            col = 14
            hdr = (f"{'Channel':>8s}  {'Peak(keV)':>{col}s}  "
                   f"{'Centroid(keV)':>{col}s}  {'FWHM(keV)':>{col}s}  "
                   f"{'FWHM_err':>{col}s}  {'R%':>{col}s}  "
                   f"{'R%_err':>{col}s}  {'Chi2/NDF':>{col}s}  "
                   f"{'BG_model':>14s}  Status\n")
            f.write("# " + hdr)
            for (ch_id, pk), r in sorted(self.results.items()):
                if r.success:
                    f.write(f"  {ch_id:>8d}  {pk:>{col}.2f}  "
                            f"{r.mu:>{col}.4f}  {r.fwhm:>{col}.4f}  "
                            f"{r.fwhm_err:>{col}.4f}  {r.resolution:>{col}.4f}  "
                            f"{r.resolution_err:>{col}.4f}  {r.chi2_ndf:>{col}.4f}  "
                            f"{r.bg_model:>14s}  OK\n")
                else:
                    blank = "  ".join(["nan".rjust(col)] * 6)
                    f.write(f"  {ch_id:>8d}  {pk:>{col}.2f}  "
                            f"{blank}  {'':>14s}  FAIL: {r.fail_reason[:40]}\n")
