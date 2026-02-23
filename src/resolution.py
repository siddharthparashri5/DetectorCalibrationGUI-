"""
ResolutionCalculator
====================
Fits Gaussian peaks on calibrated spectra to extract energy resolution.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional
from scipy.optimize import curve_fit


def gaussian_linear_bg(E, A, mu, sigma, B, C):
    return A * np.exp(-0.5 * ((E - mu) / sigma) ** 2) + B + C * E


def gaussian_only(E, A, mu, sigma):
    return A * np.exp(-0.5 * ((E - mu) / sigma) ** 2)


@dataclass
class ResolutionResult:
    channel_id:   int
    peak_energy:  float
    fit_range:    tuple
    mu:           float
    sigma:        float
    mu_err:       float
    sigma_err:    float
    amplitude:    float
    fwhm:         float
    fwhm_err:     float
    resolution:   float
    resolution_err: float
    chi2_ndf:     float
    success:      bool
    fail_reason:  str = ""

    @property
    def fwhm_keV(self) -> str:
        return f"{self.fwhm:.3f} ± {self.fwhm_err:.3f} keV"

    @property
    def resolution_str(self) -> str:
        return f"{self.resolution:.2f} ± {self.resolution_err:.2f} %"


class ResolutionCalculator:

    def __init__(self):
        self.results: dict = {}

    def fit_peak(self, channel_id, peak_label, energy_arr, counts_arr,
                  e_lo, e_hi, use_bg=True):
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
                chi2_ndf=float("nan"),
                success=False, fail_reason=reason)

        if len(E) < 5:
            return _bad("Fit range contains fewer than 5 bins.")

        A0   = float(C.max())
        mu0  = float(E[np.argmax(C)])
        sig0 = (e_hi - e_lo) / 6.0
        B0   = float(C.min())

        try:
            if use_bg:
                p0     = [A0, mu0, sig0, B0, 0.0]
                bounds = ([0, e_lo, 0, 0, -np.inf],
                          [np.inf, e_hi, (e_hi-e_lo), np.inf, np.inf])
                popt, pcov = curve_fit(gaussian_linear_bg, E, C,
                                        p0=p0, bounds=bounds, maxfev=20000)
                A, mu, sigma, B, Clin = popt
                perr = np.sqrt(np.abs(np.diag(pcov)))
                mu_err, sigma_err = perr[1], perr[2]
                fitted = gaussian_linear_bg(E, *popt)
            else:
                p0     = [A0, mu0, sig0]
                bounds = ([0, e_lo, 0], [np.inf, e_hi, (e_hi-e_lo)])
                popt, pcov = curve_fit(gaussian_only, E, C,
                                        p0=p0, bounds=bounds, maxfev=20000)
                A, mu, sigma = popt
                perr = np.sqrt(np.abs(np.diag(pcov)))
                mu_err, sigma_err = perr[1], perr[2]
                fitted = gaussian_only(E, *popt)
        except Exception as e:
            return _bad(f"Fit failed: {e}")

        if np.isnan(sigma) or sigma <= 0:
            return _bad("Invalid sigma from fit.")

        residuals = C - fitted
        ndf       = max(1, len(E) - len(popt))
        chi2_ndf  = float(np.sum(residuals**2) / ndf)
        fwhm      = 2.3548 * abs(sigma)
        fwhm_err  = 2.3548 * sigma_err
        res       = fwhm / abs(mu) * 100.0 if abs(mu) > 1e-9 else float("nan")
        res_err   = res * np.sqrt((fwhm_err/fwhm)**2 +
                                   (mu_err/mu)**2) if abs(mu) > 1e-9 else float("nan")

        result = ResolutionResult(
            channel_id=channel_id, peak_energy=peak_label,
            fit_range=(e_lo, e_hi),
            mu=float(mu), sigma=float(abs(sigma)),
            mu_err=float(mu_err), sigma_err=float(sigma_err),
            amplitude=float(A),
            fwhm=fwhm, fwhm_err=fwhm_err,
            resolution=res, resolution_err=res_err,
            chi2_ndf=chi2_ndf, success=True
        )
        self.results[(channel_id, peak_label)] = result
        return result

    def fwhm_trend(self, peak_label):
        rows = [(ch, r) for (ch, pk), r in self.results.items()
                if pk == peak_label and r.success and not np.isnan(r.fwhm)]
        if not rows:
            return np.array([]), np.array([]), np.array([])
        rows.sort(key=lambda x: x[0])
        channels = np.array([r[0]          for r in rows])
        fwhms    = np.array([r[1].fwhm     for r in rows])
        errs     = np.array([r[1].fwhm_err for r in rows])
        return channels, fwhms, errs

    def resolution_trend(self, peak_label):
        rows = [(ch, r) for (ch, pk), r in self.results.items()
                if pk == peak_label and r.success and not np.isnan(r.resolution)]
        if not rows:
            return np.array([]), np.array([]), np.array([])
        rows.sort(key=lambda x: x[0])
        channels = np.array([r[0]                for r in rows])
        res      = np.array([r[1].resolution     for r in rows])
        errs     = np.array([r[1].resolution_err for r in rows])
        return channels, res, errs

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
            f.write("# FWHM   = 2.3548 * sigma\n")
            f.write("# R%     = FWHM / centroid * 100\n#\n")
            col = 14
            hdr = (f"{{'Channel':>8s}}  {{'Peak(keV)':>{col}s}}  "
                   f"{{'Centroid(keV)':>{col}s}}  {{'FWHM(keV)':>{col}s}}  "
                   f"{{'FWHM_err':>{col}s}}  {{'R%':>{col}s}}  "
                   f"{{'R%_err':>{col}s}}  {{'Chi2/NDF':>{col}s}}  Status\n")
            f.write(f"# {hdr}")
            for (ch_id, pk), r in sorted(self.results.items()):
                if r.success:
                    row = (f"  {ch_id:>8d}  {pk:>{col}.2f}  "
                           f"{r.mu:>{col}.4f}  {r.fwhm:>{col}.4f}  "
                           f"{r.fwhm_err:>{col}.4f}  {r.resolution:>{col}.4f}  "
                           f"{r.resolution_err:>{col}.4f}  {r.chi2_ndf:>{col}.4f}  OK\n")
                else:
                    row = (f"  {ch_id:>8d}  {pk:>{col}.2f}  "
                           + "  ".join(["nan".rjust(col)] * 6)
                           + f"  FAIL: {r.fail_reason[:30]}\n")
                f.write(row)
