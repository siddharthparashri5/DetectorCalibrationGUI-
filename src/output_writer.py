"""
OutputWriter
============
Exports calibration results.
"""

from __future__ import annotations
import os
from datetime import datetime
from src.calib_fitter import FitResult


class OutputWriter:

    HEADER = (
        "# ============================================================\n"
        "# DetectorCalibGUI — Energy Calibration Results\n"
        "# Author: Siddharth Parashari\n"
        "# ============================================================\n"
    )

    @staticmethod
    def _ts() -> str:
        return datetime.now().strftime("%Y-%m-%d  %H:%M:%S")

    @classmethod
    def write_single(cls, result: FitResult, out_path: str,
                      source_file: str = ""):
        with open(out_path, "w") as f:
            f.write(cls.HEADER)
            f.write(f"# Date          : {cls._ts()}\n")
            if source_file:
                f.write(f"# Source file   : {source_file}\n")
            f.write(f"# Channel       : {result.channel_id}\n")
            f.write(f"# Model         : {result.model_label}\n#\n")
            if result.bad_channel:
                f.write(f"# STATUS: BAD — {result.bad_reason}\n")
                return
            f.write(f"# Chi2/NDF      : {result.chi2_ndf:.6f}\n")
            f.write(f"# Chi2          : {result.chi2:.4f}\n")
            f.write(f"# NDF           : {result.ndf}\n#\n")
            f.write("# Parameters:\n")
            for name, val, unc in zip(result.param_names,
                                       result.params, result.uncertainties):
                f.write(f"#   {name} = {val:+.8e}  ±  {unc:.3e}\n")
            f.write("#\n")
            f.write(f"# {'ADC_position':>14s}  {'Energy_keV':>12s}  {'Residual_keV':>14s}\n")
            for adc, eng, res in zip(result.adc_points,
                                      result.energy_points, result.residuals):
                f.write(f"  {adc:>14.4f}  {eng:>12.3f}  {res:>+14.5f}\n")

    @classmethod
    def write_log(cls, results: dict, out_path: str, source_file: str = ""):
        good = [r for r in results.values() if not r.bad_channel]
        bad  = [r for r in results.values() if r.bad_channel]
        with open(out_path, "w") as f:
            f.write(cls.HEADER)
            f.write(f"# Date          : {cls._ts()}\n")
            if source_file:
                f.write(f"# Source file   : {source_file}\n")
            f.write(f"# Total channels: {len(results)}\n")
            f.write(f"# Good channels : {len(good)}\n")
            f.write(f"# Bad channels  : {len(bad)}\n")
            if bad:
                f.write("# Bad IDs       : "
                        + ", ".join(str(r.channel_id) for r in bad) + "\n")
            f.write("#\n")
            for ch_id in sorted(results.keys()):
                r = results[ch_id]
                f.write(f"\n# {'─'*58}\n# Channel {ch_id}\n")
                f.write(f"# Model   : {r.model_label}\n")
                if r.bad_channel:
                    f.write(f"# STATUS  : BAD — {r.bad_reason}\n")
                    continue
                f.write(f"# Chi2/NDF: {r.chi2_ndf:.6f}  (Chi2={r.chi2:.3f}, NDF={r.ndf})\n#\n")
                f.write("# Parameters:\n")
                for name, val, unc in zip(r.param_names, r.params, r.uncertainties):
                    f.write(f"#   {name} = {val:+.8e}  ±  {unc:.3e}\n")
                f.write("#\n")
                f.write(f"# {'ADC_position':>14s}  {'Energy_keV':>12s}  {'Residual_keV':>14s}\n")
                for adc, eng, res in zip(r.adc_points, r.energy_points, r.residuals):
                    f.write(f"  {adc:>14.4f}  {eng:>12.3f}  {res:>+14.5f}\n")

    @classmethod
    def write_coeffs(cls, results: dict, out_path: str, source_file: str = ""):
        good = [r for r in results.values() if not r.bad_channel]
        param_names = good[0].param_names if good else []
        with open(out_path, "w") as f:
            f.write(cls.HEADER)
            f.write(f"# Date     : {cls._ts()}\n")
            if source_file:
                f.write(f"# Source   : {source_file}\n")
            if good:
                f.write(f"# Model    : {good[0].model_label}\n")
            f.write("#\n")
            col_w = 18
            hdr = f"{'Channel':>8s}"
            for n in param_names:
                hdr += f"  {n:>{col_w}s}"
            hdr += f"  {'Chi2/NDF':>{col_w}s}  Status"
            f.write(f"# {hdr}\n")
            for ch_id in sorted(results.keys()):
                r   = results[ch_id]
                row = f"  {ch_id:>8d}"
                if r.bad_channel or not r.success:
                    for _ in param_names:
                        row += f"  {'nan':>{col_w}s}"
                    row += f"  {'nan':>{col_w}s}  BAD: {r.bad_reason[:40]}"
                else:
                    for val in r.params:
                        row += f"  {val:>{col_w}.8e}"
                    row += f"  {r.chi2_ndf:>{col_w}.6f}  OK"
                f.write(row + "\n")

    @classmethod
    def write_all(cls, results: dict, output_dir: str,
                   source_file: str = "", prefix: str = "calibration") -> list:
        os.makedirs(output_dir, exist_ok=True)
        if len(results) == 1:
            ch_id = next(iter(results))
            path  = os.path.join(output_dir, f"{prefix}_ch{ch_id:04d}.txt")
            cls.write_single(results[ch_id], path, source_file)
            return [path]
        else:
            log    = os.path.join(output_dir, f"{prefix}_log.txt")
            coeffs = os.path.join(output_dir, f"{prefix}_coeffs.txt")
            cls.write_log(results, log, source_file)
            cls.write_coeffs(results, coeffs, source_file)
            return [log, coeffs]
