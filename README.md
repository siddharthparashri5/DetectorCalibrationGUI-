# DetectorCalibGUI

**Energy calibration tool for multi-channel detector systems.**  
Built with Python · PyROOT · PyQt5 · Matplotlib · SciPy

Author: Siddharth Parashari

---

## Features

- Load ROOT files with **TTree** branches or pre-filled **TH1** histograms
- **Interactive spectrum** viewer — zoom, pan, click to mark peaks
- **Three peak input modes**: manual click, auto-detection, type known energy
- **Global peaks** applied to all channels, with per-channel overrides
- **Polynomial fits** (degree 1–5) or user-defined custom expressions
- **Batch fit** all channels in a background thread with progress bar
- **Overview grid** — thumbnail of all spectra, colour-coded by fit quality
- **Coefficient trend plots** — p0/p1/p2/... vs channel ID
- **Automatic bad channel flagging** (no peaks, fit failure, poor χ²/NDF)
- **Two output files** for multi-channel results:
  - `calibration_log.txt` — full per-channel report
  - `calibration_coeffs.txt` — compact table for direct analysis use

---

## Requirements

- Python 3.8+
- ROOT (with PyROOT) — must be sourced before running
- Python packages: see `requirements.txt`

```bash
pip install PyQt5 matplotlib numpy scipy
```

---

## Running

```bash
# Source ROOT first
source /opt/root/bin/thisroot.sh

# Run the application
python main.py
```

---

## Workflow

1. **Open ROOT File** — choose TTree branches or TH1 histograms
2. **Add global peaks** — type energies, click on spectrum, or use Auto-detect
3. Optionally **override peaks** for specific channels
4. Choose **calibration model** (e.g. poly2 for quadratic)
5. Click **Fit All Channels** — progress bar tracks the batch
6. Review results in **Overview Grid** (green = OK, red = bad)
7. Inspect individual channels + residuals in **Single Channel** tab
8. Check parameter stability in **Coefficient Trends** tab
9. **Export Results** — selects output directory, writes both output files

---

## Output Files

### `calibration_log.txt`
Full human-readable report — parameters, uncertainties, χ²/NDF,
calibration points and residuals per channel. Bad channels flagged with reason.

### `calibration_coeffs.txt`
Compact analysis-ready table:
```
# Channel   p0              p1              Chi2/NDF  Status
       0    1.234000e-01    5.678000e+00    1.2300    OK
       1    1.198000e-01    5.701000e+00    0.9800    OK
     512    nan             nan             nan       BAD: No calibration peaks
```

---

## Custom Calibration Functions

In the model selector, choose **custom** and enter an expression using
single-letter parameter names (not `x`):

```
a*x**2 + b*x + c
a * np.exp(b * x) + c
a*x + b
```

`x` is the ADC channel value. `np` (NumPy) is available in expressions.

---

## Project Structure

```
DetectorCalibGUI/
├── main.py                  # Entry point
├── requirements.txt
├── README.md
└── src/
    ├── __init__.py
    ├── root_loader.py       # PyROOT file loading (TTree + TH1)
    ├── peak_manager.py      # Peak storage, auto-detection
    ├── calib_fitter.py      # SciPy fitting, bad channel detection
    ├── output_writer.py     # Export: log + coefficients files
    └── main_window.py       # Full PyQt5 GUI
```
