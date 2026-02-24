# Detector Calibration GUI 

A modern, user-friendly graphical interface for Detector Calibration, Resolution calculations and plotting with an auto peak detection feature and functional user override to assign ADC and Energy values. 

![Version](https://img.shields.io/badge/version-2.0-blue)
![PyROOT](https://img.shields.io/badge/PyROOT-green)
![Python](https://img.shields.io/badge/Python-3.8+-orange)

**Energy calibration tool for multi-channel detector systems.**  
Built with Python · PyROOT · PyQt5 · Matplotlib · SciPy

Author: Siddharth Parashari

---

## Features

- Key Features
• Dual ROOT backend: PyROOT (native) or uproot (pure Python) — detected automatically.
• Flexible data loading: TTree filter mode, array-branch mode, custom Draw() expressions, or pre-filled
TH1 histograms.
• Three peak detection algorithms: ROOT TSpectrum (standard and HighRes), Sliding Window, and
SciPy fallback — all with optional Gaussian confirmation.
• Gaussian peak confirmation: Each detected candidate is verified by a Gaussian fit. Accepted peaks are
centroid-refined; rejected peaks are shown as red × markers.
• Peak propagation: Detect peaks on one reference channel and automatically search all other channels
within a configurable ADC window.
• Three calibration models: Linear, nonlinear scintillator, 3-point nonlinear variant, plus user-defined
custom expressions.
• Detect + Fit All: One-click workflow — detect, propagate, and fit all channels simultaneously in a
background thread.
• Calibrated spectrum viewer: Full-range energy spectra using session or file-loaded coefficients.
• Energy resolution analysis: Interactive Gaussian fitting with multiple background models, FWHM
extraction, and per-channel trend plots.
• Rich output: Human-readable calibration log, compact coefficients table, and resolution results file.

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
  - Select Branch, ChanelID, and custom user function to populate histograms
  - Select channelID range and number of entries
    
2. **ROOT TSpectrum Peak Search**
  - Select TSpectrum search parameters suitable for your spectrum
  - Background threshold
  - Sigma (bins)
  - Number of peaks to be detected
    
3. **Add global peaks** — click on the spectrum to assign energies using CLICK, or use Auto-detect
   
4. Optionally **override peaks** for specific channels

5. Choose **calibration model** (e.g. Linear or User defined)

6. Click **Fit All Channels** — progress bar tracks the batch

7. Review results in **Overview Grid** (green = OK, red = bad)

8. Inspect individual channels + residuals in **Single Channel** tab

9. Check parameter stability in **Coefficient Trends** tab

10. **Export Results** — selects output directory, writes both output files

11. **Calibrated Spectrum** — Use calibration parameters from memory (session memory) or load calibration file to see calibrated spectrum.

12. **Per Channel Resolution** — Use the built-in peak selector (drag on the spectrum) to select the fit area to calculate the resolution (% or keV)
    - Export Fitted spectrum and/or the resolution results in output files.

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
├── generate_test_data.C    # to generate a test data file (.ROOT)
└── src/
    ├── __init__.py
    ├── root_loader.py         # PyROOT file loading (TTree + TH1)
    ├── peak_manager.py        # Peak storage, auto-detection
    ├── calib_fitter.py        # SciPy fitting, bad channel detection
    ├── output_writer.py       # Export: log + coefficients files
    ├── calib_spectrum.py      # Calibrate histograms
    ├── resolution.py          # Calculate Resolution + Export
    ├── calib_spectrum_tab.py  # Calculation tab GUI
    ├── resolution_tab.py      # Resolution tab GUI
    └── main_window.py         # Full PyQt5 GUI
```


**Last Updated**: February 21, 2026  
**Version**: 1.0  
**Maintained by**: Siddharth Parashari


## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes with clear commit messages
4. Test thoroughly with sample data
5. Document new features in README
6. Submit a pull request


## Contact & Support

- **Issues**: Open an issue on GitHub
- **Email**: siddharthparashri5@gmail.com
