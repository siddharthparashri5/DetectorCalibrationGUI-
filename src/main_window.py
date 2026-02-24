"""
MainWindow — DetectorCalibGUI
PyQt5 application. Calibration formula: E = P0·(P1^Q)^P2 + P3·Q − P0
Peak detection via ROOT TSpectrum (PyROOT) or scipy (uproot fallback).
Also supports sliding-window detection and Gaussian peak confirmation.
"""

from __future__ import annotations
import os
import numpy as np

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QTabWidget, QLabel, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
    QLineEdit, QTableWidget, QTableWidgetItem, QFileDialog, QGroupBox,
    QCheckBox, QProgressBar, QMessageBox, QScrollArea,
    QGridLayout, QSizePolicy, QHeaderView, QDialog, QFormLayout,
    QDialogButtonBox, QApplication, QSlider, QFrame
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor, QFont

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar
)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.root_loader import ROOTFileLoader
from src.peak_manager import PeakManager, Peak, CRYSTAL_KNOWN_LINES, GaussianFitInfo
from src.calib_fitter import CalibrationFitter, MODELS
from src.output_writer import OutputWriter
from src.calib_spectrum_tab import CalibratedSpectrumTab
from src.resolution_tab import ResolutionTab


# ======================================================================== #
# Worker threads
# ======================================================================== #

class FitWorker(QThread):
    progress = pyqtSignal(int, int)
    finished = pyqtSignal(dict)
    error    = pyqtSignal(str)

    def __init__(self, fitter, channel_ids, peak_manager, model, custom_expr=""):
        super().__init__()
        self.fitter      = fitter
        self.channel_ids = channel_ids
        self.peak_mgr    = peak_manager
        self.model       = model
        self.custom_expr = custom_expr

    def run(self):
        try:
            results = self.fitter.fit_all(
                self.channel_ids, self.peak_mgr, self.model,
                custom_expr=self.custom_expr,
                progress_callback=lambda d, t: self.progress.emit(d, t))
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))


class DetectAndFitWorker(QThread):
    """
    Worker for 'Fit All Channels':
      1. Detect peaks in each channel using current UI settings
      2. Propagate assignments from already-assigned channels
      3. Fit calibration model
    """
    progress = pyqtSignal(int, int, str)   # done, total, status_msg
    finished = pyqtSignal(dict)
    error    = pyqtSignal(str)

    def __init__(self, fitter, loader, peak_manager, channel_ids,
                 model, custom_expr,
                 detect_kwargs: dict,
                 ref_assignments: list):   # list of (energy, label) from current channel
        super().__init__()
        self.fitter          = fitter
        self.loader          = loader
        self.peak_mgr        = peak_manager
        self.channel_ids     = channel_ids
        self.model           = model
        self.custom_expr     = custom_expr
        self.detect_kwargs   = detect_kwargs
        self.ref_assignments = ref_assignments  # [(adc, energy, label), ...]

    def run(self):
        try:
            total = len(self.channel_ids)
            for i, ch_id in enumerate(self.channel_ids):
                self.progress.emit(i, total, f"Detecting peaks ch {ch_id}…")

                sp = self.loader.get_spectrum(ch_id)
                if sp is None:
                    continue

                positions, bg_array, _ = PeakManager.detect_peaks(
                    sp.bin_centers, sp.counts, **self.detect_kwargs)

                self.peak_mgr.set_detected(ch_id, positions)

                # Auto-assign energies from reference assignments
                # (match each ref ADC to closest detected peak within window)
                window = self.detect_kwargs.get("sw_window_adc", 50.0)
                for ref_adc, energy, label in self.ref_assignments:
                    if not positions:
                        break
                    best_adc = min(positions, key=lambda p: abs(p - ref_adc))
                    if abs(best_adc - ref_adc) <= window:
                        # avoid duplicate
                        existing = self.peak_mgr.get_peaks(ch_id)
                        if not any(abs(p.known_energy - energy) < 0.5 for p in existing):
                            self.peak_mgr.add_channel_peak(ch_id, best_adc, energy, label)

            # Now fit all channels
            for i, ch_id in enumerate(self.channel_ids):
                self.progress.emit(i, total, f"Fitting ch {ch_id}…")
                adc, eng = self.peak_mgr.get_calibration_points(ch_id)
                if len(adc) == 0:
                    continue
                self.fitter.fit_channel(ch_id, adc, eng, self.model, self.custom_expr)

            self.progress.emit(total, total, "Done")
            self.finished.emit(self.fitter.results)
        except Exception as e:
            self.error.emit(str(e))


# ======================================================================== #
# Collapsible box widget
# ======================================================================== #

class CollapsibleBox(QWidget):
    def __init__(self, title: str, parent=None, collapsed: bool = False):
        super().__init__(parent)
        self._collapsed = collapsed
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._btn = QPushButton()
        self._btn.setCheckable(True)
        self._btn.setChecked(not collapsed)
        self._btn.clicked.connect(self._toggle)
        self._btn.setStyleSheet(
            "QPushButton {"
            "  text-align: left; padding: 4px 8px;"
            "  background: qlineargradient(x1:0,y1:0,x2:0,y2:1,"
            "              stop:0 #e3f2fd, stop:1 #bbdefb);"
            "  border: 1px solid #90caf9; border-radius: 4px;"
            "  font-weight: bold; font-size: 11px; color: #0d47a1;"
            "}"
            "QPushButton:checked {"
            "  background: qlineargradient(x1:0,y1:0,x2:0,y2:1,"
            "              stop:0 #1565c0, stop:1 #1976d2);"
            "  color: white; border-color: #0d47a1;"
            "}")
        self._title = title
        self._update_btn_text()
        outer.addWidget(self._btn)

        self._content = QWidget()
        self._content.setObjectName("collapsibleContent")
        self._content.setStyleSheet(
            "#collapsibleContent {"
            "  border: 1px solid #90caf9; border-top: none;"
            "  border-radius: 0 0 4px 4px; background: #fafafa; padding: 2px;"
            "}")
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(6, 4, 6, 6)
        self._content_layout.setSpacing(4)
        self._content.setVisible(not collapsed)
        outer.addWidget(self._content)

    def _toggle(self, checked: bool):
        self._collapsed = not checked
        self._content.setVisible(checked)
        self._update_btn_text()

    def _update_btn_text(self):
        arrow = "▼" if not self._collapsed else "▶"
        self._btn.setText(f"  {arrow}  {self._title}")

    def layout(self):
        return self._content_layout

    def addWidget(self, widget):
        self._content_layout.addWidget(widget)

    def addLayout(self, layout):
        self._content_layout.addLayout(layout)


# ======================================================================== #
# Spectrum Canvas
# ======================================================================== #

class SpectrumCanvas(FigureCanvas):
    peak_clicked = pyqtSignal(float)

    def __init__(self, parent=None):
        self.fig, self.ax = plt.subplots(figsize=(7, 3.5))
        self.fig.tight_layout(pad=2)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._peak_markers    = []
        self._click_mode      = False
        self._threshold_line  = None
        self._current_max_cts = None
        self.mpl_connect("button_press_event", self._on_click)

    def set_click_mode(self, enabled: bool):
        self._click_mode = enabled
        self.setCursor(Qt.CrossCursor if enabled else Qt.ArrowCursor)

    def _on_click(self, event):
        if self._click_mode and event.inaxes == self.ax and event.button == 1:
            self.peak_clicked.emit(event.xdata)

    def plot_spectrum(self, bin_centers, counts, title=""):
        self.ax.clear()
        self._peak_markers.clear()
        self._threshold_line  = None
        self._current_max_cts = float(counts.max()) if len(counts) else None
        self.ax.step(bin_centers, counts, where="mid",
                      color="#1565c0", linewidth=0.8)
        self.ax.set_xlabel("ADC Value", fontsize=9)
        self.ax.set_ylabel("Counts",    fontsize=9)
        self.ax.set_title(title, fontsize=10)
        self.ax.set_yscale("log")
        self.ax.set_ylim(bottom=0.5)
        self.ax.grid(True, alpha=0.25, linestyle="--")
        self.fig.tight_layout(pad=1.5)
        self.draw()

    def update_threshold_line(self, threshold: float):
        if self._current_max_cts is None:
            return
        level = max(0.5, threshold * self._current_max_cts)
        if self._threshold_line is not None:
            try:
                self._threshold_line.remove()
            except Exception:
                pass
            self._threshold_line = None
        self._threshold_line = self.ax.axhline(
            level, color="#e53935", linestyle="--", linewidth=1.2,
            alpha=0.85, zorder=6,
            label=f"Threshold ({threshold:.0%} × max = {level:.0f} cts)")
        handles, labels = self.ax.get_legend_handles_labels()
        seen = {}
        for h, l in zip(handles, labels):
            seen[l] = h
        self.ax.legend(seen.values(), seen.keys(), fontsize=7, loc="upper right")
        self.fig.tight_layout(pad=1.5)
        self.draw()

    def draw_detected_peaks(self, positions: list,
                             color: str = "#FF6F00",
                             style: str = "|",
                             label_prefix: str = ""):
        """
        Draw detected peak markers.
        style="|"  → vertical lines with ADC labels
        style="x"  → cross markers at mid-height (for rejected Gaussian fits)
        """
        ymin, ymax = self.ax.get_ylim()
        mid_y = np.sqrt(ymin * ymax) if ymin > 0 else ymax * 0.3
        for x in positions:
            if style == "|":
                line = self.ax.axvline(x, color=color, linestyle=":",
                                        linewidth=1.4, alpha=0.9)
                self.ax.text(x, ymax * 0.6, f" {x:.0f}",
                              color=color, fontsize=6.5,
                              rotation=90, va="top")
                self._peak_markers.append(line)
            elif style == "x":
                sc = self.ax.plot(x, mid_y, "x", color=color,
                                   markersize=9, markeredgewidth=2.0, alpha=0.9)
                self._peak_markers.extend(sc)
        self.draw()

    def draw_gauss_fits(self, bin_centers: np.ndarray, counts: np.ndarray,
                         fit_infos: list):
        """Overlay Gaussian fit curves for confirmed peaks."""
        from src.peak_manager import _gauss_linear
        ymin, ymax = self.ax.get_ylim()
        for info in fit_infos:
            if not info.success:
                continue
            x_lo = info.centroid - info.sigma * 3
            x_hi = info.centroid + info.sigma * 3
            mask = (bin_centers >= x_lo) & (bin_centers <= x_hi)
            if mask.sum() < 3:
                continue
            x_fine = np.linspace(x_lo, x_hi, 200)
            # Reconstruct background from surrounding bins
            bg_mask  = (bin_centers >= x_lo) & (bin_centers <= x_hi)
            bg_level = float(np.percentile(counts[bg_mask], 10)) if bg_mask.sum() > 3 else 0
            y_fine   = _gauss_linear(x_fine,
                                      info.amplitude, info.centroid, info.sigma,
                                      bg_level, 0.0)
            self.ax.plot(x_fine, y_fine, "-", color="#2e7d32",
                          linewidth=1.2, alpha=0.8, zorder=7)
            self.ax.annotate(
                f"σ={info.sigma:.1f}\nχ²/NDF={info.chi2_ndf:.1f}",
                xy=(info.centroid, info.amplitude * 0.5),
                xytext=(info.centroid, info.amplitude * 0.5),
                fontsize=5.5, color="#2e7d32", va="bottom")
        self.draw()

    def draw_assigned_peaks(self, peaks: list):
        ymax = self.ax.get_ylim()[1]
        for p in peaks:
            line = self.ax.axvline(p.adc_position, color="#2E7D32",
                                    linestyle="--", linewidth=1.3, alpha=0.9)
            self.ax.text(p.adc_position, ymax * 0.85,
                          f" {p.known_energy:.0f} keV",
                          color="#2E7D32", fontsize=7,
                          rotation=90, va="top")
            self._peak_markers.append(line)
        self.draw()

    def draw_background(self, bin_centers, bg_counts):
        if bg_counts is None or len(bg_counts) == 0:
            return
        self.ax.fill_between(bin_centers, bg_counts,
                              alpha=0.18, color="#f57c00",
                              label="SNIP background estimate")
        self.ax.step(bin_centers, bg_counts, where="mid",
                      color="#f57c00", linewidth=0.7, alpha=0.6)
        handles, labels = self.ax.get_legend_handles_labels()
        seen = {}
        for h, l in zip(handles, labels):
            seen[l] = h
        self.ax.legend(seen.values(), seen.keys(), fontsize=7, loc="upper right")
        self.fig.tight_layout(pad=1.5)
        self.draw()

    def draw_known_lines_adc(self, known_lines: list, fit_result):
        if not known_lines or fit_result is None or not fit_result.success:
            return
        from scipy.optimize import brentq
        ymax = self.ax.get_ylim()[1]
        for line_info in known_lines:
            energy = line_info["energy"]
            color  = line_info.get("color", "#7b1fa2")
            label  = line_info["label"]
            try:
                adc_lo, adc_hi = 0.0, 1e6
                f_lo = fit_result.energy_at(adc_lo) - energy
                f_hi = fit_result.energy_at(adc_hi) - energy
                if f_lo * f_hi > 0:
                    continue
                adc_pos = brentq(lambda q: fit_result.energy_at(q) - energy,
                                  adc_lo, adc_hi, xtol=0.1, maxiter=100)
            except Exception:
                continue
            self.ax.axvline(adc_pos, color=color, linestyle="-.",
                             linewidth=1.1, alpha=0.75)
            self.ax.text(adc_pos, ymax * 0.45, f" {label}",
                          color=color, fontsize=6, rotation=90, va="top")
        self.fig.tight_layout(pad=1.5)
        self.draw()


class CalibCurveCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig, self.ax = plt.subplots(figsize=(7, 3.0))
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def plot_result(self, result):
        self.ax.clear()
        if result is None or not result.success:
            self.ax.text(0.5, 0.5, "No fit result yet",
                          ha="center", va="center",
                          transform=self.ax.transAxes,
                          color="#9e9e9e", fontsize=10)
            self.fig.tight_layout(pad=1.2)
            self.draw()
            return

        adc = result.adc_points
        eng = result.energy_points
        adc_limit = result.adc_max
        x_fit = np.linspace(max(0, adc.min() * 0.8), adc_limit, 500)
        y_fit = result.energy_at(x_fit)

        if np.all(np.isnan(y_fit)):
            self.ax.text(0.5, 0.5, f"Cannot plot model '{result.model}'",
                          ha="center", va="center",
                          transform=self.ax.transAxes,
                          color="#c62828", fontsize=10)
            self.fig.tight_layout(pad=1.2)
            self.draw()
            return

        self.ax.plot(x_fit, y_fit, "-", color="#1565c0",
                      linewidth=1.8, label="Calibration fit")
        self.ax.scatter(adc, eng, color="#c62828", zorder=5,
                         s=60, label="Assigned peaks")
        for a, e in zip(adc, eng):
            self.ax.annotate(f"{e:.1f} keV", xy=(a, e),
                              xytext=(4, 4), textcoords="offset points",
                              fontsize=7, color="#c62828")
        e_at_limit = result.energy_at(np.array([adc_limit]))[0]
        if not np.isnan(e_at_limit):
            self.ax.axvline(adc_limit, color="#e65100", linestyle="--",
                             linewidth=1.0, alpha=0.7,
                             label=f"Fit limit ADC {adc_limit:.0f}")
        chi2_str = (f"χ²/NDF = {result.chi2_ndf:.4f}"
                    if result.ndf > 0 else "exact fit (NDF=0)")
        self.ax.set_xlabel("ADC Value (Q)", fontsize=9)
        self.ax.set_ylabel("Energy (keV)",  fontsize=9)
        self.ax.set_title(
            f"Ch {result.channel_id}  |  {result.model_label}  |  {chi2_str}",
            fontsize=9)
        self.ax.legend(fontsize=8)
        self.ax.grid(True, alpha=0.25, linestyle="--")
        self.fig.tight_layout(pad=1.2)
        self.draw()


class TrendCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig, self.ax = plt.subplots(figsize=(8, 3))
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def plot_trends(self, results: dict, param_index: int, param_name: str):
        self.ax.clear()
        good = {ch: r for ch, r in results.items()
                if not r.bad_channel and r.success
                and param_index < len(r.params)}
        if not good:
            self.ax.text(0.5, 0.5, "No good channels",
                          ha="center", va="center",
                          transform=self.ax.transAxes)
            self.draw()
            return
        channels = sorted(good.keys())
        vals = [good[ch].params[param_index]        for ch in channels]
        errs = [good[ch].uncertainties[param_index] for ch in channels]
        bad_chs = [ch for ch, r in results.items() if r.bad_channel]
        self.ax.errorbar(channels, vals, yerr=errs, fmt="o",
                          markersize=3, linewidth=0.8,
                          color="#1565c0", ecolor="#90CAF9",
                          capsize=2, label=param_name)
        for bch in bad_chs:
            self.ax.axvline(bch, color="#f44336", alpha=0.25, linewidth=0.8)
        self.ax.set_xlabel("Channel ID", fontsize=9)
        self.ax.set_ylabel(param_name,   fontsize=9)
        self.ax.set_title(f"Parameter '{param_name}' vs Channel", fontsize=10)
        self.ax.grid(True, alpha=0.25, linestyle="--")
        bad_p = mpatches.Patch(color="#f44336", alpha=0.4,
                                label=f"Bad ({len(bad_chs)})")
        self.ax.legend(handles=[
            *self.ax.get_legend_handles_labels()[0], bad_p], fontsize=8)
        self.fig.tight_layout(pad=1.5)
        self.draw()


# ======================================================================== #
# Overview grid
# ======================================================================== #

class OverviewGrid(QScrollArea):
    channel_selected = pyqtSignal(int)
    THUMB_W, THUMB_H = 160, 100
    COLS = 8

    def __init__(self, parent=None):
        super().__init__(parent)
        self._widget = QWidget()
        self._layout = QGridLayout(self._widget)
        self._layout.setSpacing(4)
        self.setWidget(self._widget)
        self.setWidgetResizable(True)

    def populate(self, spectra: dict, results: dict = None):
        for i in reversed(range(self._layout.count())):
            w = self._layout.itemAt(i).widget()
            if w:
                w.setParent(None)
        if not spectra:
            return
        for i, ch_id in enumerate(sorted(spectra.keys())):
            sp  = spectra[ch_id]
            fig, ax = plt.subplots(
                figsize=(self.THUMB_W / 72, self.THUMB_H / 72), dpi=72)
            ax.step(sp.bin_centers, sp.counts, where="mid",
                     linewidth=0.5, color="#1565c0")
            ax.set_yscale("log")
            ax.set_title(f"Ch {ch_id}", fontsize=5, pad=1)
            ax.tick_params(labelsize=4)
            ax.set_ylim(bottom=0.5)
            fig.tight_layout(pad=0.3)
            canvas = FigureCanvas(fig)
            canvas.setFixedSize(self.THUMB_W, self.THUMB_H)
            border = "#aaaaaa"
            if results:
                r = results.get(ch_id)
                if r:
                    border = "#f44336" if r.bad_channel else "#2e7d32"
            canvas.setStyleSheet(
                f"border: 2px solid {border}; border-radius: 3px;")
            ch_capture = ch_id
            canvas.mousePressEvent = lambda e, c=ch_capture: \
                self.channel_selected.emit(c)
            row, col = divmod(i, self.COLS)
            self._layout.addWidget(canvas, row, col)
            plt.close(fig)
        self._widget.adjustSize()


# ======================================================================== #
# Dialogs
# ======================================================================== #

class LoadDialog(QDialog):
    def __init__(self, file_info: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configure Data Source")
        self.setMinimumWidth(560)

        self.result_mode        = ""
        self.result_tree        = ""
        self.result_ch_branch   = ""
        self.result_adc_branch  = ""
        self.result_nbins       = 1024
        self.result_draw_mode   = "filter"
        self.result_custom_expr = ""
        self.result_ch_first    = 0
        self.result_ch_last     = -1
        self.result_ch_step     = 1
        self.result_max_entries = 0
        self.result_ch_ids      = None

        layout = QVBoxLayout(self)

        backend = file_info.get("backend", "unknown").upper()
        blabel  = QLabel(f"Active backend: <b>{backend}</b>")
        blabel.setStyleSheet("color: #1565c0; padding: 4px;")
        layout.addWidget(blabel)

        has_trees = bool(file_info["trees"])
        has_hists = bool(file_info["histograms"])

        mode_box    = QGroupBox("Data Mode")
        mode_layout = QVBoxLayout(mode_box)
        self.rb_tree = QCheckBox(f"TTree  ({len(file_info['trees'])} found)")
        self.rb_hist = QCheckBox(f"TH1 Histograms  ({len(file_info['histograms'])} found)")
        self.rb_tree.setEnabled(has_trees)
        self.rb_hist.setEnabled(has_hists)
        if has_trees:   self.rb_tree.setChecked(True)
        elif has_hists: self.rb_hist.setChecked(True)
        mode_layout.addWidget(self.rb_tree)
        mode_layout.addWidget(self.rb_hist)
        layout.addWidget(mode_box)

        self.tree_group = QGroupBox("TTree Options")
        tg = QFormLayout(self.tree_group)

        self.cb_tree_name = QComboBox()
        for t in file_info["trees"]:
            self.cb_tree_name.addItem(
                f"{t['name']}  ({t['entries']} entries)", t["name"])

        self.cb_draw_mode = QComboBox()
        self.cb_draw_mode.addItem('Filter     Draw("adc", "channelID==N")', "filter")
        self.cb_draw_mode.addItem('Array      Draw("adc[channelID]")', "array")
        self.cb_draw_mode.addItem('Custom     user expression (%d = channel number)', "custom")

        self.cb_channel_branch = QComboBox()
        self.cb_adc_branch     = QComboBox()

        self.le_draw_custom = QLineEdit()
        self.le_draw_custom.setPlaceholderText(
            'e.g.  energy[%d]   or   sqrt(adc),channelID==%d')
        self.le_draw_custom.setEnabled(False)

        self.lbl_draw_preview = QLabel("")
        self.lbl_draw_preview.setStyleSheet(
            "color:#555; font-size:11px; font-style:italic;")
        self.lbl_draw_preview.setWordWrap(True)

        self.sb_nbins = QSpinBox()
        self.sb_nbins.setRange(64, 65536)
        self.sb_nbins.setValue(1024)

        tg.addRow("TTree:",             self.cb_tree_name)
        tg.addRow("Draw mode:",         self.cb_draw_mode)
        tg.addRow("Channel branch:",    self.cb_channel_branch)
        tg.addRow("Energy/ADC branch:", self.cb_adc_branch)
        tg.addRow("Custom expression:", self.le_draw_custom)
        tg.addRow("Draw preview:",      self.lbl_draw_preview)
        tg.addRow("Histogram bins:",    self.sb_nbins)
        layout.addWidget(self.tree_group)

        self.ch_range_box = QGroupBox("Channel Range (Optional)")
        cr = QFormLayout(self.ch_range_box)
        range_h = QHBoxLayout()
        self.sb_ch_first = QSpinBox(); self.sb_ch_first.setRange(0, 999999)
        self.sb_ch_last  = QSpinBox(); self.sb_ch_last.setRange(-1, 999999)
        self.sb_ch_last.setSpecialValueText("Auto")
        self.sb_ch_step  = QSpinBox(); self.sb_ch_step.setRange(1, 1000); self.sb_ch_step.setValue(1)
        range_h.addWidget(QLabel("First:")); range_h.addWidget(self.sb_ch_first)
        range_h.addWidget(QLabel("  Last:")); range_h.addWidget(self.sb_ch_last)
        range_h.addWidget(QLabel("  Step:")); range_h.addWidget(self.sb_ch_step)
        cr.addRow("Range:", range_h)
        self.le_ch_list = QLineEdit()
        self.le_ch_list.setPlaceholderText("Optional: comma-separated, e.g.  0,1,2,5,10")
        cr.addRow("Custom list:", self.le_ch_list)
        self.sb_max_entries = QSpinBox()
        self.sb_max_entries.setRange(0, 100_000_000); self.sb_max_entries.setValue(0)
        self.sb_max_entries.setSpecialValueText("All entries")
        self.sb_max_entries.setSingleStep(10000)
        cr.addRow("Max entries:", self.sb_max_entries)
        layout.addWidget(self.ch_range_box)

        self.hist_group = QGroupBox("Histogram Options")
        QVBoxLayout(self.hist_group).addWidget(
            QLabel(f"{len(file_info['histograms'])} histogram(s) found — all will be loaded."))
        layout.addWidget(self.hist_group)

        self.cb_tree_name.currentIndexChanged.connect(
            lambda idx: self._fill_branches(file_info, idx))
        self.cb_draw_mode.currentIndexChanged.connect(self._on_draw_mode)
        self.cb_channel_branch.currentTextChanged.connect(self._update_preview)
        self.cb_adc_branch.currentTextChanged.connect(self._update_preview)
        self.le_draw_custom.textChanged.connect(self._update_preview)
        self.rb_tree.toggled.connect(self._toggle)
        self.rb_hist.toggled.connect(self._toggle)

        self._fill_branches(file_info, 0)
        self._toggle()
        self._on_draw_mode(0)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)
        self._file_info = file_info

    def _fill_branches(self, info, idx):
        if not info["trees"] or idx >= len(info["trees"]):
            return
        self.cb_channel_branch.clear()
        self.cb_adc_branch.clear()
        for b in info["trees"][idx]["branches"]:
            self.cb_channel_branch.addItem(b)
            self.cb_adc_branch.addItem(b)
        if len(info["trees"][idx]["branches"]) > 1:
            self.cb_adc_branch.setCurrentIndex(1)
        self._update_preview()

    def _on_draw_mode(self, idx: int):
        mode = self.cb_draw_mode.currentData()
        self.cb_channel_branch.setEnabled(mode == "filter")
        self.le_draw_custom.setEnabled(mode == "custom")
        self.ch_range_box.setVisible(True)
        self._update_preview()

    def _update_preview(self):
        mode = self.cb_draw_mode.currentData()
        ch   = self.cb_channel_branch.currentText()
        adc  = self.cb_adc_branch.currentText()
        if mode == "filter":
            self.lbl_draw_preview.setText(f'Preview:  tree.Draw("{adc}", "{ch}==N")')
        elif mode == "array":
            self.lbl_draw_preview.setText(f'Preview:  tree.Draw("{adc}[N]")')
        elif mode == "custom":
            expr = self.le_draw_custom.text() or "%d"
            self.lbl_draw_preview.setText(f'Preview:  tree.Draw("{expr}") with %d→channel')

    def _toggle(self):
        use_tree = self.rb_tree.isChecked()
        self.tree_group.setVisible(use_tree)
        self.hist_group.setVisible(not use_tree)
        if use_tree:
            self._on_draw_mode(self.cb_draw_mode.currentIndex())
        else:
            self.ch_range_box.setVisible(False)

    def _accept(self):
        if self.rb_tree.isChecked():
            self.result_mode        = "ttree"
            self.result_tree        = self.cb_tree_name.currentData()
            self.result_ch_branch   = self.cb_channel_branch.currentText()
            self.result_adc_branch  = self.cb_adc_branch.currentText()
            self.result_nbins       = self.sb_nbins.value()
            self.result_draw_mode   = self.cb_draw_mode.currentData()
            self.result_custom_expr = self.le_draw_custom.text().strip()
            self.result_ch_first    = self.sb_ch_first.value()
            self.result_ch_last     = self.sb_ch_last.value()
            self.result_ch_step     = self.sb_ch_step.value()
            self.result_max_entries = self.sb_max_entries.value()
            raw = self.le_ch_list.text().strip()
            if raw:
                try:
                    self.result_ch_ids = [int(x.strip()) for x in raw.split(",") if x.strip()]
                except ValueError:
                    QMessageBox.warning(self, "Invalid Channel List",
                        "Channel list must be comma-separated integers.")
                    return
            else:
                self.result_ch_ids = None
        else:
            self.result_mode   = "th1"
            self.result_ch_ids = None
        self.accept()


class AssignEnergyDialog(QDialog):
    COMMON_SOURCES = [
        ("511.0",  "Na-22 / annihilation"), ("1274.5", "Na-22"),
        ("661.7",  "Cs-137"),
        ("88",     "Lu-176"), ("202", "Lu-176"), ("307", "Lu-176"),
        ("1173.2", "Co-60"),  ("122.1", "Co-57"), ("344.3", "Eu-152"),
        ("1460.8", "K-40"),   ("1764.5", "Ra-226"), ("2614.5", "Tl-208"),
        ("59.5",   "Am-241"), ("88.0", "Cd-109"),
    ]

    def __init__(self, adc_position: float = 0.0, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Assign Known Energy to Peak")
        self.setMinimumWidth(480)
        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.sb_adc = QDoubleSpinBox()
        self.sb_adc.setRange(0, 1e9); self.sb_adc.setDecimals(3)
        self.sb_adc.setValue(adc_position)

        self.sb_energy = QDoubleSpinBox()
        self.sb_energy.setRange(0, 1e9); self.sb_energy.setDecimals(3)

        self.le_label = QLineEdit()
        self.le_label.setPlaceholderText("optional label")

        form.addRow("ADC position (Q):", self.sb_adc)
        form.addRow("Known energy (keV):", self.sb_energy)
        form.addRow("Label:", self.le_label)
        layout.addLayout(form)

        layout.addWidget(QLabel("<b>Quick-select common gamma sources:</b>"))
        tbl = QTableWidget(len(self.COMMON_SOURCES), 2)
        tbl.setHorizontalHeaderLabels(["Energy (keV)", "Source"])
        tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        tbl.setMaximumHeight(180)
        tbl.setEditTriggers(QTableWidget.NoEditTriggers)
        tbl.setSelectionBehavior(QTableWidget.SelectRows)
        for row, (e, src) in enumerate(self.COMMON_SOURCES):
            tbl.setItem(row, 0, QTableWidgetItem(e))
            tbl.setItem(row, 1, QTableWidgetItem(src))
        tbl.cellClicked.connect(self._quick_select)
        tbl.cellDoubleClicked.connect(self._quick_select)
        layout.addWidget(tbl)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _quick_select(self, row: int, col: int):
        e   = float(self.COMMON_SOURCES[row][0])
        src = self.COMMON_SOURCES[row][1]
        self.sb_energy.setValue(e)
        if not self.le_label.text():
            self.le_label.setText(src)


class _LoadingProgressDialog(QDialog):
    def __init__(self, title: str, n_channels: int, parent=None):
        super().__init__(parent, Qt.WindowTitleHint | Qt.CustomizeWindowHint)
        self.setWindowTitle(title)
        self.setFixedWidth(380)
        layout = QVBoxLayout(self)
        self.lbl = QLabel(f"Loading 0 / {n_channels} channels…")
        layout.addWidget(self.lbl)
        self.bar = QProgressBar()
        self.bar.setRange(0, n_channels); self.bar.setValue(0)
        layout.addWidget(self.bar)

    def update(self, done: int, total: int):
        self.lbl.setText(f"Loading {done} / {total} channels…")
        self.bar.setValue(done)
        QApplication.processEvents()


# ======================================================================== #
# Main Window
# ======================================================================== #

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Detector Energy Calibration & Resolution")
        self.setMinimumSize(1150, 980)

        self.loader           = ROOTFileLoader()
        self.peak_mgr         = PeakManager()
        self.fitter           = CalibrationFitter()
        self.fit_results: dict = {}
        self.current_channel: int = -1
        self._detected_positions: list = []
        self._click_mode = False
        self._last_assigned: dict | None = None
        self.channel_crystal: dict = {}
        self._last_gauss_infos: list = []   # store last Gaussian fit results

        self._build_ui()
        self._apply_style()

    # ------------------------------------------------------------------ #
    # UI construction
    # ------------------------------------------------------------------ #

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setSpacing(0)
        root_layout.setContentsMargins(0, 0, 0, 0)

        root_layout.addWidget(self._build_toolbar())

        splitter = QSplitter(Qt.Horizontal)
        root_layout.addWidget(splitter, stretch=1)

        left = self._build_left_panel()
        left.setMinimumWidth(300)
        left.setMaximumWidth(430)
        splitter.addWidget(left)

        self.tabs = QTabWidget()
        splitter.addWidget(self.tabs)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        self._build_detail_tab()
        self._build_grid_tab()
        self._build_trends_tab()
        self._build_calib_spectrum_tab()
        self._build_resolution_tab()

        self.statusBar().showMessage("Ready — open a ROOT file to begin.")
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximumWidth(220)
        self.statusBar().addPermanentWidget(self.progress_bar)

    def _build_toolbar(self) -> QWidget:
        bar = QWidget()
        bar.setObjectName("toolbar")
        bar.setFixedHeight(46)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(8, 4, 8, 4)

        self.btn_open = QPushButton("📂  Open ROOT File")
        self.btn_open.clicked.connect(self._open_file)
        layout.addWidget(self.btn_open)

        self.lbl_file = QLabel("No file loaded")
        self.lbl_file.setObjectName("fileLabel")
        layout.addWidget(self.lbl_file, stretch=1)

        layout.addWidget(QLabel("Model:"))
        self.cb_model = QComboBox()
        self.cb_model.addItem("Linear  E = P0 + P1·Q",                           "linear")
        self.cb_model.addItem("Nonlinear  E = P0·(P1^Q)^P2 + P3·Q − P0",        "nonlinear")
        self.cb_model.addItem("Nonlinear 3-pt  E = P0·P1^Q + P3·Q − P0 (P2=1)", "nonlinear_3pt")
        self.cb_model.addItem("Custom", "custom")
        self.cb_model.setMinimumWidth(340)
        self.cb_model.currentIndexChanged.connect(self._on_model_changed)
        layout.addWidget(self.cb_model)

        self.le_custom_expr = QLineEdit()
        self.le_custom_expr.setPlaceholderText(
            "Custom expression, e.g.:  a*x**2 + b*x + c   (x = ADC value)")
        self.le_custom_expr.setEnabled(False)
        self.le_custom_expr.setMinimumWidth(300)
        layout.addWidget(self.le_custom_expr)

        self.btn_export = QPushButton("Export")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self._export)
        layout.addWidget(self.btn_export)

        return bar

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        # ── Channel navigation ──────────────────────────────────────── #
        nav_box = QGroupBox("Channel Navigator")
        nav_l   = QVBoxLayout(nav_box)
        nav_h   = QHBoxLayout()
        self.btn_prev = QPushButton("◀"); self.btn_prev.setFixedWidth(100)
        self.btn_prev.clicked.connect(self._prev_channel)
        self.cb_channel = QComboBox()
        self.cb_channel.currentIndexChanged.connect(self._on_channel_changed)
        self.btn_next = QPushButton("▶"); self.btn_next.setFixedWidth(100)
        self.btn_next.clicked.connect(self._next_channel)
        nav_h.addWidget(self.btn_prev)
        nav_h.addWidget(self.cb_channel, stretch=1)
        nav_h.addWidget(self.btn_next)
        nav_l.addLayout(nav_h)
        self.lbl_ch_info = QLabel("—")
        self.lbl_ch_info.setWordWrap(True)
        self.lbl_ch_info.setStyleSheet("font-size: 11px; color: #424242;")
        nav_l.addWidget(self.lbl_ch_info)
        layout.addWidget(nav_box)

        # ══════════════════════════════════════════════════════════════ #
        # Peak detection (collapsible)
        # ══════════════════════════════════════════════════════════════ #
        det_box = CollapsibleBox("🔍  Peak Detection", collapsed=True)

        # ── Detection method selector ─────────────────────────────── #
        method_h = QHBoxLayout()
        method_h.addWidget(QLabel("Method:"))
        self.cb_detect_method = QComboBox()
        self.cb_detect_method.addItem("TSpectrum (ROOT/scipy)", "tspectrum")
        self.cb_detect_method.addItem("Sliding Window",          "sliding_window")
        self.cb_detect_method.currentIndexChanged.connect(self._on_detect_method_changed)
        method_h.addWidget(self.cb_detect_method, stretch=1)
        det_box.addLayout(method_h)

        # ── TSpectrum options ─────────────────────────────────────── #
        self._tspec_widget = QWidget()
        tspec_l = QVBoxLayout(self._tspec_widget)
        tspec_l.setContentsMargins(0, 0, 0, 0)
        tspec_l.setSpacing(3)

        thresh_h = QHBoxLayout()
        thresh_h.addWidget(QLabel("Threshold:"))
        self.sl_threshold = QSlider(Qt.Horizontal)
        self.sl_threshold.setRange(1, 50); self.sl_threshold.setValue(5)
        self.sl_threshold.setToolTip(
            "TSpectrum threshold: fraction of tallest peak height.\n"
            "Lower = find more (smaller) peaks.")
        self.lbl_threshold = QLabel("5%  of max")
        self.lbl_threshold.setMinimumWidth(80)
        self.sl_threshold.valueChanged.connect(self._on_threshold_changed)
        thresh_h.addWidget(self.sl_threshold, stretch=1)
        thresh_h.addWidget(self.lbl_threshold)
        tspec_l.addLayout(thresh_h)

        mode_h = QHBoxLayout()
        mode_h.addWidget(QLabel("Search mode:"))
        self.cb_tspec_mode = QComboBox()
        self.cb_tspec_mode.addItem("Standard  (Search)",               "standard")
        self.cb_tspec_mode.addItem("High Resolution  (SearchHighRes)", "highres")
        self.cb_tspec_mode.setCurrentIndex(1)
        mode_h.addWidget(self.cb_tspec_mode, stretch=1)
        tspec_l.addLayout(mode_h)

        iter_h = QHBoxLayout()
        iter_h.addWidget(QLabel("Iterations (HighRes):"))
        self.sb_iterations = QSpinBox()
        self.sb_iterations.setRange(1, 50); self.sb_iterations.setValue(10)
        iter_h.addWidget(self.sb_iterations)
        tspec_l.addLayout(iter_h)

        sigma_h = QHBoxLayout()
        sigma_h.addWidget(QLabel("Sigma (bins):"))
        self.sb_sigma = QDoubleSpinBox()
        self.sb_sigma.setRange(0.5, 20.0); self.sb_sigma.setSingleStep(0.5)
        self.sb_sigma.setValue(2.0)
        sigma_h.addWidget(self.sb_sigma)
        tspec_l.addLayout(sigma_h)

        bg_h = QHBoxLayout()
        self.chk_bg_subtract = QCheckBox("Subtract background (SNIP)")
        self.chk_bg_subtract.setChecked(True)
        bg_h.addWidget(self.chk_bg_subtract)
        self.sb_bg_iterations = QSpinBox()
        self.sb_bg_iterations.setRange(1, 100); self.sb_bg_iterations.setValue(20)
        bg_h.addWidget(QLabel("iter:")); bg_h.addWidget(self.sb_bg_iterations)
        tspec_l.addLayout(bg_h)

        det_box.addWidget(self._tspec_widget)

        # ── Sliding window options ────────────────────────────────── #
        self._sw_widget = QWidget()
        sw_l = QVBoxLayout(self._sw_widget)
        sw_l.setContentsMargins(0, 0, 0, 0)
        sw_l.setSpacing(3)

        sw_note = QLabel(
            "Scans spectrum with a sliding window.\n"
            "Bin is a peak if it is the local maximum\n"
            "and exceeds the threshold + prominence.")
        sw_note.setStyleSheet("font-size: 10px; color: #555;")
        sw_note.setWordWrap(True)
        sw_l.addWidget(sw_note)

        sw_win_h = QHBoxLayout()
        sw_win_h.addWidget(QLabel("Window width (ADC):"))
        self.sb_sw_window = QDoubleSpinBox()
        self.sb_sw_window.setRange(1, 100000); self.sb_sw_window.setValue(50)
        self.sb_sw_window.setSingleStep(10)
        self.sb_sw_window.setToolTip(
            "Full width of the sliding window in ADC units.\n"
            "A bin is a peak only if it is the maximum within this window.")
        sw_win_h.addWidget(self.sb_sw_window)
        sw_l.addLayout(sw_win_h)

        sw_thresh_h = QHBoxLayout()
        sw_thresh_h.addWidget(QLabel("Height threshold (%):"))
        self.sb_sw_threshold = QDoubleSpinBox()
        self.sb_sw_threshold.setRange(0.1, 99); self.sb_sw_threshold.setValue(5.0)
        self.sb_sw_threshold.setSingleStep(1.0)
        self.sb_sw_threshold.setSuffix(" %")
        sw_thresh_h.addWidget(self.sb_sw_threshold)
        sw_l.addLayout(sw_thresh_h)

        sw_prom_h = QHBoxLayout()
        sw_prom_h.addWidget(QLabel("Min prominence (%):"))
        self.sb_sw_prominence = QDoubleSpinBox()
        self.sb_sw_prominence.setRange(0.1, 99); self.sb_sw_prominence.setValue(2.0)
        self.sb_sw_prominence.setSingleStep(0.5)
        self.sb_sw_prominence.setSuffix(" %")
        self.sb_sw_prominence.setToolTip(
            "Minimum local prominence as % of global max.\n"
            "Peak must stand out above the local baseline by this amount.")
        sw_prom_h.addWidget(self.sb_sw_prominence)
        sw_l.addLayout(sw_prom_h)

        sw_smooth_h = QHBoxLayout()
        sw_smooth_h.addWidget(QLabel("Pre-smooth σ (bins):"))
        self.sb_sw_smooth = QDoubleSpinBox()
        self.sb_sw_smooth.setRange(0, 20); self.sb_sw_smooth.setValue(1.5)
        self.sb_sw_smooth.setSingleStep(0.5)
        self.sb_sw_smooth.setToolTip(
            "Gaussian smoothing before sliding-window detection.\n"
            "0 = no smoothing.")
        sw_smooth_h.addWidget(self.sb_sw_smooth)
        sw_l.addLayout(sw_smooth_h)

        self._sw_widget.setVisible(False)
        det_box.addWidget(self._sw_widget)

        # ── Shared options ─────────────────────────────────────────── #
        shared_l = QVBoxLayout()

        maxpk_h = QHBoxLayout()
        maxpk_h.addWidget(QLabel("Max peaks:"))
        self.sb_max_peaks = QSpinBox()
        self.sb_max_peaks.setRange(1, 30); self.sb_max_peaks.setValue(10)
        maxpk_h.addWidget(self.sb_max_peaks)
        shared_l.addLayout(maxpk_h)

        ped_h = QHBoxLayout()
        ped_h.addWidget(QLabel("Pedestal cut (ADC):"))
        self.sb_pedestal = QSpinBox()
        self.sb_pedestal.setRange(0, 100000); self.sb_pedestal.setValue(0)
        self.sb_pedestal.setSingleStep(10)
        ped_h.addWidget(self.sb_pedestal)
        shared_l.addLayout(ped_h)

        # ── Gaussian confirmation ─────────────────────────────────── #
        gauss_h = QHBoxLayout()
        self.chk_gauss_confirm = QCheckBox("Gaussian confirm peaks")
        self.chk_gauss_confirm.setChecked(False)
        self.chk_gauss_confirm.setToolTip(
            "After detection, fit a Gaussian to each candidate peak.\n"
            "Peaks that cannot be fit are rejected (shown as red ×).\n"
            "Accepted peaks are refined to the Gaussian centroid.")
        gauss_h.addWidget(self.chk_gauss_confirm)
        self.sb_gauss_window = QDoubleSpinBox()
        self.sb_gauss_window.setRange(1, 10000); self.sb_gauss_window.setValue(20)
        self.sb_gauss_window.setSingleStep(5)
        self.sb_gauss_window.setToolTip("Half-width of Gaussian fit window (ADC units).")
        gauss_h.addWidget(QLabel("win ±"))
        gauss_h.addWidget(self.sb_gauss_window)
        shared_l.addLayout(gauss_h)
        det_box.addLayout(shared_l)

        self.btn_detect = QPushButton("🔍  Detect Peaks (Current channel)")
        self.btn_detect.setEnabled(False)
        self.btn_detect.clicked.connect(self._detect_peaks)
        det_box.addWidget(self.btn_detect)

        self.lbl_detected = QLabel("No peaks detected yet.")
        self.lbl_detected.setWordWrap(True)
        self.lbl_detected.setStyleSheet("font-size: 10px; color: #555;")
        det_box.addWidget(self.lbl_detected)
        layout.addWidget(det_box)

        # ── Detected peaks table ──────────────────────────────────── #
        assign_box = QGroupBox("Assign Energies to Detected Peaks")
        assign_l   = QVBoxLayout(assign_box)
        assign_l.addWidget(QLabel("Toggle 🖱 Click to select peaks manually",
                                   styleSheet="font-size:10px; color:#556;"))

        self.tbl_detected = QTableWidget(0, 3)
        self.tbl_detected.setHorizontalHeaderLabels(["ADC", "Energy", "Label"])
        self.tbl_detected.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl_detected.setMinimumHeight(140)
        self.tbl_detected.setMaximumHeight(200)
        self.tbl_detected.itemDoubleClicked.connect(self._on_detected_table_dclick)
        assign_l.addWidget(self.tbl_detected)

        ab = QHBoxLayout(); ab.setSpacing(3)
        self.btn_assign = QPushButton("✏ Assign")
        self.btn_assign.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.btn_assign.clicked.connect(self._assign_selected_peak)
        self.btn_add_manual = QPushButton("+ Manual (all ch.)")
        self.btn_add_manual.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.btn_add_manual.clicked.connect(self._add_manual_point_all)
        self.btn_click_pk = QPushButton("🖱 Click")
        self.btn_click_pk.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.btn_click_pk.setCheckable(True)
        self.btn_click_pk.toggled.connect(self._toggle_click_mode)
        ab.addWidget(self.btn_assign)
        ab.addWidget(self.btn_add_manual)
        ab.addWidget(self.btn_click_pk)
        assign_l.addLayout(ab)
        layout.addWidget(assign_box)

        # ── Propagate / detect all channels ──────────────────────── #
        prop_box = QGroupBox("Detect Peaks in All Channels")
        prop_l   = QVBoxLayout(prop_box)
        prop_l.addWidget(QLabel(
            "After assigning an energy on the current channel, click below "
            "to find that same peak in every channel and add it to calib points.",
            wordWrap=True, styleSheet="font-size:10px; color:#555;"))

        self.lbl_last_assigned = QLabel("No peak assigned yet.")
        self.lbl_last_assigned.setWordWrap(True)
        self.lbl_last_assigned.setStyleSheet(
            "font-size:10px; color:#226; font-weight:bold; "
            "background:#eef; border-radius:3px; padding:2px;")
        prop_l.addWidget(self.lbl_last_assigned)

        win_h = QHBoxLayout()
        win_h.addWidget(QLabel("Search window (± ADC):"))
        self.sb_prop_window = QSpinBox()
        self.sb_prop_window.setRange(1, 10000); self.sb_prop_window.setValue(50)
        win_h.addWidget(self.sb_prop_window)
        prop_l.addLayout(win_h)

        self.btn_propagate = QPushButton("🔍  Detect Last Peak → All Channels")
        self.btn_propagate.setEnabled(False)
        self.btn_propagate.clicked.connect(self._detect_peaks_all_channels)
        prop_l.addWidget(self.btn_propagate)
        layout.addWidget(prop_box)

        # ── Calibration points table ──────────────────────────────── #
        cal_box = QGroupBox("Calibration Points (Current channel)")
        cal_l   = QVBoxLayout(cal_box)
        self.tbl_cal = QTableWidget(0, 3)
        self.tbl_cal.setHorizontalHeaderLabels(["ADC", "Energy", "Label"])
        self.tbl_cal.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl_cal.setMinimumHeight(120)
        self.tbl_cal.setMaximumHeight(200)
        cal_l.addWidget(self.tbl_cal)

        cb2 = QHBoxLayout()
        self.btn_del_cal    = QPushButton("− Remove")
        self.btn_del_cal.clicked.connect(self._delete_cal_point)
        self.btn_use_global = QPushButton("↑ Use Global")
        self.btn_use_global.clicked.connect(self._reset_to_global)
        cb2.addWidget(self.btn_del_cal)
        cb2.addWidget(self.btn_use_global)
        cal_l.addLayout(cb2)

        self.chk_override = QCheckBox("Exclude this channel from energy propagation")
        self.chk_override.toggled.connect(self._on_exclude_toggled)
        cal_l.addWidget(self.chk_override)
        layout.addWidget(cal_box)

        # ── Fit buttons ────────────────────────────────────────────── #
        self.btn_fit_one = QPushButton("▶  Fit This Channel")
        self.btn_fit_one.setEnabled(False)
        self.btn_fit_one.clicked.connect(self._fit_current)
        layout.addWidget(self.btn_fit_one)

        fit_all_h = QHBoxLayout()
        self.btn_fit_all = QPushButton("⚡  Fit All Channels")
        self.btn_fit_all.setEnabled(False)
        self.btn_fit_all.clicked.connect(self._fit_all)
        self.btn_fit_all.setToolTip(
            "Fit calibration model to all loaded channels.\n"
            "Uses the calibration points already assigned.\n\n"
            "TIP: First assign at least one energy per channel\n"
            "(or use 'Detect + Fit All' to auto-detect and fit).")
        fit_all_h.addWidget(self.btn_fit_all)

        self.btn_detect_fit_all = QPushButton("🔍⚡  Detect + Fit All")
        self.btn_detect_fit_all.setEnabled(False)
        self.btn_detect_fit_all.clicked.connect(self._detect_and_fit_all)
        self.btn_detect_fit_all.setToolTip(
            "For every loaded channel:\n"
            "  1. Run peak detection with current settings\n"
            "  2. Propagate energy assignments from current channel\n"
            "  3. Fit calibration model\n\n"
            "Requires at least one energy assignment on the current channel.")
        fit_all_h.addWidget(self.btn_detect_fit_all)
        layout.addLayout(fit_all_h)

        layout.addStretch()

        self.lbl_prop_status = QLabel("")
        self.lbl_prop_status.setWordWrap(True)
        self.lbl_prop_status.setStyleSheet("font-size:10px; color:#555;")
        layout.addWidget(self.lbl_prop_status)

        self.lbl_bad = QLabel("")
        self.lbl_bad.setWordWrap(True)
        self.lbl_bad.setStyleSheet("font-size:11px;")
        layout.addWidget(self.lbl_bad)

        scroll = QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setFrameShape(scroll.NoFrame)
        return scroll

    def _build_detail_tab(self):
        tab    = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        self.spectrum_canvas  = SpectrumCanvas()
        self.spectrum_canvas.peak_clicked.connect(self._on_peak_click)
        self.spectrum_toolbar = NavigationToolbar(self.spectrum_canvas, tab)
        layout.addWidget(self.spectrum_toolbar)
        layout.addWidget(self.spectrum_canvas, stretch=3)
        self.calib_canvas = CalibCurveCanvas()
        layout.addWidget(self.calib_canvas, stretch=2)
        self.tabs.addTab(tab, "🔬 Single Channel")

    def _build_grid_tab(self):
        tab    = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(4, 4, 4, 4)
        info = QLabel(
            "Thumbnail grid — click any thumbnail to open that channel. "
            "Green = good fit | Red = bad channel | Grey = not yet fitted.")
        info.setStyleSheet("color: #555; font-size: 11px;")
        layout.addWidget(info)
        self.overview_grid = OverviewGrid()
        self.overview_grid.channel_selected.connect(self._go_to_channel)
        layout.addWidget(self.overview_grid)
        self.tabs.addTab(tab, "📊 Overview Grid")

    def _build_trends_tab(self):
        tab    = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(4, 4, 4, 4)
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Parameter:"))
        self.cb_trend_param = QComboBox()
        self.cb_trend_param.addItems(["P0", "P1"])
        self.cb_trend_param.currentIndexChanged.connect(self._update_trends)
        ctrl.addWidget(self.cb_trend_param)
        ctrl.addStretch()
        layout.addLayout(ctrl)
        self.trend_canvas = TrendCanvas()
        layout.addWidget(self.trend_canvas)
        self.tabs.addTab(tab, "📈 Coefficient Trends")

    # ------------------------------------------------------------------ #
    # File loading
    # ------------------------------------------------------------------ #

    def _open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open ROOT File", "", "ROOT Files (*.root);;All Files (*)")
        if not path:
            return
        try:
            info = self.loader.open(path)
        except RuntimeError as e:
            QMessageBox.critical(self, "Backend Error", str(e)); return
        except Exception as e:
            QMessageBox.critical(self, "File Error", str(e)); return

        if not info["trees"] and not info["histograms"]:
            QMessageBox.warning(self, "Empty File",
                                "No TTrees or TH1 histograms found."); return

        dlg = LoadDialog(info, self)
        if dlg.exec_() != QDialog.Accepted:
            return

        try:
            if dlg.result_mode == "ttree":
                self.loader.load_from_ttree(
                    tree_name      = dlg.result_tree,
                    channel_branch = dlg.result_ch_branch,
                    adc_branch     = dlg.result_adc_branch,
                    n_bins         = dlg.result_nbins,
                    draw_mode      = dlg.result_draw_mode,
                    custom_expr    = dlg.result_custom_expr,
                    channel_ids    = dlg.result_ch_ids,
                    ch_first       = dlg.result_ch_first,
                    ch_last        = dlg.result_ch_last,
                    ch_step        = dlg.result_ch_step,
                    max_entries    = dlg.result_max_entries)
            else:
                self.loader.load_from_th1()
        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e)); return

        self.lbl_file.setText(os.path.basename(path))
        self._populate_channel_combo()
        self.btn_fit_all.setEnabled(True)
        self.btn_fit_one.setEnabled(True)
        self.btn_detect.setEnabled(True)
        self.btn_detect_fit_all.setEnabled(True)
        n       = len(self.loader.spectra)
        backend = self.loader.backend.upper()
        self._calib_spectrum_tab.inject(self.loader, self.fit_results)
        self.overview_grid.populate(self.loader.spectra, None)
        self.statusBar().showMessage(
            f"Loaded {n} channel(s) from {os.path.basename(path)}  [{backend} backend]")

    def _populate_channel_combo(self):
        self.cb_channel.blockSignals(True)
        self.cb_channel.clear()
        for ch_id in self.loader.get_channel_ids():
            self.cb_channel.addItem(f"Channel {ch_id}", ch_id)
        self.cb_channel.blockSignals(False)
        if self.cb_channel.count() > 0:
            self.cb_channel.setCurrentIndex(0)
            self._on_channel_changed(0)

    # ------------------------------------------------------------------ #
    # Channel navigation
    # ------------------------------------------------------------------ #

    def _on_channel_changed(self, idx: int):
        if idx < 0 or not self.cb_channel.count():
            return
        ch_id = self.cb_channel.currentData()
        if ch_id is None:
            return
        self.current_channel = ch_id
        self._detected_positions = []
        self._last_gauss_infos   = []
        self._update_detail_view(ch_id)
        self._update_override_ui(ch_id)

    def _prev_channel(self):
        i = self.cb_channel.currentIndex()
        if i > 0: self.cb_channel.setCurrentIndex(i - 1)

    def _next_channel(self):
        i = self.cb_channel.currentIndex()
        if i < self.cb_channel.count() - 1: self.cb_channel.setCurrentIndex(i + 1)

    def _go_to_channel(self, ch_id: int):
        for i in range(self.cb_channel.count()):
            if self.cb_channel.itemData(i) == ch_id:
                self.cb_channel.setCurrentIndex(i)
                self.tabs.setCurrentIndex(0)
                break

    def _update_detail_view(self, ch_id: int):
        sp = self.loader.get_spectrum(ch_id)
        if sp is None:
            return
        self.spectrum_canvas.plot_spectrum(
            sp.bin_centers, sp.counts,
            title=f"Channel {ch_id} — {sp.n_entries} entries  ({sp.source})")
        detected = self.peak_mgr.get_detected(ch_id)
        if detected:
            self.spectrum_canvas.draw_detected_peaks(detected, color="#FF6F00")
        assigned = self.peak_mgr.get_peaks(ch_id)
        if assigned:
            self.spectrum_canvas.draw_assigned_peaks(assigned)
        self.calib_canvas.plot_result(self.fit_results.get(ch_id))
        self._refresh_detected_table(ch_id)
        self._refresh_cal_table(ch_id)

        n_det = len(self.peak_mgr.get_detected(ch_id))
        excl  = " ⛔ excluded" if self.peak_mgr.is_excluded(ch_id) else ""
        info  = (f"File Loaded; Open Peak Detection or Assign Peaks manually\n"
                 f"Entries  : {sp.n_entries}\n"
                 f"Source   : {sp.source}\n"
                 f"Detected : {n_det} peak(s)\n"
                 f"Cal pts  : {self.peak_mgr.n_calibration_points(ch_id)}{excl}")
        r = self.fit_results.get(ch_id)
        if r:
            status = "❌ BAD" if r.bad_channel else "✅ OK"
            chi2_s = f"{r.chi2_ndf:.4f}" if r.ndf > 0 else "exact"
            info  += f"\nFit     : {status}  χ²/NDF={chi2_s}"
        self.lbl_ch_info.setStyleSheet(
            "font-size:10px; color:#226; font-weight:bold; "
            "background:#eef; border-radius:3px; padding:2px;")
        self.lbl_ch_info.setText(info)

    def _update_override_ui(self, ch_id: int):
        excluded = self.peak_mgr.is_excluded(ch_id)
        self.chk_override.blockSignals(True)
        self.chk_override.setChecked(excluded)
        self.chk_override.blockSignals(False)

    # ------------------------------------------------------------------ #
    # Detection method toggle
    # ------------------------------------------------------------------ #

    def _on_detect_method_changed(self, idx: int):
        method = self.cb_detect_method.currentData()
        self._tspec_widget.setVisible(method == "tspectrum")
        self._sw_widget.setVisible(method == "sliding_window")

    # ------------------------------------------------------------------ #
    # Peak detection — current channel
    # ------------------------------------------------------------------ #

    def _build_detect_kwargs(self) -> dict:
        """Collect all detection parameters into a dict for detect_peaks()."""
        method = self.cb_detect_method.currentData()
        return dict(
            sigma            = self.sb_sigma.value(),
            threshold        = self.sl_threshold.value() / 100.0,
            max_peaks        = self.sb_max_peaks.value(),
            backend          = self.loader.backend,
            pedestal_cut     = float(self.sb_pedestal.value()),
            tspec_mode       = self.cb_tspec_mode.currentData(),
            iterations       = self.sb_iterations.value(),
            bg_subtract      = self.chk_bg_subtract.isChecked(),
            bg_iterations    = self.sb_bg_iterations.value(),
            detection_method = method,
            sw_window_adc    = self.sb_sw_window.value(),
            sw_min_prominence= self.sb_sw_prominence.value() / 100.0,
            sw_smooth_sigma  = self.sb_sw_smooth.value(),
            gauss_confirm    = self.chk_gauss_confirm.isChecked(),
            gauss_window_adc = self.sb_gauss_window.value(),
        )

    def _detect_peaks(self):
        """Phase 1 — detect peaks on current channel."""
        ch_id = self.current_channel
        sp    = self.loader.get_spectrum(ch_id)
        if sp is None:
            return

        kwargs = self._build_detect_kwargs()

        positions, bg_array, gauss_infos = PeakManager.detect_peaks(
            sp.bin_centers, sp.counts, **kwargs)

        self._last_gauss_infos = gauss_infos
        self.peak_mgr.set_detected(ch_id, positions)

        # Redraw spectrum
        self.spectrum_canvas.plot_spectrum(
            sp.bin_centers, sp.counts,
            title=f"Channel {ch_id} — {sp.n_entries} entries")

        # SNIP background overlay (TSpectrum only)
        if kwargs["bg_subtract"] and bg_array is not None:
            bc = sp.bin_centers
            if kwargs["pedestal_cut"] > 0:
                bc = bc[bc >= kwargs["pedestal_cut"]]
            if len(bc) == len(bg_array):
                self.spectrum_canvas.draw_background(bc, bg_array)

        # Threshold line (TSpectrum only)
        if kwargs["detection_method"] == "tspectrum":
            self.spectrum_canvas.update_threshold_line(kwargs["threshold"])

        if not positions:
            self.lbl_detected.setText(
                "No peaks found — try lowering threshold or adjusting window size.")
        else:
            method_lbl = ("Sliding Window" if kwargs["detection_method"] == "sliding_window"
                          else ("TSpectrum HighRes" if kwargs["tspec_mode"] == "highres"
                                else "TSpectrum"))
            gauss_lbl = ""
            if kwargs["gauss_confirm"]:
                n_input = len(gauss_infos) if gauss_infos else len(positions)
                n_acc   = len(positions)
                n_rej   = n_input - n_acc
                gauss_lbl = f"  [Gauss: {n_acc} accepted, {n_rej} rejected]"

            self.lbl_detected.setText(
                f"{len(positions)} peak(s) [{method_lbl}]{gauss_lbl}:\n"
                + ", ".join(f"{p:.1f}" for p in positions))
            self.btn_propagate.setEnabled(True)
            self._refresh_detected_table(ch_id)

            # Draw accepted peaks (orange |)
            self.spectrum_canvas.draw_detected_peaks(positions, color="#FF6F00", style="|")

            # Draw rejected peaks (red x) and Gaussian fits
            if kwargs["gauss_confirm"] and gauss_infos:
                rejected = [info.adc for info in gauss_infos if not info.success]
                if rejected:
                    self.spectrum_canvas.draw_detected_peaks(
                        rejected, color="#e53935", style="x")
                # Overlay Gaussian curves on accepted peaks
                accepted_infos = [info for info in gauss_infos if info.success]
                if accepted_infos:
                    self.spectrum_canvas.draw_gauss_fits(
                        sp.bin_centers, sp.counts, accepted_infos)

        assigned = self.peak_mgr.get_peaks(ch_id)
        if assigned:
            self.spectrum_canvas.draw_assigned_peaks(assigned)

    def _on_threshold_changed(self, v: int):
        pct = v
        self.lbl_threshold.setText(f"{pct}%  of max")
        self.spectrum_canvas.update_threshold_line(v / 100.0)

    # ------------------------------------------------------------------ #
    # Detect + Fit All Channels
    # ------------------------------------------------------------------ #

    def _detect_and_fit_all(self):
        """
        For every channel:
          1. Detect peaks with current settings
          2. Propagate energy assignments from current channel peaks
          3. Fit calibration model
        """
        if not self.loader.spectra:
            return

        # Collect reference assignments from current channel
        src_peaks = self.peak_mgr.get_peaks(self.current_channel)
        if not src_peaks:
            QMessageBox.information(
                self, "No Reference Assignments",
                "Assign at least one energy to a peak on the current channel "
                "before running 'Detect + Fit All'.\n\n"
                "The energy assignments from the current channel will be "
                "propagated to all other channels by matching detected peaks.")
            return

        ref_assignments = [(p.adc_position, p.known_energy, p.label)
                           for p in src_peaks]

        ch_ids      = self.loader.get_channel_ids()
        model       = self.cb_model.currentData()
        custom_expr = self.le_custom_expr.text().strip()
        kwargs      = self._build_detect_kwargs()

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, len(ch_ids))
        self.btn_detect_fit_all.setEnabled(False)
        self.btn_fit_all.setEnabled(False)

        self._daf_worker = DetectAndFitWorker(
            self.fitter, self.loader, self.peak_mgr,
            ch_ids, model, custom_expr, kwargs, ref_assignments)
        self._daf_worker.progress.connect(
            lambda d, t, msg: (self.progress_bar.setValue(d),
                               self.statusBar().showMessage(msg)))
        self._daf_worker.finished.connect(self._on_detect_fit_all_done)
        self._daf_worker.error.connect(
            lambda e: QMessageBox.critical(self, "Error", e))
        self._daf_worker.start()

    def _on_detect_fit_all_done(self, results: dict):
        self.fit_results = results
        self._calib_spectrum_tab.inject(self.loader, self.fit_results)
        self.progress_bar.setVisible(False)
        self.btn_detect_fit_all.setEnabled(True)
        self.btn_fit_all.setEnabled(True)
        self.btn_export.setEnabled(True)
        self._update_detail_view(self.current_channel)
        self._update_bad_label()
        self._update_trends()
        self.overview_grid.populate(self.loader.spectra, self.fit_results)
        bad = self.fitter.bad_channels()
        self.statusBar().showMessage(
            f"Detect+Fit complete — {len(results)} channels | "
            f"{len(bad)} bad channel(s)")

    # ------------------------------------------------------------------ #
    # Peak assignment
    # ------------------------------------------------------------------ #

    def _on_detected_table_dclick(self, item):
        row = item.row()
        adc_item = self.tbl_detected.item(row, 0)
        if adc_item:
            self._open_assign_dialog(float(adc_item.text()))

    def _assign_selected_peak(self):
        row = self.tbl_detected.currentRow()
        if row < 0:
            return
        adc_item = self.tbl_detected.item(row, 0)
        if adc_item:
            self._open_assign_dialog(float(adc_item.text()))

    def _open_assign_dialog(self, adc_pos: float):
        dlg = AssignEnergyDialog(adc_pos, self)
        if dlg.exec_() != QDialog.Accepted:
            return
        adc   = dlg.sb_adc.value()
        eng   = dlg.sb_energy.value()
        lbl   = dlg.le_label.text()
        ch_id = self.current_channel

        all_channels = self.loader.get_channel_ids()
        zero_point   = (adc == 0.0 and eng == 0.0)

        if zero_point:
            for c in all_channels:
                if not self.peak_mgr.is_excluded(c):
                    self.peak_mgr.add_channel_peak(c, 0.0, 0.0, lbl or "origin")
            self._refresh_cal_table(ch_id)
            self._update_detail_view(ch_id)
            msg = f"Zero-point (0 ADC, 0 keV) added to {len(all_channels)} channel(s)."
            self.lbl_prop_status.setText(msg)
            self.lbl_last_assigned.setText("✔ Zero-point added to all channels.")
            self.statusBar().showMessage(msg)
            return

        self.peak_mgr.add_channel_peak(ch_id, adc, eng, lbl)
        self._last_assigned = {"adc": adc, "energy": eng, "label": lbl}
        self.btn_propagate.setEnabled(True)
        self.lbl_last_assigned.setText(
            f"Last assigned: {eng} keV  ({lbl})  at ADC {adc:.1f}  "
            f"→ click  🔍 Detect → All  to find in every channel")

        self._refresh_detected_table(ch_id)
        self._refresh_cal_table(ch_id)
        self._update_detail_view(ch_id)

        msg = (f"Assigned {eng} keV ({lbl}) at ADC {adc:.1f} on ch {ch_id}. "
               f"Click Detect → All to propagate.")
        self.lbl_prop_status.setText(msg)
        self.statusBar().showMessage(msg)

    def _add_manual_point_all(self):
        dlg = AssignEnergyDialog(0.0, self)
        if dlg.exec_() != QDialog.Accepted:
            return
        adc = dlg.sb_adc.value()
        eng = dlg.sb_energy.value()
        lbl = dlg.le_label.text()
        all_channels = self.loader.get_channel_ids()
        n_added = 0
        for c in all_channels:
            if not self.peak_mgr.is_excluded(c):
                if c in self.peak_mgr.channel_peaks:
                    self.peak_mgr.channel_peaks[c] = [
                        p for p in self.peak_mgr.channel_peaks[c]
                        if abs(p.known_energy - eng) > 0.01]
                self.peak_mgr.add_channel_peak(c, adc, eng, lbl)
                n_added += 1
        ch_id = self.current_channel
        self._refresh_cal_table(ch_id)
        self._update_detail_view(ch_id)
        msg = f"Manual point ({adc:.1f} ADC, {eng} keV, '{lbl}') added to {n_added} channel(s)."
        self.lbl_prop_status.setText(msg)
        self.statusBar().showMessage(msg)

    def _on_peak_click(self, x: float):
        self._open_assign_dialog(x)

    def _toggle_click_mode(self, checked: bool):
        self._click_mode = checked
        self.spectrum_canvas.set_click_mode(checked)
        self.statusBar().showMessage(
            "Click mode ON — click on spectrum to place a peak"
            if checked else "Click mode OFF")

    def _delete_cal_point(self):
        row   = self.tbl_cal.currentRow()
        ch_id = self.current_channel
        if row < 0 or ch_id < 0:
            return
        self.peak_mgr.remove_channel_peak(ch_id, row)
        self._refresh_cal_table(ch_id)
        self._update_detail_view(ch_id)

    def _reset_to_global(self):
        ch_id = self.current_channel
        if ch_id < 0:
            return
        self.peak_mgr.reset_channel(ch_id)
        self._refresh_cal_table(ch_id)
        self._update_detail_view(ch_id)

    def _on_exclude_toggled(self, checked: bool):
        ch_id = self.current_channel
        if ch_id < 0:
            return
        self.peak_mgr.set_excluded(ch_id, checked)
        status = "excluded from" if checked else "included in"
        self.statusBar().showMessage(
            f"Channel {ch_id} is now {status} energy propagation.")

    def _refresh_detected_table(self, ch_id: int):
        detected = self.peak_mgr.get_detected(ch_id)
        assigned = self.peak_mgr.get_peaks(ch_id)
        tol = 5.0

        def find_assigned(adc):
            best, best_d = None, tol
            for p in assigned:
                d = abs(p.adc_position - adc)
                if d < best_d:
                    best_d = d; best = p
            return best

        self.tbl_detected.setRowCount(0)
        for pos in detected:
            row = self.tbl_detected.rowCount()
            self.tbl_detected.insertRow(row)
            self.tbl_detected.setItem(row, 0, QTableWidgetItem(f"{pos:.2f}"))
            pk = find_assigned(pos)
            if pk:
                self.tbl_detected.setItem(row, 1,
                    QTableWidgetItem(f"{pk.known_energy:.2f} keV"))
                self.tbl_detected.setItem(row, 2,
                    QTableWidgetItem(pk.label))
                for col in range(3):
                    item = self.tbl_detected.item(row, col)
                    if item:
                        item.setBackground(QColor(180, 230, 180))
            else:
                self.tbl_detected.setItem(row, 1,
                    QTableWidgetItem("—  (double-click to assign)"))
                self.tbl_detected.setItem(row, 2, QTableWidgetItem(""))

    def _refresh_cal_table(self, ch_id: int):
        peaks = self.peak_mgr.get_peaks(ch_id)
        self.tbl_cal.setRowCount(0)
        for p in peaks:
            row = self.tbl_cal.rowCount()
            self.tbl_cal.insertRow(row)
            self.tbl_cal.setItem(row, 0, QTableWidgetItem(f"{p.adc_position:.2f}"))
            self.tbl_cal.setItem(row, 1, QTableWidgetItem(f"{p.known_energy:.2f}"))
            self.tbl_cal.setItem(row, 2, QTableWidgetItem(p.label))

    # ------------------------------------------------------------------ #
    # Detect peaks in ALL channels
    # ------------------------------------------------------------------ #

    def _detect_peaks_all_channels(self):
        if not self._last_assigned:
            QMessageBox.information(self, "No Peak Assigned",
                "Detect a peak on the current channel and assign its energy first.")
            return

        ref_adc  = self._last_assigned["adc"]
        energy   = self._last_assigned["energy"]
        label    = self._last_assigned["label"]
        window   = self.sb_prop_window.value()
        all_channels = self.loader.get_channel_ids()
        src_ch       = self.current_channel
        kwargs       = self._build_detect_kwargs()

        n_ok = 0; n_excl = 0; n_failed = 0
        n_total = len([c for c in all_channels if c != src_ch])

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, max(n_total, 1))

        for i, ch_id in enumerate(all_channels):
            if ch_id == src_ch:
                continue
            self.progress_bar.setValue(i + 1)
            QApplication.processEvents()

            if self.peak_mgr.is_excluded(ch_id):
                n_excl += 1; continue

            sp = self.loader.get_spectrum(ch_id)
            if sp is None:
                n_failed += 1; continue

            lo   = ref_adc - window
            hi   = ref_adc + window
            mask = (sp.bin_centers >= lo) & (sp.bin_centers <= hi)
            if mask.sum() < 3:
                n_failed += 1; continue

            win_centers = sp.bin_centers[mask]
            win_counts  = sp.counts[mask]

            candidates, _, _ = PeakManager.detect_peaks(
                win_centers, win_counts,
                sigma=kwargs["sigma"],
                threshold=kwargs["threshold"],
                max_peaks=kwargs["max_peaks"],
                backend=kwargs["backend"],
                pedestal_cut=0.0,
                tspec_mode=kwargs["tspec_mode"],
                iterations=kwargs["iterations"],
                bg_subtract=kwargs["bg_subtract"],
                bg_iterations=kwargs["bg_iterations"],
                detection_method=kwargs["detection_method"],
                sw_window_adc=min(kwargs["sw_window_adc"], float(window)),
                sw_min_prominence=kwargs["sw_min_prominence"],
                sw_smooth_sigma=kwargs["sw_smooth_sigma"],
                gauss_confirm=False)

            if candidates:
                best_adc = min(candidates, key=lambda x: abs(x - ref_adc))
            else:
                from scipy.ndimage import gaussian_filter1d
                smooth   = gaussian_filter1d(win_counts.astype(float),
                                             sigma=max(1, kwargs["sigma"]))
                idx_max  = int(np.argmax(smooth))
                hw       = max(2, int(len(smooth) * 0.1))
                lo_i     = max(0, idx_max - hw)
                hi_i     = min(len(smooth), idx_max + hw + 1)
                wslice   = smooth[lo_i:hi_i]
                cslice   = win_centers[lo_i:hi_i]
                best_adc = (float(np.sum(cslice * wslice) / wslice.sum())
                            if wslice.sum() > 0 else ref_adc)

            if ch_id in self.peak_mgr.channel_peaks:
                self.peak_mgr.channel_peaks[ch_id] = [
                    p for p in self.peak_mgr.channel_peaks[ch_id]
                    if abs(p.known_energy - energy) > 0.01]
            self.peak_mgr.add_channel_peak(ch_id, best_adc, energy, label)
            n_ok += 1

        self.progress_bar.setVisible(False)
        self._update_detail_view(src_ch)

        msg = (f"Added {energy} keV ({label}) to calib points of "
               f"{n_ok}/{n_total} channel(s).")
        if n_excl:   msg += f"  ({n_excl} excluded)"
        if n_failed: msg += f"  ({n_failed} had no spectrum in window)"
        self.lbl_prop_status.setText(msg)
        self.lbl_last_assigned.setText(
            f"✔ {energy} keV ({label}) added to {n_ok} channel(s).")
        self.statusBar().showMessage(msg)

    # ------------------------------------------------------------------ #
    # Fitting
    # ------------------------------------------------------------------ #

    def _fit_current(self):
        ch_id = self.current_channel
        if ch_id < 0:
            return
        adc, eng = self.peak_mgr.get_calibration_points(ch_id)
        if len(adc) == 0:
            QMessageBox.warning(self, "No Peaks", "Assign calibration peaks first.")
            return
        model       = self.cb_model.currentData()
        custom_expr = self.le_custom_expr.text().strip()
        result = self.fitter.fit_channel(ch_id, adc, eng, model, custom_expr)
        self.fit_results[ch_id] = result
        self._update_detail_view(ch_id)
        self._update_bad_label()
        self.overview_grid.populate(self.loader.spectra, self.fit_results)
        self.btn_export.setEnabled(True)

    def _fit_all(self):
        ch_ids = self.loader.get_channel_ids()
        if not ch_ids:
            return
        model       = self.cb_model.currentData()
        custom_expr = self.le_custom_expr.text().strip()

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, len(ch_ids))
        self.btn_fit_all.setEnabled(False)

        self._worker = FitWorker(self.fitter, ch_ids, self.peak_mgr,
                                  model, custom_expr)
        self._worker.progress.connect(lambda d, t: self.progress_bar.setValue(d))
        self._worker.finished.connect(self._on_fit_all_done)
        self._worker.error.connect(lambda e: QMessageBox.critical(self, "Fit Error", e))
        self._worker.start()

    def _on_fit_all_done(self, results: dict):
        self.fit_results = results
        self._calib_spectrum_tab.inject(self.loader, self.fit_results)
        self.progress_bar.setVisible(False)
        self.btn_fit_all.setEnabled(True)
        self.btn_export.setEnabled(True)
        self._update_detail_view(self.current_channel)
        self._update_bad_label()
        self._update_trends()
        self.overview_grid.populate(self.loader.spectra, self.fit_results)
        bad = self.fitter.bad_channels()
        self.statusBar().showMessage(
            f"Fit complete — {len(results)} channels | {len(bad)} bad channel(s)")

    # ------------------------------------------------------------------ #
    # Export
    # ------------------------------------------------------------------ #

    def _export(self):
        if not self.fit_results:
            return
        out_dir = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if not out_dir:
            return
        try:
            paths = OutputWriter.write_all(
                self.fit_results, out_dir, source_file=self.loader.filename)
            QMessageBox.information(self, "Exported",
                "Files saved:\n" + "\n".join(paths))
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    # ------------------------------------------------------------------ #
    # Trends & summary
    # ------------------------------------------------------------------ #

    def _update_trends(self):
        if not self.fit_results:
            return
        idx  = self.cb_trend_param.currentIndex()
        name = self.cb_trend_param.currentText()
        self.trend_canvas.plot_trends(self.fit_results, idx, name)

    def _update_bad_label(self):
        bad = self.fitter.bad_channels()
        if bad:
            ids = ", ".join(str(b) for b in bad[:20])
            self.lbl_bad.setText(
                f"⚠ {len(bad)} bad channel(s):\n{ids}"
                + ("..." if len(bad) > 20 else ""))
            self.lbl_bad.setStyleSheet("color: #c62828; font-size: 11px;")
        else:
            self.lbl_bad.setText("✅ All channels OK")
            self.lbl_bad.setStyleSheet("color: #2e7d32; font-size: 11px;")

    # ------------------------------------------------------------------ #
    # Calibrated Spectrum + Resolution tabs
    # ------------------------------------------------------------------ #

    def _build_calib_spectrum_tab(self):
        self._calib_spectrum_tab = CalibratedSpectrumTab()
        self._calib_spectrum_tab.inject(self.loader, self.fit_results)
        self._calib_spectrum_tab.channel_for_resolution.connect(
            self._send_to_resolution_tab)
        self.tabs.addTab(self._calib_spectrum_tab, "⚡ Calibrated Spectrum")

    def _build_resolution_tab(self):
        self._resolution_tab = ResolutionTab()
        self._resolution_tab._main_window = self
        self.tabs.addTab(self._resolution_tab, "📐 Resolution")

    def _send_to_resolution_tab(self, ch_id: int, cal):
        self._resolution_tab.receive_spectrum(ch_id, cal)
        for i in range(self.tabs.count()):
            if "Resolution" in self.tabs.tabText(i):
                self.tabs.setCurrentIndex(i)
                break

    # ------------------------------------------------------------------ #
    # Model change
    # ------------------------------------------------------------------ #

    def _on_model_changed(self, idx: int):
        model = self.cb_model.currentData()
        self.le_custom_expr.setEnabled(model == "custom")
        param_names = {
            "linear":        ["P0", "P1"],
            "nonlinear":     ["P0", "P1", "P2", "P3"],
            "nonlinear_3pt": ["P0", "P1", "P3"],
            "custom":        ["p0", "p1", "p2", "p3"],
        }.get(model, ["P0", "P1"])
        self.cb_trend_param.blockSignals(True)
        self.cb_trend_param.clear()
        self.cb_trend_param.addItems(param_names)
        self.cb_trend_param.blockSignals(False)

    # ------------------------------------------------------------------ #
    # Style
    # ------------------------------------------------------------------ #

    def _apply_style(self):
        self.setStyleSheet("""
            QMainWindow, QWidget { background-color: #f5f5f5;
                                    color: #212121; font-size: 13px; }
            #toolbar { background-color: #ffffff;
                        border-bottom: 1px solid #e0e0e0; }
            #fileLabel { color: #1565c0; font-weight: bold; padding: 0 8px; }
            QPushButton {
                background-color: #ffffff; color: #212121;
                border: 1px solid #bdbdbd; border-radius: 5px;
                padding: 5px 10px; min-width: 70px;
            }
            QPushButton:hover  { background-color: #e3f2fd; border-color: #1565c0; }
            QPushButton:pressed { background-color: #bbdefb; }
            QPushButton:checked { background-color: #1565c0; color: #fff; }
            QPushButton:disabled { color: #9e9e9e; background-color: #eeeeee; }
            QGroupBox {
                border: 1px solid #e0e0e0; border-radius: 6px;
                margin-top: 8px; padding-top: 4px;
                font-weight: bold; color: #1565c0; background-color: #ffffff;
            }
            QGroupBox QPushButton { padding: 4px 4px; min-width: 0px; }
            QGroupBox::title { subcontrol-origin: margin; left: 8px; }
            QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {
                background-color: #ffffff; border: 1px solid #bdbdbd;
                border-radius: 4px; padding: 3px 6px; color: #212121;
            }
            QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus {
                border-color: #1565c0; }
            QTableWidget {
                background-color: #ffffff; gridline-color: #e0e0e0;
                border: 1px solid #e0e0e0; border-radius: 4px;
            }
            QTableWidget::item:selected { background-color: #bbdefb; color: #212121; }
            QHeaderView::section {
                background-color: #f5f5f5; color: #1565c0;
                border: none; border-bottom: 1px solid #e0e0e0;
                padding: 4px; font-weight: bold;
            }
            QTabWidget::pane  { border: 1px solid #e0e0e0; background: #ffffff; }
            QTabBar::tab {
                background: #eeeeee; color: #616161;
                padding: 6px 16px; border-radius: 4px 4px 0 0;
            }
            QTabBar::tab:selected { background: #ffffff; color: #1565c0;
                                     border-bottom: 2px solid #1565c0; }
            QTabBar::tab:hover { background: #e3f2fd; }
            QScrollArea { border: none; }
            QStatusBar   { background: #ffffff; color: #616161;
                            border-top: 1px solid #e0e0e0; }
            QCheckBox { color: #212121; }
            QCheckBox::indicator {
                width: 14px; height: 14px;
                border: 1px solid #bdbdbd; border-radius: 3px; background: #ffffff;
            }
            QCheckBox::indicator:checked { background: #1565c0; border-color: #1565c0; }
            QSlider::groove:horizontal {
                height: 4px; background: #e0e0e0; border-radius: 2px; }
            QSlider::handle:horizontal {
                width: 14px; height: 14px; margin: -5px 0;
                background: #1565c0; border-radius: 7px; }
            QSplitter::handle { background: #e0e0e0; }
        """)
        plt.style.use("default")