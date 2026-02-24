"""
MainWindow — DetectorCalibGUI
PyQt5 application. Calibration formula: E = P0·(P1^Q)^P2 + P3·Q − P0
Peak detection via ROOT TSpectrum (PyROOT) or scipy (uproot fallback).
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
from src.peak_manager import PeakManager, Peak, CRYSTAL_KNOWN_LINES
from src.calib_fitter import CalibrationFitter, MODELS
from src.output_writer import OutputWriter
from src.calib_spectrum_tab import CalibratedSpectrumTab
from src.resolution_tab import ResolutionTab


# ======================================================================== #
# Worker thread
# ======================================================================== #

class FitWorker(QThread):
    progress = pyqtSignal(int, int)
    finished = pyqtSignal(dict)
    error    = pyqtSignal(str)

    def __init__(self, fitter, channel_ids, peak_manager, model,
                 custom_expr=""):
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


# ======================================================================== #
# Collapsible box widget
# ======================================================================== #

class CollapsibleBox(QWidget):
    """A titled panel that can be toggled open/closed by clicking the header."""

    def __init__(self, title: str, parent=None, collapsed: bool = False):
        super().__init__(parent)
        self._collapsed = collapsed

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ── Header button ────────────────────────────────────────────── #
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
            "}"
        )
        self._title = title
        self._update_btn_text()
        outer.addWidget(self._btn)

        # ── Content area ─────────────────────────────────────────────── #
        self._content = QWidget()
        self._content.setObjectName("collapsibleContent")
        self._content.setStyleSheet(
            "#collapsibleContent {"
            "  border: 1px solid #90caf9; border-top: none;"
            "  border-radius: 0 0 4px 4px;"
            "  background: #fafafa;"
            "  padding: 2px;"
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

    def layout(self):   # noqa — return inner layout for addLayout calls
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
        self._threshold_line  = None   # horizontal threshold indicator
        self._current_max_cts = None   # max counts of currently plotted spectrum
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
        """Draw/update a horizontal dashed line at threshold × max_counts.
        This is exactly what TSpectrum uses as the minimum peak height."""
        if self._current_max_cts is None:
            return
        level = threshold * self._current_max_cts
        if level < 0.5:
            level = 0.5
        # Remove previous line
        if self._threshold_line is not None:
            try:
                self._threshold_line.remove()
            except Exception:
                pass
            self._threshold_line = None
        self._threshold_line = self.ax.axhline(
            level, color="#e53935", linestyle="--", linewidth=1.2,
            alpha=0.85, zorder=6,
            label=f"TSpectrum threshold  ({threshold:.0%} × max = {level:.0f} cts)")
        # Update or add legend entry
        handles, labels = self.ax.get_legend_handles_labels()
        # Remove duplicates
        seen = {}
        for h, l in zip(handles, labels):
            seen[l] = h
        self.ax.legend(seen.values(), seen.keys(), fontsize=7, loc="upper right")
        self.fig.tight_layout(pad=1.5)
        self.draw()

    def draw_detected_peaks(self, positions: list):
        for line in self._peak_markers:
            try: line.remove()
            except Exception: pass
        self._peak_markers.clear()
        ymax = self.ax.get_ylim()[1]
        for x in positions:
            line = self.ax.axvline(x, color="#FF6F00", linestyle=":",
                                    linewidth=1.4, alpha=0.9)
            self.ax.text(x, ymax * 0.6, f" {x:.0f}",
                          color="#FF6F00", fontsize=6.5,
                          rotation=90, va="top")
            self._peak_markers.append(line)
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
        """Overlay the estimated SNIP background on the spectrum."""
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
        """Draw known crystal emission lines in ADC space using the
        calibration fit to convert keV → ADC (inverted numerically)."""
        if not known_lines or fit_result is None or not fit_result.success:
            return
        from scipy.optimize import brentq
        ymax = self.ax.get_ylim()[1]
        ymin = self.ax.get_ylim()[0]
        for line_info in known_lines:
            energy = line_info["energy"]
            color  = line_info.get("color", "#7b1fa2")
            label  = line_info["label"]
            # Invert calibration: find ADC where energy_at(ADC) = energy
            try:
                # Search over a wide ADC range
                adc_lo, adc_hi = 0.0, 1e6
                f_lo = fit_result.energy_at(adc_lo) - energy
                f_hi = fit_result.energy_at(adc_hi) - energy
                if f_lo * f_hi > 0:
                    continue   # energy outside calibration range
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

        adc_limit = result.adc_max   # last assigned peak — no extrapolation beyond
        x_fit = np.linspace(max(0, adc.min() * 0.8), adc_limit, 500)
        y_fit = result.energy_at(x_fit)

        # Check that energy_at returned valid values
        if np.all(np.isnan(y_fit)):
            self.ax.text(0.5, 0.5,
                          f"Cannot plot model '{result.model}'",
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
            self.ax.annotate(f"{e:.1f} keV",
                              xy=(a, e), xytext=(4, 4),
                              textcoords="offset points",
                              fontsize=7, color="#c62828")

        # Mark the hard limit with a vertical dashed line
        e_at_limit = result.energy_at(np.array([adc_limit]))[0]
        if not np.isnan(e_at_limit):
            self.ax.axvline(adc_limit, color="#e65100", linestyle="--",
                             linewidth=1.0, alpha=0.7,
                             label=f"Fit limit  ADC {adc_limit:.0f}"
                                   f" = {e_at_limit:.1f} keV")

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
        # Clear existing widgets
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
    """Configure data source after opening a ROOT file."""

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
        self.cb_draw_mode.addItem(
            'Filter     Draw("adc", "channelID==N")', "filter")
        self.cb_draw_mode.addItem(
            'Array      Draw("adc[channelID]")', "array")
        self.cb_draw_mode.addItem(
            'Custom     user expression (%d = channel number)', "custom")

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

        # Channel range — only shown for array/custom modes
        self.ch_range_box = QGroupBox("Channel Range")
        cr = QFormLayout(self.ch_range_box)

        range_h = QHBoxLayout()
        self.sb_ch_first = QSpinBox()
        self.sb_ch_first.setRange(0, 999999)
        self.sb_ch_first.setValue(0)
        self.sb_ch_last  = QSpinBox()
        self.sb_ch_last.setRange(-1, 999999)
        self.sb_ch_last.setValue(-1)
        self.sb_ch_last.setSpecialValueText("Auto")
        self.sb_ch_step  = QSpinBox()
        self.sb_ch_step.setRange(1, 1000)
        self.sb_ch_step.setValue(1)
        range_h.addWidget(QLabel("First:"))
        range_h.addWidget(self.sb_ch_first)
        range_h.addWidget(QLabel("  Last:"))
        range_h.addWidget(self.sb_ch_last)
        range_h.addWidget(QLabel("  Step:"))
        range_h.addWidget(self.sb_ch_step)
        cr.addRow("Range:", range_h)

        self.le_ch_list = QLineEdit()
        self.le_ch_list.setPlaceholderText(
            "Optional: comma-separated list, e.g.  0,1,2,5,10")
        cr.addRow("Custom list:", self.le_ch_list)

        self.sb_max_entries = QSpinBox()
        self.sb_max_entries.setRange(0, 100_000_000)
        self.sb_max_entries.setValue(0)
        self.sb_max_entries.setSpecialValueText("All entries")
        self.sb_max_entries.setSingleStep(10000)
        cr.addRow("Max entries:", self.sb_max_entries)

        layout.addWidget(self.ch_range_box)

        self.hist_group = QGroupBox("Histogram Options")
        QVBoxLayout(self.hist_group).addWidget(
            QLabel(f"{len(file_info['histograms'])} histogram(s) found — "
                   "all will be loaded."))
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
        mode    = self.cb_draw_mode.currentData()
        is_filt = mode == "filter"
        is_arr  = mode == "array"
        is_cust = mode == "custom"

        self.cb_channel_branch.setEnabled(is_filt)
        self.le_draw_custom.setEnabled(is_cust)
        # Channel range always visible — useful in all modes
        self.ch_range_box.setVisible(True)
        self._update_preview()

    def _update_preview(self):
        mode = self.cb_draw_mode.currentData()
        ch   = self.cb_channel_branch.currentText()
        adc  = self.cb_adc_branch.currentText()
        if mode == "filter":
            self.lbl_draw_preview.setText(
                f'Preview:  tree.Draw("{adc}", "{ch}==N")')
        elif mode == "array":
            self.lbl_draw_preview.setText(
                f'Preview:  tree.Draw("{adc}[N]")')
        elif mode == "custom":
            expr = self.le_draw_custom.text() or "%d"
            self.lbl_draw_preview.setText(
                f'Preview:  tree.Draw("{expr}") with %d→channel')

    def _toggle(self):
        use_tree = self.rb_tree.isChecked()
        self.tree_group.setVisible(use_tree)
        self.hist_group.setVisible(not use_tree)
        # Channel range always visible for TTree, hidden for TH1
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
                    self.result_ch_ids = [int(x.strip())
                                           for x in raw.split(",") if x.strip()]
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
    """Assign a known energy to a detected peak ADC position."""

    COMMON_SOURCES = [
        ("511.0",   "Na-22 / annihilation"),
        ("1274.5",  "Na-22"),
        ("661.7",   "Cs-137"),
        ("1332.5",  "Co-60"),
        ("1173.2",  "Co-60"),
        ("122.1",   "Co-57"),
        ("344.3",   "Eu-152"),
        ("1460.8",  "K-40"),
        ("1764.5",  "Ra-226"),
        ("2614.5",  "Tl-208"),
        ("59.5",    "Am-241"),
        ("88.0",    "Cd-109"),
    ]

    def __init__(self, adc_position: float = 0.0, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Assign Known Energy to Peak")
        self.setMinimumWidth(480)
        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.sb_adc = QDoubleSpinBox()
        self.sb_adc.setRange(0, 1e9)
        self.sb_adc.setDecimals(3)
        self.sb_adc.setValue(adc_position)

        self.sb_energy = QDoubleSpinBox()
        self.sb_energy.setRange(0, 1e9)
        self.sb_energy.setDecimals(3)
        self.sb_energy.setValue(0.0)

        self.le_label = QLineEdit()
        self.le_label.setPlaceholderText("optional label, e.g.  Cs-137  661.7 keV")

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


# ======================================================================== #
# Progress dialog
# ======================================================================== #

class _LoadingProgressDialog(QDialog):
    def __init__(self, title: str, n_channels: int, parent=None):
        super().__init__(parent,
            Qt.WindowTitleHint | Qt.CustomizeWindowHint)
        self.setWindowTitle(title)
        self.setFixedWidth(380)
        layout = QVBoxLayout(self)
        self.lbl = QLabel(f"Loading 0 / {n_channels} channels…")
        layout.addWidget(self.lbl)
        self.bar = QProgressBar()
        self.bar.setRange(0, n_channels)
        self.bar.setValue(0)
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
        self.setWindowTitle("DetectorCalibGUI — Energy Calibration & Resolution")
        self.setMinimumSize(1350, 880)

        self.loader           = ROOTFileLoader()
        self.peak_mgr         = PeakManager()
        self.fitter           = CalibrationFitter()
        self.fit_results: dict = {}
        self.current_channel: int = -1
        self._detected_positions: list = []
        self._click_mode = False
        self._last_assigned: dict | None = None  # {adc, energy, label}
        self.channel_crystal: dict = {}    # ch_id -> 'lyso'|'gagg'|'generic'

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
        left.setMaximumWidth(410)
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
        self.cb_model.addItem("Linear  E = P0 + P1·Q",                            "linear")
        self.cb_model.addItem("Nonlinear  E = P0·(P1^Q)^P2 + P3·Q − P0",         "nonlinear")
        self.cb_model.addItem("Nonlinear 3-pt  E = P0·P1^Q + P3·Q − P0 (P2=1)",  "nonlinear_3pt")
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
        layout.setSpacing(10)

        # ── Channel navigation ──────────────────────────────────────── #
        nav_box = QGroupBox("Channel")
        nav_l   = QVBoxLayout(nav_box)
        nav_h   = QHBoxLayout()
        self.btn_prev = QPushButton("◀")
        self.btn_prev.setFixedWidth(100)
        self.btn_prev.clicked.connect(self._prev_channel)
        self.cb_channel = QComboBox()
        self.cb_channel.currentIndexChanged.connect(self._on_channel_changed)
        self.btn_next = QPushButton("▶")
        self.btn_next.setFixedWidth(100)
        self.btn_next.clicked.connect(self._next_channel)
        nav_h.addWidget(self.btn_prev)
        nav_h.addWidget(self.cb_channel, stretch=1)
        nav_h.addWidget(self.btn_next)
        nav_l.addLayout(nav_h)
        self.lbl_ch_info = QLabel("—")
        self.lbl_ch_info.setWordWrap(True)
        self.lbl_ch_info.setStyleSheet("font-size: 11px; color: #424242;")
        nav_l.addWidget(self.lbl_ch_info)

        # ── Per-channel crystal type tag ──────────────────────────── #
        cryst_tag_h = QHBoxLayout()
        cryst_tag_h.addWidget(QLabel("Crystal type:"))
        self.cb_ch_crystal = QComboBox()
        self.cb_ch_crystal.addItem("Generic",  "generic")
        self.cb_ch_crystal.addItem("GAGG",     "gagg")
        self.cb_ch_crystal.addItem("LYSO",     "lyso")
        self.cb_ch_crystal.setToolTip(
            "Tag this channel's crystal type.\n"
            "LYSO channels can use Lu-176 internal lines (88, 202, 307 keV)\n"
            "as automatic calibration points.\n"
            "The dropdown shows LYSO channels in purple.")
        self.cb_ch_crystal.currentIndexChanged.connect(self._on_ch_crystal_changed)
        cryst_tag_h.addWidget(self.cb_ch_crystal, stretch=1)

        self.btn_tag_all_lyso = QPushButton("LYSO")
        self.btn_tag_all_lyso.setToolTip("Mark every channel as LYSO.")
        self.btn_tag_all_lyso.clicked.connect(
            lambda: self._tag_all_channels("lyso"))
        self.btn_tag_all_gagg = QPushButton("GAGG")
        self.btn_tag_all_gagg.setToolTip("Mark every channel as GAGG.")
        self.btn_tag_all_gagg.clicked.connect(
            lambda: self._tag_all_channels("gagg"))
        cryst_tag_h.addWidget(self.btn_tag_all_lyso)
        cryst_tag_h.addWidget(self.btn_tag_all_gagg)
        nav_l.addLayout(cryst_tag_h)
        layout.addWidget(nav_box)

        # ── Peak detection ──────────────────────────────────────────── #
        # ── Peak detection (collapsible) ────────────────────────────── #
        det_box = CollapsibleBox("🔍  Peak Detection  (TSpectrum / scipy)",
                                  collapsed=True)

        thresh_h = QHBoxLayout()
        thresh_h.addWidget(QLabel("Threshold:"))
        self.sl_threshold = QSlider(Qt.Horizontal)
        self.sl_threshold.setRange(1, 50)
        self.sl_threshold.setValue(5)
        self.sl_threshold.setToolTip(
            "TSpectrum threshold: fraction of the tallest peak height.\n"
            "A peak is only reported if its height \u2265 threshold \u00d7 max_peak.\n"
            "Lower = find more (smaller) peaks.\n"
            "Max 50% — at 100% only the single maximum bin qualifies.")
        self.lbl_threshold = QLabel("5%  of max")
        self.lbl_threshold.setMinimumWidth(80)
        self.sl_threshold.valueChanged.connect(self._on_threshold_changed)
        thresh_h.addWidget(self.sl_threshold, stretch=1)
        thresh_h.addWidget(self.lbl_threshold)
        det_box.addLayout(thresh_h)

        mode_h = QHBoxLayout()
        mode_h.addWidget(QLabel("Search mode:"))
        self.cb_tspec_mode = QComboBox()
        self.cb_tspec_mode.addItem("Standard  (Search)",               "standard")
        self.cb_tspec_mode.addItem("High Resolution  (SearchHighRes)", "highres")
        self.cb_tspec_mode.setCurrentIndex(1)
        self.cb_tspec_mode.setToolTip(
            "Standard: fast, good for well-separated peaks.\n"
            "High Resolution: deconvolution — resolves closely-spaced peaks.")
        mode_h.addWidget(self.cb_tspec_mode, stretch=1)
        det_box.addLayout(mode_h)

        iter_h = QHBoxLayout()
        iter_h.addWidget(QLabel("Iterations (HighRes):"))
        self.sb_iterations = QSpinBox()
        self.sb_iterations.setRange(1, 50)
        self.sb_iterations.setValue(10)
        self.sb_iterations.setToolTip(
            "Deconvolution iterations for SearchHighRes.\n"
            "More = sharper, slower.  Typical: 5\u201320.")
        iter_h.addWidget(self.sb_iterations)
        det_box.addLayout(iter_h)

        sigma_h = QHBoxLayout()
        sigma_h.addWidget(QLabel("Sigma (bins):"))
        self.sb_sigma = QDoubleSpinBox()
        self.sb_sigma.setRange(0.5, 20.0)
        self.sb_sigma.setSingleStep(0.5)
        self.sb_sigma.setValue(2.0)
        sigma_h.addWidget(self.sb_sigma)
        det_box.addLayout(sigma_h)

        maxpk_h = QHBoxLayout()
        maxpk_h.addWidget(QLabel("Max peaks:"))
        self.sb_max_peaks = QSpinBox()
        self.sb_max_peaks.setRange(1, 30)
        self.sb_max_peaks.setValue(10)
        maxpk_h.addWidget(self.sb_max_peaks)
        det_box.addLayout(maxpk_h)

        # ── Refinement options ──────────────────────────────────────────── #
        refine_h = QHBoxLayout()
        self.chk_refine_peaks = QCheckBox("Refine peaks (sliding + Gaussian fit)")
        self.chk_refine_peaks.setChecked(True)
        self.chk_refine_peaks.setToolTip(
            "After detection, refine peak centers using a sliding window\n"
            "and local Gaussian fit. Rejects poorly fitted peaks.")
        refine_h.addWidget(self.chk_refine_peaks)
        det_box.addLayout(refine_h)

        refwin_h = QHBoxLayout()
        refwin_h.addWidget(QLabel("Refine window (ADC):"))
        self.sb_refine_window = QSpinBox()
        self.sb_refine_window.setRange(5, 200)
        self.sb_refine_window.setValue(30)
        self.sb_refine_window.setSingleStep(5)
        self.sb_refine_window.setToolTip(
            "Half-width of the local window used for Gaussian refinement.\n"
            "Typical: 20–50 ADC.")
        refwin_h.addWidget(self.sb_refine_window)
        det_box.addLayout(refwin_h)

        ped_h = QHBoxLayout()
        ped_h.addWidget(QLabel("Pedestal cut (ADC):"))
        self.sb_pedestal = QSpinBox()
        self.sb_pedestal.setRange(0, 100000)
        self.sb_pedestal.setValue(0)
        self.sb_pedestal.setSingleStep(10)
        self.sb_pedestal.setToolTip(
            "Ignore all peaks below this ADC value.\n"
            "Set above the pedestal to avoid false detections.  0 = disabled.")
        ped_h.addWidget(self.sb_pedestal)
        det_box.addLayout(ped_h)

        bg_h = QHBoxLayout()
        self.chk_bg_subtract = QCheckBox("Subtract background (SNIP)")
        self.chk_bg_subtract.setChecked(True)
        self.chk_bg_subtract.setToolTip(
            "Use TSpectrum::Background() (SNIP) before peak search.\n"
            "Removes Lu-176 beta continuum in LYSO spectra.")
        bg_h.addWidget(self.chk_bg_subtract)
        self.sb_bg_iterations = QSpinBox()
        self.sb_bg_iterations.setRange(1, 100)
        self.sb_bg_iterations.setValue(20)
        self.sb_bg_iterations.setToolTip(
            "SNIP iterations.  More = smoother BG.  LYSO: try 20\u201340.")
        bg_h.addWidget(QLabel("iter:"))
        bg_h.addWidget(self.sb_bg_iterations)
        det_box.addLayout(bg_h)

        cryst_h = QHBoxLayout()
        cryst_h.addWidget(QLabel("Crystal:"))
        self.cb_crystal = QComboBox()
        self.cb_crystal.addItem("Generic", "generic")
        self.cb_crystal.addItem("GAGG",    "gagg")
        self.cb_crystal.addItem("LYSO",    "lyso")
        self.cb_crystal.setToolTip(
            "LYSO: overlays Lu-176 lines (88, 202, 307 keV) after calibration.")
        cryst_h.addWidget(self.cb_crystal, stretch=1)
        self.chk_show_known = QCheckBox("Show known lines")
        self.chk_show_known.setChecked(True)
        cryst_h.addWidget(self.chk_show_known)
        det_box.addLayout(cryst_h)

        self.btn_detect = QPushButton("\U0001f50d  Detect Peaks (Current channel)")
        self.btn_detect.setEnabled(False)
        self.btn_detect.clicked.connect(self._detect_peaks)
        det_box.addWidget(self.btn_detect)

        self.lbl_detected = QLabel("No peaks detected yet.")
        self.lbl_detected.setWordWrap(True)
        self.lbl_detected.setStyleSheet("font-size: 10px; color: #555;")
        det_box.addWidget(self.lbl_detected)

        layout.addWidget(det_box)

        # ── Detected peaks table ─────────────────────────────────── #
        assign_box = QGroupBox("Assign Energies to Detected Peaks")
        assign_l   = QVBoxLayout(assign_box)
        prop_info  = QLabel("Toggle 🖱 Click to select peaks manually")
        prop_info.setStyleSheet("font-size: 10px; color: #556;")
        assign_l.addWidget(prop_info)

        self.tbl_detected = QTableWidget(0, 3)
        self.tbl_detected.setHorizontalHeaderLabels(["ADC", "Energy", "Label"])
        self.tbl_detected.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch)
        self.tbl_detected.setMinimumHeight(140)
        self.tbl_detected.setMaximumHeight(200)
        self.tbl_detected.itemDoubleClicked.connect(
            self._on_detected_table_dclick)
        assign_l.addWidget(self.tbl_detected)

        ab = QHBoxLayout()
        ab.setSpacing(3)
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

        # ── Detect peaks in all channels ──────────────────────────── #
        prop_box = QGroupBox("Detect Peaks in All Channels")
        prop_l   = QVBoxLayout(prop_box)
        prop_info2 = QLabel(
            "After assigning an energy on the current channel, click below "
            "to find that same peak in every channel (TSpectrum within ± window) "
            "and add it to each channel's calibration points. "
            "Repeat for every peak you want. Manual points (e.g. 0 ADC = 0 keV) "
            "are added to all channels directly.")
        prop_info2.setWordWrap(True)
        prop_info2.setStyleSheet("font-size: 10px; color: #555;")
        prop_l.addWidget(prop_info2)

        self.lbl_last_assigned = QLabel("No peak assigned yet.")
        self.lbl_last_assigned.setWordWrap(True)
        self.lbl_last_assigned.setStyleSheet(
            "font-size: 10px; color: #226; font-weight: bold; "
            "background: #eef; border-radius: 3px; padding: 2px;")
        prop_l.addWidget(self.lbl_last_assigned)

        win_h = QHBoxLayout()
        win_h.addWidget(QLabel("Search window (± ADC):"))
        self.sb_prop_window = QSpinBox()
        self.sb_prop_window.setRange(1, 10000)
        self.sb_prop_window.setValue(50)
        self.sb_prop_window.setToolTip(
            "Half-width of the ADC search window around each peak.\n"
            "TSpectrum re-detects each peak within this window\n"
            "for every channel independently.\n"
            "e.g. 50 means ± 50 ADC counts around the reference position.")
        win_h.addWidget(self.sb_prop_window)
        prop_l.addLayout(win_h)

        self.btn_propagate = QPushButton("🔍  Detect Last Peak → All Channels")
        self.btn_propagate.setEnabled(False)
        self.btn_propagate.clicked.connect(self._detect_peaks_all_channels)
        prop_l.addWidget(self.btn_propagate)

        sep = QLabel("─────  LYSO internal lines  ─────")
        sep.setStyleSheet("color: #7b1fa2; font-size: 9px;")
        sep.setAlignment(Qt.AlignCenter)
        prop_l.addWidget(sep)

        lyso_win_h = QHBoxLayout()
        lyso_win_h.addWidget(QLabel("Search window (± ADC):"))
        self.sb_lyso_window = QSpinBox()
        self.sb_lyso_window.setRange(5, 2000)
        self.sb_lyso_window.setValue(80)
        self.sb_lyso_window.setToolTip(
            "Search window around predicted ADC position of each Lu-176 line.\n"
            "The prediction uses the current rough calibration.\n"
            "Wider = more tolerant of calibration uncertainty.")
        lyso_win_h.addWidget(self.sb_lyso_window)
        prop_l.addLayout(lyso_win_h)

        self.btn_lyso_assign = QPushButton("⚡  Auto-assign LYSO lines (88 / 202 / 307 keV)")
        self.btn_lyso_assign.setEnabled(False)
        self.btn_lyso_assign.setStyleSheet(
            "QPushButton { background: #f3e5f5; color: #4a148c; "
            "border: 1px solid #ce93d8; border-radius: 3px; padding: 3px; }"
            "QPushButton:hover { background: #e1bee7; }"
            "QPushButton:disabled { color: #aaa; }")
        self.btn_lyso_assign.setToolTip(
            "For every channel tagged as LYSO:\n"
            "1. Use existing calib points to build a rough calibration\n"
            "2. Predict ADC positions of 88, 202, 307 keV Lu-176 lines\n"
            "3. Run TSpectrum in a window around each prediction\n"
            "4. Add found peaks directly to that channel's calib points\n\n"
            "Requires ≥ 1 existing calib point per LYSO channel\n"
            "(e.g. assign 511 keV first, then run this).")
        self.btn_lyso_assign.clicked.connect(self._auto_assign_lyso_lines)
        prop_l.addWidget(self.btn_lyso_assign)

        self.lbl_prop_status = QLabel("")
        self.lbl_prop_status.setWordWrap(True)
        self.lbl_prop_status.setStyleSheet("font-size: 10px; color: #555;")
        prop_l.addWidget(self.lbl_prop_status)
        layout.addWidget(prop_box)

        # ── Calibration points table ─────────────────────────────── #
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
        self.chk_override.setToolTip(
            "When checked, energy assignments made on other channels\n"
            "will NOT be propagated to this channel.\n"
            "You can still assign energies manually here.")
        cal_l.addWidget(self.chk_override)
        layout.addWidget(cal_box)

        # ── Fit buttons ──────────────────────────────────────────── #
        self.btn_fit_one = QPushButton("▶  Fit This Channel")
        self.btn_fit_one.setEnabled(False)
        self.btn_fit_one.clicked.connect(self._fit_current)
        layout.addWidget(self.btn_fit_one)

        self.btn_fit_all = QPushButton("⚡  Fit All Channels")
        self.btn_fit_all.setEnabled(False)
        self.btn_fit_all.clicked.connect(self._fit_all)
        layout.addWidget(self.btn_fit_all)

        layout.addStretch()

        self.lbl_bad = QLabel("")
        self.lbl_bad.setWordWrap(True)
        self.lbl_bad.setStyleSheet("font-size: 11px;")
        layout.addWidget(self.lbl_bad)

        # ── Wrap in scroll area so nothing is hidden on small screens ── #
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
            "Thumbnail grid of all channels.  "
            "Click any thumbnail to open that channel. "
            "Green = good fit | Red = bad channel | "
            "Grey = not yet fitted.")
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
            self, "Open ROOT File", "",
            "ROOT Files (*.root);;All Files (*)")
        if not path:
            return
        try:
            info = self.loader.open(path)
        except RuntimeError as e:
            QMessageBox.critical(self, "Backend Error", str(e))
            return
        except Exception as e:
            QMessageBox.critical(self, "File Error", str(e))
            return

        if not info["trees"] and not info["histograms"]:
            QMessageBox.warning(self, "Empty File",
                                "No TTrees or TH1 histograms found.")
            return

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
                    max_entries    = dlg.result_max_entries,
                )
            else:
                self.loader.load_from_th1()
        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e))
            return

        self.lbl_file.setText(os.path.basename(path))
        self._populate_channel_combo()
        self.btn_fit_all.setEnabled(True)
        self.btn_fit_one.setEnabled(True)
        self.btn_detect.setEnabled(True)
        n       = len(self.loader.spectra)
        backend = self.loader.backend.upper()
        self._calib_spectrum_tab.inject(self.loader, self.fit_results)

        # Populate overview grid immediately after loading (no fit results yet)
        self.overview_grid.populate(self.loader.spectra, None)

        self.statusBar().showMessage(
            f"Loaded {n} channel(s) from {os.path.basename(path)}"
            f"  [{backend} backend]")

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
        self._update_detail_view(ch_id)
        self._update_override_ui(ch_id)
        self._sync_ch_crystal_combo(ch_id)
        self._update_lyso_btn_state()

    def _prev_channel(self):
        i = self.cb_channel.currentIndex()
        if i > 0:
            self.cb_channel.setCurrentIndex(i - 1)

    def _next_channel(self):
        i = self.cb_channel.currentIndex()
        if i < self.cb_channel.count() - 1:
            self.cb_channel.setCurrentIndex(i + 1)

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
            self.spectrum_canvas.draw_detected_peaks(detected)
        assigned = self.peak_mgr.get_peaks(ch_id)
        if assigned:
            self.spectrum_canvas.draw_assigned_peaks(assigned)

        self.calib_canvas.plot_result(self.fit_results.get(ch_id))
        self._refresh_detected_table(ch_id)
        self._refresh_cal_table(ch_id)

        n_det = len(self.peak_mgr.get_detected(ch_id))
        excl  = " ⛔ excluded" if self.peak_mgr.is_excluded(ch_id) else ""
        info  = (f"Entries  : {sp.n_entries}\n"
                 f"Source   : {sp.source}\n"
                 f"Detected : {n_det} peak(s)\n"
                 f"Cal pts  : {self.peak_mgr.n_calibration_points(ch_id)}{excl}")
        r = self.fit_results.get(ch_id)
        if r:
            status = "❌ BAD" if r.bad_channel else "✅ OK"
            chi2_s = f"{r.chi2_ndf:.4f}" if r.ndf > 0 else "exact"
            info  += f"\nFit     : {status}  χ²/NDF={chi2_s}"
            if r.note:
                info += f"\nNote    : {r.note[:60]}"
        self.lbl_ch_info.setText(info)

    def _update_override_ui(self, ch_id: int):
        excluded = self.peak_mgr.is_excluded(ch_id)
        self.chk_override.blockSignals(True)
        self.chk_override.setChecked(excluded)
        self.chk_override.blockSignals(False)

    # ------------------------------------------------------------------ #
    # Peak detection
    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    # Per-channel crystal type
    # ------------------------------------------------------------------ #

    def _on_ch_crystal_changed(self, idx: int):
        """Tag the current channel with the selected crystal type."""
        ch_id = self.current_channel
        if ch_id < 0:
            return
        crystal = self.cb_ch_crystal.currentData()
        self.channel_crystal[ch_id] = crystal
        self._refresh_channel_dropdown_colors()
        # Enable LYSO button if any LYSO channel has calib points
        self._update_lyso_btn_state()
        self.statusBar().showMessage(
            f"Channel {ch_id} tagged as {crystal.upper()}")

    def _tag_all_channels(self, crystal: str):
        """Tag every loaded channel with the given crystal type."""
        for ch_id in self.loader.get_channel_ids():
            self.channel_crystal[ch_id] = crystal
        # Update current channel combobox to match
        self.cb_ch_crystal.blockSignals(True)
        for i in range(self.cb_ch_crystal.count()):
            if self.cb_ch_crystal.itemData(i) == crystal:
                self.cb_ch_crystal.setCurrentIndex(i)
                break
        self.cb_ch_crystal.blockSignals(False)
        self._refresh_channel_dropdown_colors()
        self._update_lyso_btn_state()
        self.statusBar().showMessage(
            f"All {len(self.loader.get_channel_ids())} channels tagged as {crystal.upper()}")

    def _refresh_channel_dropdown_colors(self):
        """Color LYSO channels purple, GAGG channels green in the dropdown."""
        from PyQt5.QtGui import QColor
        for i in range(self.cb_channel.count()):
            ch_id = self.cb_channel.itemData(i)
            crystal = self.channel_crystal.get(ch_id, "generic")
            if crystal == "lyso":
                self.cb_channel.setItemData(i, QColor("#7b1fa2"),
                                             Qt.ForegroundRole)
            elif crystal == "gagg":
                self.cb_channel.setItemData(i, QColor("#1565c0"),
                                             Qt.ForegroundRole)
            else:
                self.cb_channel.setItemData(i, QColor("#212121"),
                                             Qt.ForegroundRole)

    def _update_lyso_btn_state(self):
        """Enable the LYSO auto-assign button if ≥1 LYSO channel
        has at least 1 existing calibration point."""
        lyso_channels = [ch for ch, c in self.channel_crystal.items()
                         if c == "lyso"]
        enabled = any(
            len(self.peak_mgr.get_peaks(ch)) >= 1
            for ch in lyso_channels)
        self.btn_lyso_assign.setEnabled(enabled or bool(lyso_channels))

    def _sync_ch_crystal_combo(self, ch_id: int):
        """Update the crystal combo to match the stored tag for ch_id."""
        crystal = self.channel_crystal.get(ch_id, "generic")
        self.cb_ch_crystal.blockSignals(True)
        for i in range(self.cb_ch_crystal.count()):
            if self.cb_ch_crystal.itemData(i) == crystal:
                self.cb_ch_crystal.setCurrentIndex(i)
                break
        self.cb_ch_crystal.blockSignals(False)

    # ------------------------------------------------------------------ #
    # LYSO auto-assign bootstrap
    # ------------------------------------------------------------------ #

    def _auto_assign_lyso_lines(self):
        """
        For every channel tagged as LYSO:

        1.  Collect existing calib points (assigned by user so far, e.g. 511 keV).
        2.  Fit a temporary LINEAR calibration from those points
            (linear is good enough for the bootstrap prediction).
        3.  Invert it to predict ADC positions of the three Lu-176 lines
            (88.34, 201.83, 306.78 keV).
        4.  Run TSpectrum::SearchHighRes in a ± window around each prediction
            on the SNIP-subtracted spectrum.
        5.  If a peak is found, add (ADC_found, energy, "Lu-176 XkeV") to
            that channel's calibration points.
        6.  Skip any energy that already has a calib point on this channel.
        """
        from scipy.optimize import brentq
        import numpy as np

        LYSO_LINES = [
            (88.34,  "Lu-176 88keV"),
            (201.83, "Lu-176 202keV"),
            (306.78, "Lu-176 307keV"),
        ]

        lyso_channels = [ch for ch, c in self.channel_crystal.items()
                         if c == "lyso"]
        if not lyso_channels:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.information(self, "No LYSO Channels",
                "Tag at least one channel as LYSO using the Crystal type "
                "selector in the Channel panel.")
            return

        window        = self.sb_lyso_window.value()
        sigma         = self.sb_sigma.value()
        threshold     = self.sl_threshold.value() / 100.0
        max_peaks     = self.sb_max_peaks.value()
        backend       = self.loader.backend
        tspec_mode    = self.cb_tspec_mode.currentData()
        iterations    = self.sb_iterations.value()
        bg_subtract   = self.chk_bg_subtract.isChecked()
        bg_iters      = self.sb_bg_iterations.value()

        n_total   = len(lyso_channels)
        n_ok      = 0
        n_skip    = 0
        n_failed  = 0
        lines_added = 0

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, n_total)

        for prog_i, ch_id in enumerate(lyso_channels):
            self.progress_bar.setValue(prog_i + 1)
            QApplication.processEvents()

            sp = self.loader.get_spectrum(ch_id)
            if sp is None:
                n_failed += 1
                continue

            existing = self.peak_mgr.get_peaks(ch_id)
            if not existing:
                # No calib points at all — can't bootstrap
                n_skip += 1
                continue

            # ── Step 1: Build rough linear calibration from existing points ──
            adcs = np.array([p.adc_position  for p in existing])
            engs = np.array([p.known_energy   for p in existing])

            if len(adcs) == 1:
                # Only one point: assume origin passes through (0,0)
                adcs = np.array([0.0, adcs[0]])
                engs = np.array([0.0, engs[0]])

            # Least-squares linear fit: E = slope * ADC + intercept
            try:
                coeffs = np.polyfit(adcs, engs, 1)   # [slope, intercept]
                slope, intercept = coeffs
                if abs(slope) < 1e-9:
                    n_skip += 1
                    continue
                # Inverse: ADC = (E - intercept) / slope
                def energy_to_adc(e):
                    return (e - intercept) / slope
            except Exception:
                n_skip += 1
                continue

            # ── Step 2: For each Lu-176 line, predict ADC and search ──────
            ch_added = 0
            existing_energies = {p.known_energy for p in existing}

            for target_energy, label in LYSO_LINES:
                # Skip if already assigned
                if any(abs(e - target_energy) < 1.0 for e in existing_energies):
                    continue

                pred_adc = energy_to_adc(target_energy)
                lo = pred_adc - window
                hi = pred_adc + window

                # Slice spectrum
                mask = ((sp.bin_centers >= max(lo, 0)) &
                        (sp.bin_centers <= hi))
                if mask.sum() < 5:
                    continue

                win_centers = sp.bin_centers[mask]
                win_counts  = sp.counts[mask]

                candidates, _ = PeakManager.detect_peaks(
                    win_centers, win_counts,
                    sigma=sigma, threshold=threshold,
                    max_peaks=max_peaks, backend=backend,
                    pedestal_cut=0.0,
                    tspec_mode=tspec_mode, iterations=iterations,
                    bg_subtract=bg_subtract, bg_iterations=bg_iters)

                if candidates:
                    # Take closest to prediction
                    best = min(candidates, key=lambda x: abs(x - pred_adc))
                else:
                    # Fallback: centroid of smoothed window
                    from scipy.ndimage import gaussian_filter1d
                    sm  = gaussian_filter1d(win_counts.astype(float),
                                            sigma=max(1, int(sigma)))
                    idx = int(np.argmax(sm))
                    hw  = max(2, len(sm) // 8)
                    sl  = slice(max(0, idx-hw), min(len(sm), idx+hw+1))
                    w   = sm[sl]; c = win_centers[sl]
                    best = (float(np.sum(c*w)/w.sum())
                            if w.sum() > 0 else pred_adc)

                # Remove duplicate at same energy first
                if ch_id in self.peak_mgr.channel_peaks:
                    self.peak_mgr.channel_peaks[ch_id] = [
                        p for p in self.peak_mgr.channel_peaks[ch_id]
                        if abs(p.known_energy - target_energy) > 0.5]

                self.peak_mgr.add_channel_peak(ch_id, best,
                                                target_energy, label)
                ch_added += 1
                lines_added += 1

            if ch_added > 0:
                n_ok += 1
            else:
                n_skip += 1

        self.progress_bar.setVisible(False)
        self._refresh_cal_table(self.current_channel)
        self._update_detail_view(self.current_channel)

        msg = (f"LYSO auto-assign: added {lines_added} line(s) across "
               f"{n_ok}/{n_total} LYSO channel(s).")
        if n_skip:
            msg += f"  ({n_skip} skipped — no calib points yet)"
        if n_failed:
            msg += f"  ({n_failed} no spectrum)"
        self.lbl_prop_status.setText(msg)
        self.statusBar().showMessage(msg)

    def _on_threshold_changed(self, v: int):
        """Update label and redraw threshold line live as slider moves."""
        pct = v               # integer percent
        self.lbl_threshold.setText(f"{pct}%  of max")
        # Live-update threshold line on spectrum without re-detecting
        threshold = v / 100.0
        self.spectrum_canvas.update_threshold_line(threshold)

    def _detect_peaks(self):
        """Phase 1 — detect peaks on current channel. No energy assigned yet."""
        ch_id = self.current_channel
        sp    = self.loader.get_spectrum(ch_id)
        if sp is None:
            return

        threshold    = self.sl_threshold.value() / 100.0
        sigma        = self.sb_sigma.value()
        max_peaks    = self.sb_max_peaks.value()
        pedestal_cut = float(self.sb_pedestal.value())
        backend      = self.loader.backend
        tspec_mode    = self.cb_tspec_mode.currentData()
        iterations    = self.sb_iterations.value()
        bg_subtract   = self.chk_bg_subtract.isChecked()
        bg_iterations = self.sb_bg_iterations.value()
        crystal       = self.cb_crystal.currentData()
        show_known    = self.chk_show_known.isChecked()

        
        refine = self.chk_refine_peaks.isChecked()  # or True for now

        positions, bg_array, refine_info = PeakManager.detect_peaks(
            sp.bin_centers, sp.counts,
            sigma=sigma, threshold=threshold,
            max_peaks=max_peaks, backend=backend,
            pedestal_cut=pedestal_cut,
            tspec_mode=tspec_mode, iterations=iterations,
            bg_subtract=bg_subtract, bg_iterations=bg_iterations,
            refine=refine,
            refine_window_adc=30.0,
            max_chi2_ndf=10.0)
        

        # Store raw positions in manager (no energy yet)
        self.peak_mgr.set_detected(ch_id, positions)

        # Always redraw spectrum first
        self.spectrum_canvas.plot_spectrum(
            sp.bin_centers, sp.counts,
            title=f"Channel {ch_id} — {sp.n_entries} entries")

        # Overlay SNIP background if computed
        if bg_subtract and bg_array is not None:
            # Align bg_array to bin_centers after pedestal cut
            bc = sp.bin_centers
            if pedestal_cut > 0:
                mask = bc >= pedestal_cut
                bc   = bc[mask]
            if len(bc) == len(bg_array):
                self.spectrum_canvas.draw_background(bc, bg_array)

        self.spectrum_canvas.update_threshold_line(threshold)

        if not positions:
            self.lbl_detected.setText(
                "No peaks found — try lowering threshold or sigma.\n"
                + ("SNIP background subtracted — check orange overlay." if bg_subtract else ""))
        else:
            mode_lbl = ("TSpectrum HighRes" if tspec_mode == "highres"
                        else ("TSpectrum" if backend == "pyroot" else "scipy"))
            bg_lbl = "  [BG subtracted]" if bg_subtract else ""
            
            msg = f"{len(positions)} peak(s) detected  [{mode_lbl}]{bg_lbl}"

            if refine:
                msg += f"\nRefined with sliding Gaussian fit"
            if bad:
                msg += f"\n⚠ {len(bad)} peak(s) rejected (bad fit)"

            msg += "\n" + ", ".join(f"{p:.1f}" for p in positions)
            self.lbl_detected.setText(msg)
            
            self.btn_propagate.setEnabled(True)
            self._refresh_detected_table(ch_id)
            self.spectrum_canvas.draw_detected_peaks(positions, color="green")
            bad = []
            # ── NEW: highlight rejected peaks (if refinement enabled) ──────────
            if refine and refine_info:
                bad = [r for r in refine_info if not r.success]
            if bad:
                bad_adc = [r.adc for r in bad]
                self.spectrum_canvas.draw_detected_peaks(
                    bad_adc, color="red", style="x")

        # Overlay known crystal lines in ADC space (if calibration available)
        if show_known and crystal in CRYSTAL_KNOWN_LINES:
            known = CRYSTAL_KNOWN_LINES[crystal]
            fit   = self.fit_results.get(ch_id)
            if known and fit:
                self.spectrum_canvas.draw_known_lines_adc(known, fit)

        assigned = self.peak_mgr.get_peaks(ch_id)
        if assigned:
            self.spectrum_canvas.draw_assigned_peaks(assigned)

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
        """
        Assign a known energy to a detected peak position.

        - If adc_pos == 0 and energy == 0: zero-point, added directly to ALL
          channels' calibration points (no TSpectrum search needed).
        - Otherwise: saved on the current channel and stored as _last_assigned
          so the user can then click "Detect in All → <energy>" to find the
          same peak in every channel and add it to their calib points.
        """
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
            # Add (0 ADC, 0 keV) to every channel immediately — no search needed
            for c in all_channels:
                if not self.peak_mgr.is_excluded(c):
                    self.peak_mgr.add_channel_peak(c, 0.0, 0.0,
                                                    lbl or "origin")
            self._refresh_cal_table(ch_id)
            self._update_detail_view(ch_id)
            msg = (f"Zero-point (0 ADC, 0 keV) added to "
                   f"{len(all_channels)} channel(s).")
            self.lbl_prop_status.setText(msg)
            self.lbl_last_assigned.setText("✔ Zero-point added to all channels.")
            self.statusBar().showMessage(msg)
            return

        # Save on the current channel
        self.peak_mgr.add_channel_peak(ch_id, adc, eng, lbl)

        # Remember this assignment so "Detect in All" knows the target
        self._last_assigned = {"adc": adc, "energy": eng, "label": lbl}
        self.btn_propagate.setEnabled(True)
        self.lbl_last_assigned.setText(
            f"Last assigned: {eng} keV  ({lbl})  at ADC {adc:.1f}  "
            f"→ click  🔍 Detect → All  to find in every channel")

        self._refresh_detected_table(ch_id)
        self._refresh_cal_table(ch_id)
        self._update_detail_view(ch_id)
        self._update_lyso_btn_state()

        msg = f"Assigned {eng} keV ({lbl}) at ADC {adc:.1f} on ch {ch_id}. Click Detect → All to propagate."
        self.lbl_prop_status.setText(msg)
        self.statusBar().showMessage(msg)

    def _add_manual_point_all(self):
        """
        Open the assign dialog with ADC=0 pre-filled.
        If the user enters (0, 0) it goes to all channels.
        If the user enters any other (ADC, energy), it also goes to all
        channels directly — this is useful for known fixed points like
        (0 ADC, 0 keV) that are common to every channel.
        """
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
                # Remove any existing point at the same energy (avoid duplicates)
                if c in self.peak_mgr.channel_peaks:
                    self.peak_mgr.channel_peaks[c] = [
                        p for p in self.peak_mgr.channel_peaks[c]
                        if abs(p.known_energy - eng) > 0.01
                    ]
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
        """Remove this channel's specific peak assignments (keeps detected positions)."""
        ch_id = self.current_channel
        if ch_id < 0:
            return
        self.peak_mgr.reset_channel(ch_id)
        self._refresh_cal_table(ch_id)
        self._update_detail_view(ch_id)

    def _on_exclude_toggled(self, checked: bool):
        """Mark/unmark this channel as excluded from energy propagation."""
        ch_id = self.current_channel
        if ch_id < 0:
            return
        self.peak_mgr.set_excluded(ch_id, checked)
        status = "excluded from" if checked else "included in"
        self.statusBar().showMessage(
            f"Channel {ch_id} is now {status} energy propagation.")

    def _refresh_detected_table(self, ch_id: int):
        """
        Refresh the detected peaks table for ch_id.
        Shows all raw detected positions; if an energy has been assigned
        to that position (via channel_peaks), show it inline.
        """
        detected = self.peak_mgr.get_detected(ch_id)
        assigned = self.peak_mgr.get_peaks(ch_id)   # energy-labelled peaks

        # Build a quick lookup: adc -> Peak for assigned peaks
        tol = 5.0  # ADC tolerance to match detected → assigned
        def find_assigned(adc):
            best = None
            best_d = tol
            for p in assigned:
                d = abs(p.adc_position - adc)
                if d < best_d:
                    best_d = d
                    best = p
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
                # Colour the row green to show it's assigned
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
    # Detect peaks in ALL channels for the last assigned energy
    # ------------------------------------------------------------------ #

    def _detect_peaks_all_channels(self):
        """
        For the peak most recently assigned on the current channel
        (stored in self._last_assigned), run TSpectrum inside a
        ± window around that ADC position on every other channel,
        find the best local maximum, and add (ADC_ch, energy, label)
        directly to each channel's calibration points.

        This accumulates — calling it again for a different peak (1275 keV)
        appends to existing calibration points without removing the 511 keV
        entry already there.

        Channels flagged as excluded are skipped.
        """
        if not self._last_assigned:
            QMessageBox.information(self, "No Peak Assigned",
                "Detect a peak on the current channel and assign its energy "
                "first, then click this button.")
            return

        ref_adc  = self._last_assigned["adc"]
        energy   = self._last_assigned["energy"]
        label    = self._last_assigned["label"]
        window   = self.sb_prop_window.value()

        all_channels = self.loader.get_channel_ids()
        src_ch       = self.current_channel

        threshold    = self.sl_threshold.value() / 100.0
        sigma        = self.sb_sigma.value()
        max_peaks    = self.sb_max_peaks.value()
        pedestal_cut = float(self.sb_pedestal.value())
        backend      = self.loader.backend
        tspec_mode   = self.cb_tspec_mode.currentData()
        iterations   = self.sb_iterations.value()

        n_ok     = 0
        n_excl   = 0
        n_failed = 0
        n_total  = len([c for c in all_channels if c != src_ch])

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, max(n_total, 1))

        for i, ch_id in enumerate(all_channels):
            if ch_id == src_ch:
                continue
            self.progress_bar.setValue(i + 1)
            QApplication.processEvents()

            if self.peak_mgr.is_excluded(ch_id):
                n_excl += 1
                continue

            sp = self.loader.get_spectrum(ch_id)
            if sp is None:
                n_failed += 1
                continue

            # Slice spectrum to search window
            lo   = ref_adc - window
            hi   = ref_adc + window
            mask = (sp.bin_centers >= lo) & (sp.bin_centers <= hi)

            if mask.sum() < 3:
                # Window outside this channel's ADC range — skip
                n_failed += 1
                continue

            win_centers = sp.bin_centers[mask]
            win_counts  = sp.counts[mask]

            # Run TSpectrum / scipy inside the window
            result_tuple = PeakManager.detect_peaks(
                win_centers, win_counts,
                sigma        = sigma,
                threshold    = threshold,
                max_peaks    = max_peaks,
                backend      = backend,
                pedestal_cut = 0.0,
                tspec_mode   = tspec_mode,
                iterations   = iterations,
                bg_subtract  = self.chk_bg_subtract.isChecked(),
                bg_iterations= self.sb_bg_iterations.value())
            candidates = result_tuple[0]  # positions only

            if candidates:
                # Take the candidate closest to the reference ADC
                best_adc = min(candidates, key=lambda x: abs(x - ref_adc))
            else:
                # Fallback: centroid of the smoothed local maximum
                from scipy.ndimage import gaussian_filter1d
                smooth   = gaussian_filter1d(win_counts.astype(float),
                                             sigma=max(1, sigma))
                idx_max  = int(np.argmax(smooth))
                hw       = max(2, int(len(smooth) * 0.1))
                lo_i     = max(0, idx_max - hw)
                hi_i     = min(len(smooth), idx_max + hw + 1)
                wslice   = smooth[lo_i:hi_i]
                cslice   = win_centers[lo_i:hi_i]
                best_adc = (float(np.sum(cslice * wslice) / wslice.sum())
                            if wslice.sum() > 0 else ref_adc)

            # Add directly to this channel's calibration points (cumulative)
            # Remove duplicate at same energy first to allow re-running
            if ch_id in self.peak_mgr.channel_peaks:
                tol = window * 0.5
                self.peak_mgr.channel_peaks[ch_id] = [
                    p for p in self.peak_mgr.channel_peaks[ch_id]
                    if abs(p.known_energy - energy) > 0.01
                ]
            self.peak_mgr.add_channel_peak(ch_id, best_adc, energy, label)
            n_ok += 1

        self.progress_bar.setVisible(False)
        self._update_detail_view(src_ch)

        msg = (f"Added {energy} keV ({label}) to calib points of "
               f"{n_ok}/{n_total} channel(s).")
        if n_excl:
            msg += f"  ({n_excl} excluded)"
        if n_failed:
            msg += f"  ({n_failed} had no spectrum in window)"
        self.lbl_prop_status.setText(msg)
        self.lbl_last_assigned.setText(
            f"✔ {energy} keV ({label}) added to {n_ok} channel(s). "
            f"Now detect the next peak and assign its energy.")
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
            QMessageBox.warning(self, "No Peaks",
                                "Assign calibration peaks first.")
            return
        model       = self.cb_model.currentData()
        custom_expr = self.le_custom_expr.text().strip()
        result = self.fitter.fit_channel(ch_id, adc, eng, model, custom_expr)
        self.fit_results[ch_id] = result
        self._update_detail_view(ch_id)
        self._update_bad_label()

        # ── FIX: update overview grid after single-channel fit ── #
        self.overview_grid.populate(self.loader.spectra, self.fit_results)

    def _fit_all(self):
        ch_ids = self.loader.get_channel_ids()
        if not ch_ids:
            return
        model       = self.cb_model.currentData()
        custom_expr = self.le_custom_expr.text().strip()

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, len(ch_ids))
        self.btn_fit_all.setEnabled(False)

        self._worker = FitWorker(
            self.fitter, ch_ids, self.peak_mgr, model, custom_expr)
        self._worker.progress.connect(
            lambda d, t: self.progress_bar.setValue(d))
        self._worker.finished.connect(self._on_fit_all_done)
        self._worker.error.connect(
            lambda e: QMessageBox.critical(self, "Fit Error", e))
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
        # ── Populate overview with fit results ── #
        self.overview_grid.populate(self.loader.spectra, self.fit_results)
        bad = self.fitter.bad_channels()
        self.statusBar().showMessage(
            f"Fit complete — {len(results)} channels | "
            f"{len(bad)} bad channel(s)")

    # ------------------------------------------------------------------ #
    # Export
    # ------------------------------------------------------------------ #

    def _export(self):
        if not self.fit_results:
            return
        out_dir = QFileDialog.getExistingDirectory(
            self, "Select Output Directory")
        if not out_dir:
            return
        try:
            paths = OutputWriter.write_all(
                self.fit_results, out_dir,
                source_file=self.loader.filename)
            QMessageBox.information(
                self, "Exported",
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
    # Calibrated Spectrum tab
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
            QPushButton:hover  { background-color: #e3f2fd;
                                  border-color: #1565c0; }
            QPushButton:pressed { background-color: #bbdefb; }
            QPushButton:checked { background-color: #1565c0; color: #fff; }
            QPushButton:disabled { color: #9e9e9e;
                                    background-color: #eeeeee; }
            QGroupBox {
                border: 1px solid #e0e0e0; border-radius: 6px;
                margin-top: 8px; padding-top: 4px;
                font-weight: bold; color: #1565c0;
                background-color: #ffffff;
            }
            QGroupBox QPushButton { padding: 4px 4px; min-width: 0px; }
            QGroupBox::title { subcontrol-origin: margin; left: 8px; }
            QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {
                background-color: #ffffff; border: 1px solid #bdbdbd;
                border-radius: 4px; padding: 3px 6px; color: #212121;
            }
            QComboBox:focus, QSpinBox:focus,
            QDoubleSpinBox:focus, QLineEdit:focus {
                border-color: #1565c0;
            }
            QTableWidget {
                background-color: #ffffff; gridline-color: #e0e0e0;
                border: 1px solid #e0e0e0; border-radius: 4px;
            }
            QTableWidget::item:selected { background-color: #bbdefb;
                                           color: #212121; }
            QHeaderView::section {
                background-color: #f5f5f5; color: #1565c0;
                border: none; border-bottom: 1px solid #e0e0e0;
                padding: 4px; font-weight: bold;
            }
            QTabWidget::pane  { border: 1px solid #e0e0e0;
                                 background: #ffffff; }
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
                border: 1px solid #bdbdbd; border-radius: 3px;
                background: #ffffff;
            }
            QCheckBox::indicator:checked { background: #1565c0;
                                            border-color: #1565c0; }
            QSlider::groove:horizontal {
                height: 4px; background: #e0e0e0; border-radius: 2px; }
            QSlider::handle:horizontal {
                width: 14px; height: 14px; margin: -5px 0;
                background: #1565c0; border-radius: 7px; }
            QSplitter::handle { background: #e0e0e0; }
        """)
        plt.style.use("default")
