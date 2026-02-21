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
from src.peak_manager import PeakManager, Peak
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
        self._peak_markers = []
        self._click_mode   = False
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

    def draw_detected_peaks(self, positions: list[float]):
        """Draw auto-detected peak positions (before energy assignment)."""
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

    def draw_assigned_peaks(self, peaks: list[Peak]):
        """Draw peaks that have known energy assigned (green)."""
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

        x_fit = np.linspace(max(0, adc.min() * 0.8), adc.max() * 1.15, 500)
        y_fit = result.energy_at(x_fit)

        self.ax.plot(x_fit, y_fit, "-", color="#1565c0",
                      linewidth=1.8, label="Calibration fit")
        self.ax.scatter(adc, eng, color="#c62828", zorder=5,
                         s=60, label="Assigned peaks")

        # Annotate each point with its energy label
        for a, e, p in zip(adc, eng, result.adc_points):
            self.ax.annotate(f"{e:.1f} keV",
                              xy=(a, e), xytext=(4, 4),
                              textcoords="offset points",
                              fontsize=7, color="#c62828")

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

    def populate(self, spectra: dict, results: dict | None = None):
        for i in reversed(range(self._layout.count())):
            w = self._layout.itemAt(i).widget()
            if w:
                w.setParent(None)

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

        # Result fields
        self.result_mode       = ""
        self.result_tree       = ""
        self.result_ch_branch  = ""
        self.result_adc_branch = ""
        self.result_nbins      = 1024
        self.result_draw_mode  = "filter"
        self.result_custom_expr = ""
        self.result_ch_first   = 0
        self.result_ch_last    = -1
        self.result_ch_step    = 1
        self.result_max_entries = 0
        self.result_structure  = "filter"

        layout = QVBoxLayout(self)

        # Backend label
        backend = file_info.get("backend", "unknown").upper()
        blabel  = QLabel(f"Active backend: <b>{backend}</b>")
        blabel.setStyleSheet("color: #1565c0; padding: 4px;")
        layout.addWidget(blabel)

        has_trees = bool(file_info["trees"])
        has_hists = bool(file_info["histograms"])

        # ── Mode (TTree or TH1) ──────────────────────────────────────── #
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

        # ── TTree options ────────────────────────────────────────────── #
        self.tree_group = QGroupBox("TTree Options")
        tg = QFormLayout(self.tree_group)

        self.cb_tree_name = QComboBox()
        for t in file_info["trees"]:
            self.cb_tree_name.addItem(
                f"{t['name']}  ( - Totoal {t['entries']} entries)", t["name"])

        # Draw mode
        self.cb_draw_mode = QComboBox()
        self.cb_draw_mode.addItem(
            'Filter     Draw("adc", "channelID==N")', "filter")
        self.cb_draw_mode.addItem(
            'Array      Draw("adc[channelID]")', "array")
        self.cb_draw_mode.addItem(
            'Custom     user expression (ROOT mode)  (%d = channel number)', "custom")

        self.cb_channel_branch = QComboBox()
        self.cb_adc_branch     = QComboBox()

        # Custom expression line
        self.le_draw_custom = QLineEdit()
        self.le_draw_custom.setPlaceholderText(
            'e.g.  energy[%d]   or   sqrt(adc),channelID==%d')
        self.le_draw_custom.setEnabled(False)

        # Preview label — shows the exact Draw() call
        self.lbl_draw_preview = QLabel("")
        self.lbl_draw_preview.setStyleSheet(
            "color:#555; font-size:11px; font-style:italic;")
        self.lbl_draw_preview.setWordWrap(True)

        self.sb_nbins = QSpinBox()
        self.sb_nbins.setRange(64, 65536)
        self.sb_nbins.setValue(1024)

        tg.addRow("TTree:",           self.cb_tree_name)
        tg.addRow("Draw mode:",       self.cb_draw_mode)
        tg.addRow("Channel branch:",  self.cb_channel_branch)
        tg.addRow("Energy/ADC branch:", self.cb_adc_branch)
        tg.addRow("Custom expression:", self.le_draw_custom)
        tg.addRow("Draw preview:",    self.lbl_draw_preview)
        tg.addRow("Histogram bins:",  self.sb_nbins)

        layout.addWidget(self.tree_group)

        # ── Channel range ────────────────────────────────────────────── #
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

        # Custom channel list
        self.le_ch_list = QLineEdit()
        self.le_ch_list.setPlaceholderText(
            "Optional: comma-separated list, e.g.  0,1,2,5,10  "
            "(overrides First/Last/Step)")
        cr.addRow("Custom list:", self.le_ch_list)

        # Max entries
        self.sb_max_entries = QSpinBox()
        self.sb_max_entries.setRange(0, 100_000_000)
        self.sb_max_entries.setValue(0)
        self.sb_max_entries.setSpecialValueText("Choose entries")
        self.sb_max_entries.setSingleStep(10000)
        cr.addRow("Max entries per channel:", self.sb_max_entries)

        layout.addWidget(self.ch_range_box)

        # ── TH1 options ──────────────────────────────────────────────── #
        self.hist_group = QGroupBox("Histogram Options")
        QVBoxLayout(self.hist_group).addWidget(
            QLabel(f"{len(file_info['histograms'])} histogram(s) found — "
                   "all will be loaded."))
        layout.addWidget(self.hist_group)

        # ── Connect signals ──────────────────────────────────────────── #
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

        # For array/custom mode, channel range is required
        hint = ""
        if is_arr:
            hint = "Array mode: set channel First/Last range below."
        elif is_cust:
            hint = "Custom mode: use %d as placeholder for channel number."
        elif is_filt:
            hint = "Filter mode: channel IDs auto-discovered from data."
        self.lbl_draw_preview.setText(hint)
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
        self.ch_range_box.setVisible(use_tree)
        self.hist_group.setVisible(not use_tree)

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

            # Parse custom channel list
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

    # Common gamma-ray energies from frequently used sources
    # User can always type any value freely
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

        # Free energy entry — user can type anything
        self.sb_energy = QDoubleSpinBox()
        self.sb_energy.setRange(0, 1e9)
        self.sb_energy.setDecimals(3)
        self.sb_energy.setValue(0.0)
        self.sb_energy.setToolTip(
            "Enter any energy value in keV.\n"
            "Use the quick-select table below for common sources.")

        self.le_label = QLineEdit()
        self.le_label.setPlaceholderText("optional label, e.g.  Cs-137  661.7 keV")

        form.addRow("ADC position (Q):", self.sb_adc)
        form.addRow("Known energy (keV):", self.sb_energy)
        form.addRow("Label:", self.le_label)
        layout.addLayout(form)

        # Collapsible quick-select table
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
        tbl.cellDoubleClicked.connect(self._quick_select)
        tbl.cellClicked.connect(self._quick_select)
        layout.addWidget(tbl)
        layout.addWidget(QLabel(
            "<i>Click a row to select that energy, or type any value above.</i>"))

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
        self._detected_positions: list[float] = []   # TSpectrum results
        self._click_mode = False

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
        left.setMinimumWidth(240)
        left.setMaximumWidth(300)
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

        self.statusBar().showMessage(
            "Ready — open a ROOT file to begin.")
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

        # Model selector in toolbar
        layout.addWidget(QLabel("Model:"))
        self.cb_model = QComboBox()
        self.cb_model.addItem("Linear  E = P0 + P1·Q",                      "linear")
        self.cb_model.addItem("Nonlinear  E = P0·(P1^Q)^P2 + P3·Q − P0",   "nonlinear")
        self.cb_model.addItem("Custom", "custom")
        self.cb_model.setMinimumWidth(300)
        self.cb_model.currentIndexChanged.connect(self._on_model_changed)
        layout.addWidget(self.cb_model)
        self.le_custom_expr = QLineEdit()
        self.le_custom_expr.setPlaceholderText(
            "Custom expression, e.g.:  a*x**2 + b*x + c   (x = ADC value)")
        self.le_custom_expr.setEnabled(False)
        self.le_custom_expr.setMinimumWidth(320)
        layout.addWidget(self.le_custom_expr)


        self.btn_fit_all = QPushButton("⚡ Fit All Channels")
        self.btn_fit_all.setEnabled(False)
        self.btn_fit_all.clicked.connect(self._fit_all)
        layout.addWidget(self.btn_fit_all)

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

        # ── Channel navigation ─────────────────────────────────────── #
        nav_box = QGroupBox("Channel")
        nav_l   = QVBoxLayout(nav_box)
        nav_h   = QHBoxLayout()
        self.btn_prev = QPushButton("◀")
        self.btn_prev.setFixedWidth(36)
        self.btn_prev.clicked.connect(self._prev_channel)
        self.cb_channel = QComboBox()
        self.cb_channel.currentIndexChanged.connect(self._on_channel_changed)
        self.btn_next = QPushButton("▶")
        self.btn_next.setFixedWidth(36)
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

        # ── Peak detection (TSpectrum) ─────────────────────────────── #
        det_box = QGroupBox("Peak Detection  (TSpectrum / scipy)")
        det_l   = QVBoxLayout(det_box)

        thresh_h = QHBoxLayout()
        thresh_h.addWidget(QLabel("Threshold:"))
        self.sl_threshold = QSlider(Qt.Horizontal)
        self.sl_threshold.setRange(1, 50)
        self.sl_threshold.setValue(5)
        self.sl_threshold.setTickInterval(5)
        self.sl_threshold.setTickPosition(QSlider.TicksBelow)
        self.lbl_threshold = QLabel("0.05")
        self.sl_threshold.valueChanged.connect(
            lambda v: self.lbl_threshold.setText(f"{v/100:.2f}"))
        thresh_h.addWidget(self.sl_threshold, stretch=1)
        thresh_h.addWidget(self.lbl_threshold)
        det_l.addLayout(thresh_h)

        sigma_h = QHBoxLayout()
        sigma_h.addWidget(QLabel("Sigma (bins):"))
        self.sb_sigma = QDoubleSpinBox()
        self.sb_sigma.setRange(0.5, 20.0)
        self.sb_sigma.setSingleStep(0.5)
        self.sb_sigma.setValue(2.0)
        sigma_h.addWidget(self.sb_sigma)
        det_l.addLayout(sigma_h)

        maxpk_h = QHBoxLayout()
        maxpk_h.addWidget(QLabel("Max peaks:"))
        self.sb_max_peaks = QSpinBox()
        self.sb_max_peaks.setRange(1, 30)
        self.sb_max_peaks.setValue(10)
        maxpk_h.addWidget(self.sb_max_peaks)
        det_l.addLayout(maxpk_h)

        self.btn_detect = QPushButton("🔍  Detect Peaks (Current channel)")
        self.btn_detect.setEnabled(False)
        self.btn_detect.clicked.connect(self._detect_peaks)
        det_l.addWidget(self.btn_detect)

        self.lbl_detected = QLabel("No peaks detected yet.")
        self.lbl_detected.setWordWrap(True)
        self.lbl_detected.setStyleSheet("font-size: 10px; color: #555;")
        det_l.addWidget(self.lbl_detected)

        layout.addWidget(det_box)

        # ── Propagate peaks to other channels ─────────────────────── #
        prop_box = QGroupBox("Propagate Peaks to All Channels")
        prop_l   = QVBoxLayout(prop_box)

        prop_info = QLabel(
            "search for the same peaks in every other channel "
            "within a ± ADC window.")
        prop_info.setWordWrap(True)
        prop_info.setStyleSheet("font-size: 10px; color: #555;")
        prop_l.addWidget(prop_info)

        win_h = QHBoxLayout()
        win_h.addWidget(QLabel("Search window (± ADC):"))
        self.sb_prop_window = QSpinBox()
        self.sb_prop_window.setRange(1, 10000)
        self.sb_prop_window.setValue(10)
        self.sb_prop_window.setToolTip(
            "Half-width of ADC search window around each peak position.\n"
            "e.g. 50 means ± 50 ADC counts around the reference position.")
        win_h.addWidget(self.sb_prop_window)
        prop_l.addLayout(win_h)

        src_h = QHBoxLayout()
        src_h.addWidget(QLabel("Use peaks from:"))
        self.cb_prop_source = QComboBox()
        self.cb_prop_source.addItem("Current channel (detected + assigned)", "current")
        self.cb_prop_source.addItem("Global peak list only", "global")
        src_h.addWidget(self.cb_prop_source, stretch=1)
        prop_l.addLayout(src_h)

        self.chk_prop_override = QCheckBox(
            "Create per-channel overrides")
        self.chk_prop_override.setChecked(True)
        self.chk_prop_override.setToolTip(
            "Each channel gets its own peak list with positions refined "
            "to the local maximum inside the search window.")
        prop_l.addWidget(self.chk_prop_override)

        self.btn_propagate = QPushButton(
            "Propagate Peaks → All Channels")
        self.btn_propagate.setEnabled(False)
        self.btn_propagate.clicked.connect(self._propagate_peaks)
        prop_l.addWidget(self.btn_propagate)

        self.lbl_prop_status = QLabel("")
        self.lbl_prop_status.setWordWrap(True)
        self.lbl_prop_status.setStyleSheet("font-size: 10px; color: #555;")
        prop_l.addWidget(self.lbl_prop_status)

        layout.addWidget(prop_box)

        # ── Detected peaks table (assign energies) ─────────────────── #
        assign_box = QGroupBox("Assign Energies to Detected Peaks")
        assign_l   = QVBoxLayout(assign_box)

        self.tbl_detected = QTableWidget(0, 3)
        self.tbl_detected.setHorizontalHeaderLabels(
            ["ADC", "Energy", "Label"])
        self.tbl_detected.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch)
        self.tbl_detected.setMinimumHeight(200)
        self.tbl_detected.setMaximumHeight(300)
        self.tbl_detected.itemDoubleClicked.connect(
            self._on_detected_table_dclick)
        assign_l.addWidget(self.tbl_detected)

        ab = QHBoxLayout()
        self.btn_assign   = QPushButton("✏ Assign")
        self.btn_assign.clicked.connect(self._assign_selected_peak)
        self.btn_add_manual = QPushButton("+ Manual")
        self.btn_add_manual.clicked.connect(
            lambda: self._open_assign_dialog(0.0))
        self.btn_click_pk = QPushButton("🖱 Click")
        self.btn_click_pk.setCheckable(True)
        self.btn_click_pk.toggled.connect(self._toggle_click_mode)
        ab.addWidget(self.btn_assign)
        ab.addWidget(self.btn_add_manual)
        ab.addWidget(self.btn_click_pk)
        assign_l.addLayout(ab)

        layout.addWidget(assign_box)

        # ── Assigned calibration points ────────────────────────────── #
        cal_box = QGroupBox("Calibration Points (Current channel)")
        cal_l   = QVBoxLayout(cal_box)
        self.tbl_cal = QTableWidget(0, 3)
        self.tbl_cal.setHorizontalHeaderLabels(
            ["ADC", "Energy", "Label"])
        self.tbl_cal.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch)
        self.tbl_cal.setMinimumHeight(180)
        self.tbl_cal.setMaximumHeight(280)
        cal_l.addWidget(self.tbl_cal)

        cb2 = QHBoxLayout()
        self.btn_del_cal    = QPushButton("− Remove")
        self.btn_del_cal.clicked.connect(self._delete_cal_point)
        self.btn_use_global = QPushButton("↑ Use Global")
        self.btn_use_global.setToolTip(
            "Apply global peaks to all channels as default")
        self.btn_use_global.clicked.connect(self._reset_to_global)
        cb2.addWidget(self.btn_del_cal)
        cb2.addWidget(self.btn_use_global)
        cal_l.addLayout(cb2)

        self.chk_override = QCheckBox("Override: use channel-specific peaks")
        self.chk_override.toggled.connect(self._on_override_toggled)
        cal_l.addWidget(self.chk_override)

        layout.addWidget(cal_box)

        # ── Fit single channel ─────────────────────────────────────── #
        self.btn_fit_one = QPushButton("▶  Fit This Channel")
        self.btn_fit_one.setEnabled(False)
        self.btn_fit_one.clicked.connect(self._fit_current)
        layout.addWidget(self.btn_fit_one)

        layout.addStretch()

        self.lbl_bad = QLabel("")
        self.lbl_bad.setWordWrap(True)
        self.lbl_bad.setStyleSheet("font-size: 11px;")
        layout.addWidget(self.lbl_bad)

        return panel

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
        info = QLabel("Click any thumbnail to open that channel. "
                       "Green = good fit | Red = bad channel.")
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
        self.cb_trend_param.addItems(["P0", "P1", "P2", "P3"])
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
        # Keep calib spectrum tab in sync with new loader
        self._calib_spectrum_tab.inject(self.loader, self.fit_results)
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
            title=f"Channel {ch_id} — {sp.n_entries} entries  "
                  f"({sp.source})")

        # Draw detected (orange) then assigned (green)
        if self._detected_positions:
            self.spectrum_canvas.draw_detected_peaks(
                self._detected_positions)
        assigned = self.peak_mgr.get_peaks(ch_id)
        if assigned:
            self.spectrum_canvas.draw_assigned_peaks(assigned)

        self.calib_canvas.plot_result(self.fit_results.get(ch_id))
        self._refresh_cal_table(ch_id)

        info = (f"Entries : {sp.n_entries}\n"
                f"Source  : {sp.source}\n"
                f"Cal pts : {self.peak_mgr.n_calibration_points(ch_id)}")
        r = self.fit_results.get(ch_id)
        if r:
            status = "❌ BAD" if r.bad_channel else "✅ OK"
            info  += f"\nFit     : {status}  χ²/NDF={r.chi2_ndf:.4f}"
        self.lbl_ch_info.setText(info)

    def _update_override_ui(self, ch_id: int):
        has = self.peak_mgr.has_override(ch_id)
        self.chk_override.blockSignals(True)
        self.chk_override.setChecked(has)
        self.chk_override.blockSignals(False)

    # ------------------------------------------------------------------ #
    # Peak detection
    # ------------------------------------------------------------------ #

    def _detect_peaks(self):
        ch_id = self.current_channel
        sp    = self.loader.get_spectrum(ch_id)
        if sp is None:
            return

        threshold = self.sl_threshold.value() / 100.0
        sigma     = self.sb_sigma.value()
        max_peaks = self.sb_max_peaks.value()
        backend   = self.loader.backend

        positions = PeakManager.detect_peaks(
            sp.bin_centers, sp.counts,
            sigma=sigma, threshold=threshold,
            max_peaks=max_peaks, backend=backend)

        self._detected_positions = positions

        if not positions:
            self.lbl_detected.setText(
                "No peaks found — try lowering threshold or increasing sigma.")
            QMessageBox.information(
                self, "No Peaks",
                "No peaks detected. Try:\n"
                "  • Lowering the threshold\n"
                "  • Increasing sigma\n"
                "  • Using 'Click' mode to mark peaks manually")
            return

        self.lbl_detected.setText(
            f"{len(positions)} peak(s) detected  "
            f"[{'TSpectrum' if backend=='pyroot' else 'scipy'}]:\n"
            + ", ".join(f"{p:.1f}" for p in positions))
        self.btn_propagate.setEnabled(True)

        # Populate detected peaks table (energy column = unassigned)
        self.tbl_detected.setRowCount(0)
        for pos in positions:
            row = self.tbl_detected.rowCount()
            self.tbl_detected.insertRow(row)
            self.tbl_detected.setItem(
                row, 0, QTableWidgetItem(f"{pos:.2f}"))
            self.tbl_detected.setItem(
                row, 1, QTableWidgetItem("—  (double-click to assign)"))
            self.tbl_detected.setItem(
                row, 2, QTableWidgetItem(""))

        # Draw on spectrum
        self.spectrum_canvas.plot_spectrum(
            sp.bin_centers, sp.counts,
            title=f"Channel {ch_id} — {sp.n_entries} entries")
        self.spectrum_canvas.draw_detected_peaks(positions)
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
            adc = float(adc_item.text())
            self._open_assign_dialog(adc)

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
        adc = dlg.sb_adc.value()
        eng = dlg.sb_energy.value()
        lbl = dlg.le_label.text()
        ch_id = self.current_channel

        if self.chk_override.isChecked():
            self.peak_mgr.add_channel_peak(ch_id, adc, eng, lbl)
        else:
            self.peak_mgr.add_global_peak(adc, eng, lbl)

        self._refresh_cal_table(ch_id)
        self._update_detail_view(ch_id)
        self.btn_propagate.setEnabled(True)

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
        if self.chk_override.isChecked():
            self.peak_mgr.remove_channel_peak(ch_id, row)
        else:
            self.peak_mgr.remove_global_peak(row)
        self._refresh_cal_table(ch_id)
        self._update_detail_view(ch_id)

    def _reset_to_global(self):
        ch_id = self.current_channel
        if ch_id < 0:
            return
        self.peak_mgr.reset_channel_to_global(ch_id)
        self.chk_override.blockSignals(True)
        self.chk_override.setChecked(False)
        self.chk_override.blockSignals(False)
        self._refresh_cal_table(ch_id)
        self._update_detail_view(ch_id)

    def _on_override_toggled(self, checked: bool):
        ch_id = self.current_channel
        if ch_id < 0:
            return
        if not checked:
            self.peak_mgr.reset_channel_to_global(ch_id)
        self._refresh_cal_table(ch_id)
        self._update_detail_view(ch_id)

    def _refresh_cal_table(self, ch_id: int):
        peaks = self.peak_mgr.get_peaks(ch_id)
        self.tbl_cal.setRowCount(0)
        for p in peaks:
            row = self.tbl_cal.rowCount()
            self.tbl_cal.insertRow(row)
            self.tbl_cal.setItem(
                row, 0, QTableWidgetItem(f"{p.adc_position:.2f}"))
            self.tbl_cal.setItem(
                row, 1, QTableWidgetItem(f"{p.known_energy:.2f}"))
            self.tbl_cal.setItem(
                row, 2, QTableWidgetItem(p.label))

    # ------------------------------------------------------------------ #
    # Fitting
    # ------------------------------------------------------------------ #

    def _propagate_peaks(self):
        """
        For each peak assigned on the reference channel (current or global),
        search all other channels for the best matching peak within
        ± sb_prop_window ADC counts and assign it with the same known energy.
        """
        ref_ch  = self.current_channel
        window  = self.sb_prop_window.value()
        source  = self.cb_prop_source.currentData()
        override = self.chk_prop_override.isChecked()

        # Gather reference peaks
        if source == "current":
            ref_peaks = self.peak_mgr.get_peaks(ref_ch)
        else:
            ref_peaks = self.peak_mgr.global_peaks

        if not ref_peaks:
            QMessageBox.warning(self, "No Reference Peaks",
                "Assign at least one peak on the current channel first, "
                "or switch source to 'Global peak list'.")
            return

        all_channels = self.loader.get_channel_ids()
        target_chs   = [ch for ch in all_channels if ch != ref_ch]

        if not target_chs:
            QMessageBox.information(self, "Single Channel",
                "Only one channel loaded — nothing to propagate to.")
            return

        backend = self.loader.backend
        sigma   = self.sb_sigma.value()
        thresh  = self.sl_threshold.value() / 100.0
        n_found_total = 0
        n_failed      = 0

        for ch_id in target_chs:
            sp = self.loader.get_spectrum(ch_id)
            if sp is None:
                n_failed += 1
                continue

            new_peaks = PeakManager.find_peaks_in_windows(
                bin_centers = sp.bin_centers,
                counts      = sp.counts,
                reference_peaks = ref_peaks,
                window      = window,
                sigma       = sigma,
                threshold   = thresh,
                backend     = backend)

            if not new_peaks:
                n_failed += 1
                continue

            if override:
                # Set as per-channel override
                self.peak_mgr.set_channel_peaks(ch_id, new_peaks)
            else:
                # Replace global peaks with found positions
                # (only if all ref peaks were found)
                if len(new_peaks) == len(ref_peaks):
                    self.peak_mgr.set_channel_peaks(ch_id, new_peaks)

            n_found_total += 1

        # Refresh current view
        self._update_detail_view(ref_ch)

        msg = (f"Propagated {len(ref_peaks)} peak(s) to "
               f"{n_found_total}/{len(target_chs)} channel(s).")
        if n_failed > 0:
            msg += f"  ({n_failed} channel(s) had no matching peaks.)"
        self.lbl_prop_status.setText(msg)
        self.statusBar().showMessage(msg)

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
        # Keep calibrated spectrum tab in sync with latest fit results
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
        # Store a direct reference so it can access calibrated spectra
        self._resolution_tab._main_window = self
        self.tabs.addTab(self._resolution_tab, "📐 Resolution")

    def _send_to_resolution_tab(self, ch_id: int, cal):
        self._resolution_tab.receive_spectrum(ch_id, cal)
        # Switch to resolution tab
        for i in range(self.tabs.count()):
            if "Resolution" in self.tabs.tabText(i):
                self.tabs.setCurrentIndex(i)
                break

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

    # ---- inserted methods (model change + custom expr support) ----

    def _on_model_changed(self, idx: int):
        model = self.cb_model.currentData()
        self.le_custom_expr.setEnabled(model == "custom")
        # Update trend param combo for nonlinear (4 params) vs linear (2)
        param_names = {"linear": ["P0","P1"],
                       "nonlinear": ["P0","P1","P2","P3"],
                       "custom": ["p0","p1","p2","p3"]}.get(model, ["P0","P1"])
        self.cb_trend_param.blockSignals(True)
        self.cb_trend_param.clear()
        self.cb_trend_param.addItems(param_names)
        self.cb_trend_param.blockSignals(False)

