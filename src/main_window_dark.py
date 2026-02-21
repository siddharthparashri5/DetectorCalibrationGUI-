"""
MainWindow — DetectorCalibGUI
Full PyQt5 application window.
Layout:
  Left panel  : file loading, data source selection, channel nav
  Center top  : spectrum canvas (Matplotlib embedded)
  Center mid  : calibration curve canvas
  Right panel : peak table, fit model, export controls
  Bottom      : status bar + progress
  Tabs        : Single Channel | Grid Overview | Coefficient Trends
"""

from __future__ import annotations
import os
import numpy as np
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QTabWidget, QLabel, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
    QLineEdit, QTableWidget, QTableWidgetItem, QFileDialog, QGroupBox,
    QCheckBox, QProgressBar, QStatusBar, QMessageBox, QScrollArea,
    QGridLayout, QSizePolicy, QHeaderView, QDialog, QFormLayout,
    QDialogButtonBox, QTextEdit, QApplication
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QColor, QFont, QIcon

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
from src.calib_fitter import CalibrationFitter
from src.output_writer import OutputWriter


# ======================================================================== #
# Worker thread for batch fitting
# ======================================================================== #

class FitWorker(QThread):
    progress = pyqtSignal(int, int)
    finished = pyqtSignal(dict)
    error    = pyqtSignal(str)

    def __init__(self, fitter, channel_ids, peak_manager, model, custom_expr):
        super().__init__()
        self.fitter       = fitter
        self.channel_ids  = channel_ids
        self.peak_manager = peak_manager
        self.model        = model
        self.custom_expr  = custom_expr

    def run(self):
        try:
            results = self.fitter.fit_all(
                self.channel_ids, self.peak_manager,
                self.model, self.custom_expr,
                progress_callback=lambda d, t: self.progress.emit(d, t)
            )
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))


# ======================================================================== #
# Spectrum Canvas
# ======================================================================== #

class SpectrumCanvas(FigureCanvas):
    """
    Interactive spectrum plot.
    Left-click  → mark a peak (fires peak_clicked signal with x position)
    """
    peak_clicked = pyqtSignal(float)

    def __init__(self, parent=None):
        self.fig, self.ax = plt.subplots(figsize=(7, 3.5))
        self.fig.tight_layout(pad=2)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._peak_lines = []
        self._click_mode = False
        self.mpl_connect("button_press_event", self._on_click)

    def set_click_mode(self, enabled: bool):
        self._click_mode = enabled
        if enabled:
            self.setCursor(Qt.CrossCursor)
        else:
            self.setCursor(Qt.ArrowCursor)

    def _on_click(self, event):
        if not self._click_mode:
            return
        if event.inaxes != self.ax:
            return
        if event.button == 1:
            self.peak_clicked.emit(event.xdata)

    def plot_spectrum(self, bin_centers: np.ndarray, counts: np.ndarray,
                       title: str = "", channel_id: int = -1):
        self.ax.clear()
        self._peak_lines.clear()
        self.ax.step(bin_centers, counts, where="mid",
                      color="#2196F3", linewidth=0.9, label="Spectrum")
        self.ax.set_xlabel("ADC / Bin", fontsize=9)
        self.ax.set_ylabel("Counts",    fontsize=9)
        self.ax.set_title(title or f"Channel {channel_id}", fontsize=10)
        self.ax.set_yscale("log")
        self.ax.set_ylim(bottom=0.5)
        self.ax.grid(True, alpha=0.3)
        self.fig.tight_layout(pad=1.5)
        self.draw()

    def draw_peaks(self, peaks: list[Peak], color="#E91E63"):
        """Draw vertical lines for current peak positions."""
        for line in self._peak_lines:
            try:
                line.remove()
            except Exception:
                pass
        self._peak_lines.clear()
        for p in peaks:
            line = self.ax.axvline(p.adc_position, color=color,
                                    linestyle="--", linewidth=1.2, alpha=0.8)
            self.ax.text(p.adc_position, self.ax.get_ylim()[1] * 0.7,
                          f" {p.known_energy:.0f} keV",
                          color=color, fontsize=7, rotation=90, va="top")
            self._peak_lines.append(line)
        self.draw()


class CalibCurveCanvas(FigureCanvas):
    """Shows the fitted calibration curve: ADC vs Energy."""

    def __init__(self, parent=None):
        self.fig, (self.ax_main, self.ax_res) = plt.subplots(
            2, 1, figsize=(7, 3), gridspec_kw={"height_ratios": [3, 1]})
        self.fig.tight_layout(pad=1.5)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def plot_result(self, result):
        from src.calib_fitter import FitResult
        self.ax_main.clear()
        self.ax_res.clear()

        if result is None or not result.success:
            self.ax_main.text(0.5, 0.5, "No fit result",
                               ha="center", va="center",
                               transform=self.ax_main.transAxes)
            self.draw()
            return

        adc = result.adc_points
        eng = result.energy_points
        res = result.residuals

        # Fitted curve
        x_fit = np.linspace(adc.min() * 0.9, adc.max() * 1.1, 500)
        try:
            import numpy.polynomial.polynomial as poly
            if result.model.startswith("Polynomial"):
                y_fit = np.polyval(result.params, x_fit)
            else:
                y_fit = None
        except Exception:
            y_fit = None

        if y_fit is not None:
            self.ax_main.plot(x_fit, y_fit, "-", color="#1976D2",
                               linewidth=1.5, label="Fit")
        self.ax_main.scatter(adc, eng, color="#E91E63", zorder=5,
                              s=50, label="Cal. points")
        self.ax_main.set_ylabel("Energy (keV)", fontsize=8)
        self.ax_main.set_title(
            f"Ch {result.channel_id} | {result.model} | "
            f"χ²/NDF={result.chi2_ndf:.3f}", fontsize=9)
        self.ax_main.legend(fontsize=8)
        self.ax_main.grid(True, alpha=0.3)

        # Residuals
        self.ax_res.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        self.ax_res.scatter(adc, res, color="#FF5722", s=40, zorder=5)
        self.ax_res.set_xlabel("ADC / Bin", fontsize=8)
        self.ax_res.set_ylabel("Residual\n(keV)", fontsize=7)
        self.ax_res.grid(True, alpha=0.3)

        self.fig.tight_layout(pad=1.2)
        self.draw()


# ======================================================================== #
# Overview Grid
# ======================================================================== #

class OverviewGrid(QScrollArea):
    """
    Thumbnail grid of all channel spectra.
    Click a thumbnail to navigate to that channel in the detail view.
    """
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
        self._thumbnails: dict[int, QLabel] = {}

    def populate(self, spectra: dict, results: dict | None = None):
        # Clear existing
        for i in reversed(range(self._layout.count())):
            self._layout.itemAt(i).widget().setParent(None)
        self._thumbnails.clear()

        channel_ids = sorted(spectra.keys())
        for i, ch_id in enumerate(channel_ids):
            sp = spectra[ch_id]

            # Render thumbnail
            fig, ax = plt.subplots(figsize=(self.THUMB_W/72,
                                             self.THUMB_H/72), dpi=72)
            ax.step(sp.bin_centers, sp.counts, where="mid",
                     linewidth=0.5, color="#2196F3")
            ax.set_yscale("log")
            ax.set_title(f"Ch {ch_id}", fontsize=5, pad=1)
            ax.tick_params(labelsize=4)
            ax.set_ylim(bottom=0.5)
            fig.tight_layout(pad=0.3)

            canvas = FigureCanvas(fig)
            canvas.setFixedSize(self.THUMB_W, self.THUMB_H)

            # Color border by status
            border_color = "#444"
            if results:
                r = results.get(ch_id)
                if r:
                    border_color = "#f44336" if r.bad_channel else "#4CAF50"
            canvas.setStyleSheet(
                f"border: 2px solid {border_color}; border-radius: 3px;")

            # Click → select channel
            ch_id_capture = ch_id
            canvas.mousePressEvent = lambda e, c=ch_id_capture: (
                self.channel_selected.emit(c))

            row, col = divmod(i, self.COLS)
            self._layout.addWidget(canvas, row, col)
            plt.close(fig)

        self._widget.adjustSize()


# ======================================================================== #
# Trend Plots
# ======================================================================== #

class TrendCanvas(FigureCanvas):
    """Plots calibration coefficient value vs channel ID."""

    def __init__(self, parent=None):
        self.fig, self.axes = plt.subplots(1, 1, figsize=(8, 3))
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def plot_trends(self, results: dict, param_index: int = 0,
                     param_name: str = "p0"):
        self.axes.clear()
        good = {ch: r for ch, r in results.items()
                if not r.bad_channel and r.success
                and param_index < len(r.params)}
        if not good:
            self.axes.text(0.5, 0.5, "No good channels to plot",
                            ha="center", va="center",
                            transform=self.axes.transAxes)
            self.draw()
            return

        channels = sorted(good.keys())
        vals = [good[ch].params[param_index]   for ch in channels]
        errs = [good[ch].uncertainties[param_index] for ch in channels]

        # Mark bad channels on x axis
        bad_chs = [ch for ch, r in results.items() if r.bad_channel]

        self.axes.errorbar(channels, vals, yerr=errs, fmt="o",
                            markersize=3, linewidth=0.8,
                            color="#1976D2", ecolor="#90CAF9",
                            capsize=2, label=param_name)
        for bch in bad_chs:
            self.axes.axvline(bch, color="#f44336", alpha=0.3,
                               linewidth=0.6)

        self.axes.set_xlabel("Channel ID", fontsize=9)
        self.axes.set_ylabel(param_name, fontsize=9)
        self.axes.set_title(f"Calibration Parameter '{param_name}' vs Channel",
                             fontsize=10)
        self.axes.grid(True, alpha=0.3)
        bad_patch = mpatches.Patch(color="#f44336", alpha=0.4,
                                    label=f"Bad ({len(bad_chs)})")
        self.axes.legend(handles=[*self.axes.get_legend_handles_labels()[0],
                                   bad_patch], fontsize=8)
        self.fig.tight_layout(pad=1.5)
        self.draw()


# ======================================================================== #
# Dialog: Load ROOT file — choose mode + branches
# ======================================================================== #

class LoadDialog(QDialog):
    def __init__(self, file_info: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configure Data Source")
        self.setMinimumWidth(480)
        self.result_mode = ""
        self.result_tree = ""
        self.result_ch_branch = ""
        self.result_adc_branch = ""
        self.result_hist_names = []
        self.result_nbins = 1024

        layout = QVBoxLayout(self)

        has_trees = bool(file_info["trees"])
        has_hists = bool(file_info["histograms"])

        # Mode selection
        mode_box = QGroupBox("Data Mode")
        mode_layout = QVBoxLayout(mode_box)
        self.rb_tree = QCheckBox("TTree (build spectra from branches)")
        self.rb_hist = QCheckBox("TH1 Histograms (pre-filled)")
        self.rb_tree.setEnabled(has_trees)
        self.rb_hist.setEnabled(has_hists)
        if has_trees:
            self.rb_tree.setChecked(True)
        elif has_hists:
            self.rb_hist.setChecked(True)
        mode_layout.addWidget(self.rb_tree)
        mode_layout.addWidget(self.rb_hist)
        layout.addWidget(mode_box)

        # TTree options
        self.tree_group = QGroupBox("TTree Options")
        tg = QFormLayout(self.tree_group)
        self.cb_tree_name = QComboBox()
        for t in file_info["trees"]:
            self.cb_tree_name.addItem(
                f"{t['name']} ({t['entries']} entries)", t["name"])
        self.cb_channel_branch = QComboBox()
        self.cb_adc_branch     = QComboBox()
        self.sb_nbins          = QSpinBox()
        self.sb_nbins.setRange(64, 65536)
        self.sb_nbins.setValue(1024)
        tg.addRow("TTree:",          self.cb_tree_name)
        tg.addRow("Channel branch:", self.cb_channel_branch)
        tg.addRow("ADC branch:",     self.cb_adc_branch)
        tg.addRow("Histogram bins:", self.sb_nbins)
        layout.addWidget(self.tree_group)
        self.cb_tree_name.currentIndexChanged.connect(self._update_branches)
        self._populate_branches(file_info)

        # TH1 options
        self.hist_group = QGroupBox("Histogram Options")
        hg = QVBoxLayout(self.hist_group)
        hg.addWidget(QLabel(f"{len(file_info['histograms'])} histogram(s) found — all will be loaded."))
        layout.addWidget(self.hist_group)

        # Toggle visibility
        self.rb_tree.toggled.connect(self._toggle_mode)
        self.rb_hist.toggled.connect(self._toggle_mode)
        self._toggle_mode()

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._file_info = file_info

    def _populate_branches(self, file_info):
        if not file_info["trees"]:
            return
        tree = file_info["trees"][0]
        for b in tree["branches"]:
            self.cb_channel_branch.addItem(b)
            self.cb_adc_branch.addItem(b)
        if len(tree["branches"]) > 1:
            self.cb_adc_branch.setCurrentIndex(1)

    def _update_branches(self, idx):
        self.cb_channel_branch.clear()
        self.cb_adc_branch.clear()
        tree = self._file_info["trees"][idx]
        for b in tree["branches"]:
            self.cb_channel_branch.addItem(b)
            self.cb_adc_branch.addItem(b)

    def _toggle_mode(self):
        use_tree = self.rb_tree.isChecked()
        self.tree_group.setVisible(use_tree)
        self.hist_group.setVisible(not use_tree)

    def _accept(self):
        if self.rb_tree.isChecked():
            self.result_mode       = "ttree"
            self.result_tree       = self.cb_tree_name.currentData()
            self.result_ch_branch  = self.cb_channel_branch.currentText()
            self.result_adc_branch = self.cb_adc_branch.currentText()
            self.result_nbins      = self.sb_nbins.value()
        else:
            self.result_mode = "th1"
        self.accept()


# ======================================================================== #
# Peak Assignment Dialog
# ======================================================================== #

class AddPeakDialog(QDialog):
    def __init__(self, adc_position: float = 0.0, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Assign Peak Energy")
        layout = QFormLayout(self)
        self.sb_adc = QDoubleSpinBox()
        self.sb_adc.setRange(0, 1e9)
        self.sb_adc.setDecimals(2)
        self.sb_adc.setValue(adc_position)
        self.sb_energy = QDoubleSpinBox()
        self.sb_energy.setRange(0, 1e7)
        self.sb_energy.setDecimals(2)
        self.sb_energy.setValue(511.0)
        self.le_label = QLineEdit()
        self.le_label.setPlaceholderText("e.g. 511 keV annihilation")

        # Common energies quick buttons
        common_box = QHBoxLayout()
        for e in [511.0, 1274.5, 661.7, 1332.5, 1173.2, 122.1]:
            btn = QPushButton(f"{e:.1f}")
            btn.setFixedWidth(60)
            btn.clicked.connect(lambda _, en=e: self.sb_energy.setValue(en))
            common_box.addWidget(btn)

        layout.addRow("ADC position:", self.sb_adc)
        layout.addRow("Known energy (keV):", self.sb_energy)
        layout.addRow("Quick energies:", common_box)  # type: ignore
        layout.addRow("Label (optional):", self.le_label)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)


# ======================================================================== #
# Main Window
# ======================================================================== #

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DetectorCalibGUI — Energy Calibration Tool")
        self.setMinimumSize(1300, 850)

        # Core objects
        self.loader      = ROOTFileLoader()
        self.peak_mgr    = PeakManager()
        self.fitter      = CalibrationFitter()
        self.fit_results: dict = {}
        self.current_channel: int = -1
        self._click_mode = False

        self._build_ui()
        self._apply_style()

    # ------------------------------------------------------------------ #
    # UI Construction
    # ------------------------------------------------------------------ #

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setSpacing(0)
        root_layout.setContentsMargins(0, 0, 0, 0)

        # ── Toolbar ──────────────────────────────────────────────────── #
        toolbar = self._build_toolbar()
        root_layout.addWidget(toolbar)

        # ── Main splitter ─────────────────────────────────────────────── #
        splitter = QSplitter(Qt.Horizontal)
        root_layout.addWidget(splitter, stretch=1)

        # Left panel
        left = self._build_left_panel()
        left.setMinimumWidth(220)
        left.setMaximumWidth(280)
        splitter.addWidget(left)

        # Tab area (center + right)
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.North)
        splitter.addWidget(self.tabs)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        self._build_detail_tab()
        self._build_grid_tab()
        self._build_trends_tab()

        # ── Status bar ───────────────────────────────────────────────── #
        self.statusBar().showMessage("Ready — load a ROOT file to begin.")
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximumWidth(200)
        self.statusBar().addPermanentWidget(self.progress_bar)

    def _build_toolbar(self) -> QWidget:
        bar = QWidget()
        bar.setObjectName("toolbar")
        bar.setFixedHeight(44)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(8, 4, 8, 4)

        self.btn_open = QPushButton("📂  Open ROOT File")
        self.btn_open.clicked.connect(self._open_file)
        layout.addWidget(self.btn_open)

        self.lbl_file = QLabel("No file loaded")
        self.lbl_file.setObjectName("fileLabel")
        layout.addWidget(self.lbl_file, stretch=1)

        self.btn_fit_all = QPushButton("⚡  Fit All Channels")
        self.btn_fit_all.setEnabled(False)
        self.btn_fit_all.clicked.connect(self._fit_all)
        layout.addWidget(self.btn_fit_all)

        self.btn_export = QPushButton("💾  Export Results")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self._export)
        layout.addWidget(self.btn_export)

        return bar

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        # Channel navigation
        nav_box = QGroupBox("Channel Navigation")
        nav_layout = QVBoxLayout(nav_box)
        nav_h = QHBoxLayout()
        self.btn_prev = QPushButton("◀")
        self.btn_prev.setFixedWidth(36)
        self.btn_prev.clicked.connect(self._prev_channel)
        self.cb_channel = QComboBox()
        self.cb_channel.setMinimumWidth(100)
        self.cb_channel.currentIndexChanged.connect(self._on_channel_changed)
        self.btn_next = QPushButton("▶")
        self.btn_next.setFixedWidth(36)
        self.btn_next.clicked.connect(self._next_channel)
        nav_h.addWidget(self.btn_prev)
        nav_h.addWidget(self.cb_channel, stretch=1)
        nav_h.addWidget(self.btn_next)
        nav_layout.addLayout(nav_h)
        self.lbl_channel_info = QLabel("—")
        self.lbl_channel_info.setWordWrap(True)
        nav_layout.addWidget(self.lbl_channel_info)
        layout.addWidget(nav_box)

        # Fit model
        model_box = QGroupBox("Calibration Model")
        model_layout = QFormLayout(model_box)
        self.cb_model = QComboBox()
        for m in ["poly1", "poly2", "poly3", "poly4", "poly5", "custom"]:
            self.cb_model.addItem(m)
        self.cb_model.currentTextChanged.connect(self._on_model_changed)
        self.le_custom_expr = QLineEdit()
        self.le_custom_expr.setPlaceholderText("e.g. a*x**2 + b*x + c")
        self.le_custom_expr.setEnabled(False)
        model_layout.addRow("Model:", self.cb_model)
        model_layout.addRow("Custom fn:", self.le_custom_expr)
        layout.addWidget(model_box)

        # Fit single channel
        self.btn_fit_one = QPushButton("▶  Fit This Channel")
        self.btn_fit_one.setEnabled(False)
        self.btn_fit_one.clicked.connect(self._fit_current)
        layout.addWidget(self.btn_fit_one)

        # Peak input
        peak_box = QGroupBox("Global Peaks (all channels)")
        peak_layout = QVBoxLayout(peak_box)
        self.tbl_global_peaks = QTableWidget(0, 3)
        self.tbl_global_peaks.setHorizontalHeaderLabels(
            ["ADC pos", "Energy (keV)", "Label"])
        self.tbl_global_peaks.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch)
        self.tbl_global_peaks.setMaximumHeight(140)
        peak_layout.addWidget(self.tbl_global_peaks)

        peak_btns = QHBoxLayout()
        self.btn_add_peak_manual = QPushButton("+ Add")
        self.btn_add_peak_manual.clicked.connect(
            lambda: self._open_add_peak_dialog(0.0, global_peak=True))
        self.btn_del_peak = QPushButton("− Del")
        self.btn_del_peak.clicked.connect(self._delete_global_peak)
        self.btn_click_peak = QPushButton("🖱 Click")
        self.btn_click_peak.setCheckable(True)
        self.btn_click_peak.toggled.connect(self._toggle_click_mode)
        self.btn_auto_peaks = QPushButton("🔍 Auto")
        self.btn_auto_peaks.clicked.connect(self._auto_find_peaks)
        peak_btns.addWidget(self.btn_add_peak_manual)
        peak_btns.addWidget(self.btn_del_peak)
        peak_btns.addWidget(self.btn_click_peak)
        peak_btns.addWidget(self.btn_auto_peaks)
        peak_layout.addLayout(peak_btns)

        self.chk_override = QCheckBox("Override peaks for this channel")
        self.chk_override.toggled.connect(self._on_override_toggled)
        peak_layout.addWidget(self.chk_override)
        layout.addWidget(peak_box)

        # Channel-specific peak table (shown when override is on)
        self.ch_peak_box = QGroupBox("Channel-Specific Peaks")
        ch_pk_layout = QVBoxLayout(self.ch_peak_box)
        self.tbl_ch_peaks = QTableWidget(0, 3)
        self.tbl_ch_peaks.setHorizontalHeaderLabels(
            ["ADC pos", "Energy (keV)", "Label"])
        self.tbl_ch_peaks.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch)
        self.tbl_ch_peaks.setMaximumHeight(120)
        ch_pk_layout.addWidget(self.tbl_ch_peaks)
        ch_pk_btns = QHBoxLayout()
        self.btn_add_ch_peak = QPushButton("+ Add")
        self.btn_add_ch_peak.clicked.connect(
            lambda: self._open_add_peak_dialog(0.0, global_peak=False))
        self.btn_del_ch_peak = QPushButton("− Del")
        self.btn_del_ch_peak.clicked.connect(self._delete_ch_peak)
        ch_pk_btns.addWidget(self.btn_add_ch_peak)
        ch_pk_btns.addWidget(self.btn_del_ch_peak)
        ch_pk_layout.addLayout(ch_pk_btns)
        self.ch_peak_box.setVisible(False)
        layout.addWidget(self.ch_peak_box)

        layout.addStretch()

        # Bad channel summary
        self.lbl_bad = QLabel("")
        self.lbl_bad.setObjectName("badLabel")
        self.lbl_bad.setWordWrap(True)
        layout.addWidget(self.lbl_bad)

        return panel

    def _build_detail_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Spectrum canvas + toolbar
        self.spectrum_canvas = SpectrumCanvas()
        self.spectrum_canvas.peak_clicked.connect(self._on_peak_click)
        self.spectrum_toolbar = NavigationToolbar(self.spectrum_canvas, tab)
        layout.addWidget(self.spectrum_toolbar)
        layout.addWidget(self.spectrum_canvas, stretch=3)

        # Calibration curve
        self.calib_canvas = CalibCurveCanvas()
        layout.addWidget(self.calib_canvas, stretch=2)

        self.tabs.addTab(tab, "🔬 Single Channel")

    def _build_grid_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(4, 4, 4, 4)
        info = QLabel("Thumbnail grid — click any channel to open it in detail view.  "
                       "Green border = good fit | Red border = bad channel.")
        info.setWordWrap(True)
        layout.addWidget(info)
        self.overview_grid = OverviewGrid()
        self.overview_grid.channel_selected.connect(self._go_to_channel)
        layout.addWidget(self.overview_grid)
        self.tabs.addTab(tab, "📊 Overview Grid")

    def _build_trends_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(4, 4, 4, 4)

        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Parameter:"))
        self.cb_trend_param = QComboBox()
        self.cb_trend_param.addItems(["p0", "p1", "p2"])
        self.cb_trend_param.currentIndexChanged.connect(self._update_trends)
        ctrl.addWidget(self.cb_trend_param)
        ctrl.addStretch()
        layout.addLayout(ctrl)

        self.trend_canvas = TrendCanvas()
        layout.addWidget(self.trend_canvas)
        self.tabs.addTab(tab, "📈 Coefficient Trends")

    # ------------------------------------------------------------------ #
    # File Loading
    # ------------------------------------------------------------------ #

    def _open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open ROOT File", "", "ROOT Files (*.root);;All Files (*)")
        if not path:
            return

        try:
            info = self.loader.open(path)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            return

        if not info["trees"] and not info["histograms"]:
            QMessageBox.warning(self, "Empty File",
                                "No TTrees or TH1 histograms found in file.")
            return

        dialog = LoadDialog(info, self)
        if dialog.exec_() != QDialog.Accepted:
            return

        try:
            if dialog.result_mode == "ttree":
                self.loader.load_from_ttree(
                    dialog.result_tree,
                    dialog.result_ch_branch,
                    dialog.result_adc_branch,
                    dialog.result_nbins)
            else:
                self.loader.load_from_th1()
        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e))
            return

        self.lbl_file.setText(os.path.basename(path))
        self._populate_channel_combo()
        self.btn_fit_all.setEnabled(True)
        self.btn_fit_one.setEnabled(True)
        n = len(self.loader.spectra)
        self.statusBar().showMessage(
            f"Loaded {n} channel(s) from {os.path.basename(path)}")

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
    # Channel Navigation
    # ------------------------------------------------------------------ #

    def _on_channel_changed(self, idx: int):
        if idx < 0 or self.cb_channel.count() == 0:
            return
        ch_id = self.cb_channel.currentData()
        if ch_id is None:
            return
        self.current_channel = ch_id
        self._update_detail_view(ch_id)
        self._update_override_ui(ch_id)

    def _prev_channel(self):
        idx = self.cb_channel.currentIndex()
        if idx > 0:
            self.cb_channel.setCurrentIndex(idx - 1)

    def _next_channel(self):
        idx = self.cb_channel.currentIndex()
        if idx < self.cb_channel.count() - 1:
            self.cb_channel.setCurrentIndex(idx + 1)

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
            title=f"Channel {ch_id} — {sp.n_entries} entries",
            channel_id=ch_id)
        effective_peaks = self.peak_mgr.get_peaks(ch_id)
        self.spectrum_canvas.draw_peaks(effective_peaks)

        result = self.fit_results.get(ch_id)
        self.calib_canvas.plot_result(result)

        info = f"Entries: {sp.n_entries}\n"
        info += f"Source: {sp.source}\n"
        info += f"Peaks: {self.peak_mgr.n_calibration_points(ch_id)}"
        if result:
            status = "❌ BAD" if result.bad_channel else "✅ OK"
            info += f"\nFit: {status}  χ²/NDF={result.chi2_ndf:.3f}"
        self.lbl_channel_info.setText(info)

    def _update_override_ui(self, ch_id: int):
        has_override = self.peak_mgr.has_override(ch_id)
        self.chk_override.blockSignals(True)
        self.chk_override.setChecked(has_override)
        self.chk_override.blockSignals(False)
        self.ch_peak_box.setVisible(has_override)
        if has_override:
            self._refresh_ch_peak_table(ch_id)

    # ------------------------------------------------------------------ #
    # Peak Management
    # ------------------------------------------------------------------ #

    def _toggle_click_mode(self, checked: bool):
        self._click_mode = checked
        self.spectrum_canvas.set_click_mode(checked)
        if checked:
            self.statusBar().showMessage(
                "Click mode ON — left-click on spectrum to mark a peak")
        else:
            self.statusBar().showMessage("Click mode OFF")

    def _on_peak_click(self, x: float):
        """Called when user clicks on spectrum canvas."""
        self._open_add_peak_dialog(
            x, global_peak=not self.chk_override.isChecked())

    def _open_add_peak_dialog(self, adc_pos: float, global_peak: bool = True):
        dlg = AddPeakDialog(adc_pos, self)
        if dlg.exec_() != QDialog.Accepted:
            return
        adc  = dlg.sb_adc.value()
        eng  = dlg.sb_energy.value()
        lbl  = dlg.le_label.text()

        if global_peak:
            self.peak_mgr.add_global_peak(adc, eng, lbl)
            self._refresh_global_peak_table()
        else:
            ch_id = self.current_channel
            if ch_id < 0:
                return
            self.peak_mgr.add_channel_peak(ch_id, adc, eng, lbl)
            self._refresh_ch_peak_table(ch_id)

        self._update_detail_view(self.current_channel)

    def _auto_find_peaks(self):
        ch_id = self.current_channel
        sp = self.loader.get_spectrum(ch_id)
        if sp is None:
            return
        positions = PeakManager.auto_find_peaks(sp.bin_centers, sp.counts)
        if not positions:
            QMessageBox.information(self, "Auto-find", "No peaks detected.")
            return

        # Show found positions, let user assign energies one by one
        for adc_pos in positions:
            dlg = AddPeakDialog(adc_pos, self)
            dlg.setWindowTitle(f"Auto-detected peak at ADC={adc_pos:.1f}")
            if dlg.exec_() == QDialog.Accepted:
                use_global = not self.chk_override.isChecked()
                if use_global:
                    self.peak_mgr.add_global_peak(
                        dlg.sb_adc.value(), dlg.sb_energy.value(),
                        dlg.le_label.text(), auto=True)
                else:
                    self.peak_mgr.add_channel_peak(
                        ch_id, dlg.sb_adc.value(), dlg.sb_energy.value(),
                        dlg.le_label.text())

        self._refresh_global_peak_table()
        if self.chk_override.isChecked():
            self._refresh_ch_peak_table(ch_id)
        self._update_detail_view(ch_id)

    def _delete_global_peak(self):
        row = self.tbl_global_peaks.currentRow()
        if row < 0:
            return
        self.peak_mgr.remove_global_peak(row)
        self._refresh_global_peak_table()
        self._update_detail_view(self.current_channel)

    def _delete_ch_peak(self):
        row = self.tbl_ch_peaks.currentRow()
        if row < 0:
            return
        self.peak_mgr.remove_channel_peak(self.current_channel, row)
        self._refresh_ch_peak_table(self.current_channel)
        self._update_detail_view(self.current_channel)

    def _on_override_toggled(self, checked: bool):
        ch_id = self.current_channel
        if ch_id < 0:
            return
        self.ch_peak_box.setVisible(checked)
        if not checked:
            self.peak_mgr.reset_channel_to_global(ch_id)
        else:
            self._refresh_ch_peak_table(ch_id)
        self._update_detail_view(ch_id)

    def _refresh_global_peak_table(self):
        self.tbl_global_peaks.setRowCount(0)
        for p in self.peak_mgr.global_peaks:
            row = self.tbl_global_peaks.rowCount()
            self.tbl_global_peaks.insertRow(row)
            self.tbl_global_peaks.setItem(
                row, 0, QTableWidgetItem(f"{p.adc_position:.2f}"))
            self.tbl_global_peaks.setItem(
                row, 1, QTableWidgetItem(f"{p.known_energy:.2f}"))
            self.tbl_global_peaks.setItem(
                row, 2, QTableWidgetItem(p.label))

    def _refresh_ch_peak_table(self, ch_id: int):
        peaks = self.peak_mgr.channel_peaks.get(ch_id, [])
        self.tbl_ch_peaks.setRowCount(0)
        for p in peaks:
            row = self.tbl_ch_peaks.rowCount()
            self.tbl_ch_peaks.insertRow(row)
            self.tbl_ch_peaks.setItem(
                row, 0, QTableWidgetItem(f"{p.adc_position:.2f}"))
            self.tbl_ch_peaks.setItem(
                row, 1, QTableWidgetItem(f"{p.known_energy:.2f}"))
            self.tbl_ch_peaks.setItem(
                row, 2, QTableWidgetItem(p.label))

    # ------------------------------------------------------------------ #
    # Fitting
    # ------------------------------------------------------------------ #

    def _get_model_params(self) -> tuple[str, str]:
        model = self.cb_model.currentText()
        expr  = self.le_custom_expr.text().strip() if model == "custom" else ""
        return model, expr

    def _on_model_changed(self, model: str):
        self.le_custom_expr.setEnabled(model == "custom")

        # Update trend param combo
        n = int(model.replace("poly", "")) + 1 if model.startswith("poly") else 3
        self.cb_trend_param.blockSignals(True)
        self.cb_trend_param.clear()
        self.cb_trend_param.addItems([f"p{i}" for i in range(n)])
        self.cb_trend_param.blockSignals(False)

    def _fit_current(self):
        ch_id = self.current_channel
        if ch_id < 0:
            return
        adc, eng = self.peak_mgr.get_calibration_points(ch_id)
        if len(adc) == 0:
            QMessageBox.warning(self, "No Peaks",
                                "Add calibration peaks first.")
            return
        model, expr = self._get_model_params()
        result = self.fitter.fit_channel(ch_id, adc, eng, model, expr)
        self.fit_results[ch_id] = result
        self._update_detail_view(ch_id)
        self._update_bad_label()

    def _fit_all(self):
        ch_ids = self.loader.get_channel_ids()
        if not ch_ids:
            return
        model, expr = self._get_model_params()

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, len(ch_ids))
        self.btn_fit_all.setEnabled(False)

        self._worker = FitWorker(self.fitter, ch_ids, self.peak_mgr,
                                  model, expr)
        self._worker.progress.connect(
            lambda d, t: self.progress_bar.setValue(d))
        self._worker.finished.connect(self._on_fit_all_done)
        self._worker.error.connect(
            lambda e: QMessageBox.critical(self, "Fit Error", e))
        self._worker.start()

    def _on_fit_all_done(self, results: dict):
        self.fit_results = results
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
    # Trends & bad channel summary
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
            self.lbl_bad.setText(
                f"⚠ {len(bad)} bad channel(s):\n"
                + ", ".join(str(b) for b in bad[:20])
                + ("..." if len(bad) > 20 else ""))
            self.lbl_bad.setStyleSheet("color: #f44336;")
        else:
            self.lbl_bad.setText("✅ All channels OK")
            self.lbl_bad.setStyleSheet("color: #4CAF50;")

    # ------------------------------------------------------------------ #
    # Styling
    # ------------------------------------------------------------------ #

    def _apply_style(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e2e; }
            QWidget { background-color: #1e1e2e; color: #cdd6f4; font-size: 13px; }
            #toolbar { background-color: #181825; border-bottom: 1px solid #313244; }
            #fileLabel { color: #89b4fa; font-weight: bold; padding: 0 8px; }
            QPushButton {
                background-color: #313244; color: #cdd6f4;
                border: 1px solid #45475a; border-radius: 5px;
                padding: 5px 12px; min-width: 80px;
            }
            QPushButton:hover { background-color: #45475a; }
            QPushButton:pressed { background-color: #585b70; }
            QPushButton:checked { background-color: #89b4fa; color: #1e1e2e; }
            QPushButton:disabled { color: #585b70; }
            QGroupBox {
                border: 1px solid #313244; border-radius: 6px;
                margin-top: 8px; padding-top: 4px;
                font-weight: bold; color: #89b4fa;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 8px; }
            QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {
                background-color: #313244; border: 1px solid #45475a;
                border-radius: 4px; padding: 3px 6px; color: #cdd6f4;
            }
            QTableWidget {
                background-color: #181825; gridline-color: #313244;
                border: 1px solid #313244; border-radius: 4px;
            }
            QHeaderView::section {
                background-color: #313244; color: #89b4fa;
                border: none; padding: 4px;
            }
            QTabWidget::pane { border: 1px solid #313244; }
            QTabBar::tab {
                background: #181825; color: #cdd6f4;
                padding: 6px 18px; border-radius: 4px 4px 0 0;
            }
            QTabBar::tab:selected { background: #313244; color: #89b4fa; }
            QScrollArea { border: none; }
            QStatusBar { background-color: #181825; color: #a6adc8; }
            QCheckBox { color: #cdd6f4; }
            QCheckBox::indicator {
                width: 14px; height: 14px;
                border: 1px solid #45475a; border-radius: 3px;
                background: #313244;
            }
            QCheckBox::indicator:checked { background: #89b4fa; }
            #badLabel { font-size: 11px; padding: 4px; }
        """)
        # Dark matplotlib theme
        plt.style.use("dark_background")
        for fig_attr in ["spectrum_canvas", "calib_canvas", "trend_canvas"]:
            canvas = getattr(self, fig_attr, None)
            if canvas:
                canvas.fig.patch.set_facecolor("#1e1e2e")
