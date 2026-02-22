"""
ResolutionTab
=============
Interactive energy resolution analysis.

Workflow:
  1. Receive calibrated spectrum (from CalibratedSpectrumTab or load directly)
  2. User drags a range on the spectrum to define peak fit window
  3. App fits Gaussian+linear BG in that window
  4. Result: FWHM (keV) + R% shown immediately
  5. Repeat for multiple peaks / channels
  6. Trend tab plots FWHM and R% vs channel ID
  7. Export to resolution_results.txt
"""

from __future__ import annotations
import numpy as np
import os

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QGroupBox,
    QLabel, QPushButton, QComboBox, QDoubleSpinBox, QSpinBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QFileDialog,
    QMessageBox, QCheckBox, QSizePolicy, QTabWidget, QAbstractItemView
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar
)
from matplotlib.widgets import SpanSelector
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.resolution import ResolutionCalculator, gaussian_linear_bg, gaussian_only
from src.calib_spectrum import CalibratedSpectrum


class ResolutionTab(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.calc     = ResolutionCalculator()
        self._current_cal: CalibratedSpectrum | None = None
        self._span_lo: float = 0.0
        self._span_hi: float = 0.0
        self._span_selector = None
        self._fit_lines: list = []
        self._build_ui()

    # ------------------------------------------------------------------ #
    # Public: receive spectrum from CalibratedSpectrumTab
    # ------------------------------------------------------------------ #

    def receive_spectrum(self, ch_id: int, cal: CalibratedSpectrum):
        self._current_cal = cal
        self._plot_spectrum(cal)
        self.lbl_channel.setText(
            f"Channel {ch_id}  |  {cal.model}  |  "
            f"{cal.energy_centers[0]:.0f}–{cal.energy_centers[-1]:.0f} keV")
        self.btn_fit_peak.setEnabled(False)
        self.lbl_range.setText("Drag a range on the spectrum to select a peak")

    # ------------------------------------------------------------------ #
    # UI
    # ------------------------------------------------------------------ #

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        # ── Top: channel info + controls ─────────────────────────────── #
        top = QHBoxLayout()

        info_box = QGroupBox("Current Channel")
        info_l   = QHBoxLayout(info_box)
        self.lbl_channel = QLabel("No spectrum loaded — "
                                   "click '📐 Analyse Resolution →' in the "
                                   "Calibrated Spectrum tab first.")
        self.lbl_channel.setStyleSheet("color:#555; font-size:11px;")
        self.lbl_channel.setWordWrap(True)
        info_l.addWidget(self.lbl_channel)
        top.addWidget(info_box, stretch=1)

        fit_box = QGroupBox("Fit Options")
        fit_l   = QHBoxLayout(fit_box)
        fit_l.addWidget(QLabel("Background:"))
        self.cb_bg = QComboBox()
        self.cb_bg.addItem("Gaussian + linear BG", True)
        self.cb_bg.addItem("Gaussian only",         False)
        fit_l.addWidget(self.cb_bg)
        fit_l.addWidget(QLabel("Peak label (keV):"))
        self.sb_peak_label = QDoubleSpinBox()
        self.sb_peak_label.setRange(0, 1e6)
        self.sb_peak_label.setDecimals(1)
        self.sb_peak_label.setValue(511.0)
        self.sb_peak_label.setToolTip(
            "Nominal energy of this peak (used as label in results table)")
        fit_l.addWidget(self.sb_peak_label)
        self.btn_fit_peak = QPushButton("⚡  Fit Peak")
        self.btn_fit_peak.setEnabled(False)
        self.btn_fit_peak.clicked.connect(self._fit_selected_peak)
        fit_l.addWidget(self.btn_fit_peak)
        top.addWidget(fit_box)

        layout.addLayout(top)

        # ── Range info ───────────────────────────────────────────────── #
        range_h = QHBoxLayout()
        self.lbl_range = QLabel(
            "💡 Drag on the spectrum to select a fit range for a peak")
        self.lbl_range.setStyleSheet(
            "color:#1565c0; font-size:11px; padding:2px 4px;")
        range_h.addWidget(self.lbl_range)
        self.btn_clear_fits = QPushButton("✕  Clear Fits (this channel)")
        self.btn_clear_fits.clicked.connect(self._clear_channel_fits)
        range_h.addWidget(self.btn_clear_fits)
        layout.addLayout(range_h)

        # ── Main splitter: spectrum | results ────────────────────────── #
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter, stretch=1)

        # Left: spectrum canvas
        left = QWidget()
        left_l = QVBoxLayout(left)
        left_l.setContentsMargins(0, 0, 0, 0)
        self.fig_spec, self.ax_spec = plt.subplots(figsize=(7, 4))
        self.fig_spec.tight_layout(pad=1.5)
        self.canvas_spec = FigureCanvas(self.fig_spec)
        self.canvas_spec.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.toolbar_spec = NavigationToolbar(self.canvas_spec, left)
        left_l.addWidget(self.toolbar_spec)
        left_l.addWidget(self.canvas_spec)
        splitter.addWidget(left)

        # Right: results + trend sub-tabs
        right       = QWidget()
        right_l     = QVBoxLayout(right)
        right_l.setContentsMargins(4, 0, 0, 0)
        right_tabs  = QTabWidget()
        right_l.addWidget(right_tabs)

        # Results table tab
        tbl_widget = QWidget()
        tbl_l      = QVBoxLayout(tbl_widget)

        self.tbl_results = QTableWidget(0, 7)
        self.tbl_results.setHorizontalHeaderLabels([
            "Channel", "Peak (keV)", "Centroid (keV)",
            "FWHM (keV)", "R%", "χ²/NDF", "Status"])
        self.tbl_results.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch)
        self.tbl_results.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tbl_results.setSelectionBehavior(QAbstractItemView.SelectRows)
        tbl_l.addWidget(self.tbl_results)

        btn_row = QHBoxLayout()
        self.btn_fit_all_ch = QPushButton("⚡  Fit All Channels (same range)")
        self.btn_fit_all_ch.setToolTip(
            "Apply the current fit range to the same peak "
            "across all channels that have calibrated spectra.")
        self.btn_fit_all_ch.setEnabled(False)
        self.btn_fit_all_ch.clicked.connect(self._fit_all_channels)
        self.btn_export = QPushButton("💾  Export")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self._export)
        btn_row.addWidget(self.btn_fit_all_ch)
        btn_row.addWidget(self.btn_export)
        tbl_l.addLayout(btn_row)
        right_tabs.addTab(tbl_widget, "📋 Results Table")

        # Trend plot tab
        trend_widget = QWidget()
        trend_l      = QVBoxLayout(trend_widget)
        trend_ctrl   = QHBoxLayout()
        trend_ctrl.addWidget(QLabel("Peak:"))
        self.cb_trend_peak = QComboBox()
        self.cb_trend_peak.setMinimumWidth(100)
        self.cb_trend_peak.currentIndexChanged.connect(self._update_trend)
        trend_ctrl.addWidget(self.cb_trend_peak)
        trend_ctrl.addWidget(QLabel("Show:"))
        self.cb_trend_type = QComboBox()
        self.cb_trend_type.addItem("FWHM (keV)", "fwhm")
        self.cb_trend_type.addItem("Resolution R%", "res")
        self.cb_trend_type.currentIndexChanged.connect(self._update_trend)
        trend_ctrl.addWidget(self.cb_trend_type)
        trend_ctrl.addStretch()
        trend_l.addLayout(trend_ctrl)

        self.fig_trend, self.ax_trend = plt.subplots(figsize=(6, 3.5))
        self.canvas_trend = FigureCanvas(self.fig_trend)
        self.canvas_trend.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding)
        trend_l.addWidget(self.canvas_trend)
        right_tabs.addTab(trend_widget, "📈 Resolution Trend")

        splitter.setSizes([600, 420])
        splitter.addWidget(right)

        self.btn_save_trend = QPushButton("💾 Save Plot")
        self.btn_save_trend.clicked.connect(self._save_trend_plot)
        trend_ctrl.addWidget(self.btn_save_trend)

    # ------------------------------------------------------------------ #
    # Spectrum plotting + SpanSelector
    # ------------------------------------------------------------------ #

    def _plot_spectrum(self, cal: CalibratedSpectrum, clear_fits=True):
        self.ax_spec.clear()
        if clear_fits:
            self._fit_lines.clear()

        self.ax_spec.step(cal.energy_centers, cal.counts,
                           where="mid", color="#1565c0",
                           linewidth=0.9, label="Calibrated spectrum")
        self.ax_spec.set_xlabel("Energy (keV)", fontsize=9)
        self.ax_spec.set_ylabel("Counts",       fontsize=9)
        self.ax_spec.set_title(
            f"Channel {cal.channel_id}  — drag to select peak fit range",
            fontsize=9)
        self.ax_spec.set_yscale("log")
        self.ax_spec.set_ylim(bottom=0.5)
        self.ax_spec.grid(True, alpha=0.25, linestyle="--")

        # Re-draw any existing fit overlays
        for item in self._fit_lines:
            self.ax_spec.add_line(item) if hasattr(item, 'get_xdata') \
            else self.ax_spec.add_patch(item)

        self.fig_spec.tight_layout(pad=1.5)
        self.canvas_spec.draw()

        # Attach SpanSelector
        self._attach_span_selector()

    def _attach_span_selector(self):
        """Attach a draggable span selector to the spectrum axes."""
        # Must keep reference or it gets garbage collected
        self._span_selector = SpanSelector(
            self.ax_spec,
            self._on_span_selected,
            direction="horizontal",
            useblit=True,
            props=dict(facecolor="#FFF9C4", alpha=0.6, edgecolor="#F9A825"),
            interactive=True,
            drag_from_anywhere=True
        )

    def _on_span_selected(self, xmin: float, xmax: float):
        if abs(xmax - xmin) < 1e-6:
            return
        self._span_lo = xmin
        self._span_hi = xmax
        self.lbl_range.setText(
            f"Selected range: {xmin:.2f} – {xmax:.2f} keV  "
            f"(peak label: {self.sb_peak_label.value():.1f} keV)  "
            f"→ click ⚡ Fit Peak")
        self.btn_fit_peak.setEnabled(True)
        self.btn_fit_all_ch.setEnabled(True)

    # ------------------------------------------------------------------ #
    # Peak fitting
    # ------------------------------------------------------------------ #

    def _fit_selected_peak(self):
        if self._current_cal is None:
            return
        cal    = self._current_cal
        use_bg = self.cb_bg.currentData()
        label  = self.sb_peak_label.value()

        result = self.calc.fit_peak(
            channel_id = cal.channel_id,
            peak_label = label,
            energy_arr = cal.energy_centers,
            counts_arr = cal.counts,
            e_lo       = self._span_lo,
            e_hi       = self._span_hi,
            use_bg     = use_bg)

        self._add_result_row(result)
        if result.success:
            self._overlay_fit(cal, result, use_bg)
            self._update_trend_combo()
            self._update_trend()
            self.btn_export.setEnabled(True)
        else:
            QMessageBox.warning(self, "Fit Failed", result.fail_reason)

    def _overlay_fit(self, cal: CalibratedSpectrum, result, use_bg: bool):
        """Draw Gaussian overlay on the spectrum."""
        e_lo, e_hi = result.fit_range
        E_fine     = np.linspace(e_lo, e_hi, 400)
        if use_bg:
            # Reconstruct params from result for plotting
            mask  = (cal.energy_centers >= e_lo) & \
                    (cal.energy_centers <= e_hi)
            C_bg  = cal.counts[mask]
            A     = result.amplitude
            mu    = result.mu
            sigma = result.sigma
            B     = float(C_bg.min()) if len(C_bg) > 0 else 0.0
            Y     = gaussian_linear_bg(E_fine, A, mu, sigma, B, 0.0)
        else:
            Y = gaussian_only(E_fine, result.amplitude,
                               result.mu, result.sigma)

        line, = self.ax_spec.plot(
            E_fine, Y, "-", color="#c62828", linewidth=1.5,
            label=f"Fit {result.peak_energy:.0f} keV  "
                  f"FWHM={result.fwhm:.2f} keV  R={result.resolution:.1f}%",
            zorder=5)
        # FWHM markers
        fwhm_y = result.amplitude * 0.5
        self.ax_spec.annotate(
            f"FWHM={result.fwhm:.2f} keV\nR={result.resolution:.1f}%",
            xy=(result.mu, fwhm_y),
            xytext=(result.mu + (e_hi - e_lo)*0.1, fwhm_y * 1.5),
            fontsize=7, color="#c62828",
            arrowprops=dict(arrowstyle="->", color="#c62828", lw=0.8))

        self.ax_spec.legend(fontsize=7)
        self.fig_spec.tight_layout(pad=1.5)
        self.canvas_spec.draw()
        self._fit_lines.append(line)

    # ------------------------------------------------------------------ #
    # Fit all channels with same range
    # ------------------------------------------------------------------ #

    def _fit_all_channels(self):
        """
        Apply current fit range + peak label to all channels
        that have calibrated spectra available.
        """
        mw = getattr(self, "_main_window", None)
        if mw is None:
            QMessageBox.warning(self, "Cannot Access Spectra",
                "Please use '📐 Analyse Resolution →' from the "
                "Calibrated Spectrum tab to load channels first.")
            return

        cal_tab   = mw._calib_spectrum_tab
        all_cals  = cal_tab.all_calibrated_spectra()
        if not all_cals:
            QMessageBox.warning(self, "No Calibrated Spectra",
                "Apply calibration to all channels first using "
                "'▶ Apply to All' in the Calibrated Spectrum tab.")
            return

        use_bg = self.cb_bg.currentData()
        label  = self.sb_peak_label.value()
        n_ok   = 0

        for ch_id, cal in sorted(all_cals.items()):
            result = self.calc.fit_peak(
                channel_id = ch_id,
                peak_label = label,
                energy_arr = cal.energy_centers,
                counts_arr = cal.counts,
                e_lo       = self._span_lo,
                e_hi       = self._span_hi,
                use_bg     = use_bg)
            self._add_result_row(result)
            if result.success:
                n_ok += 1

        self._update_trend_combo()
        self._update_trend()
        if n_ok > 0:
            self.btn_export.setEnabled(True)
        QMessageBox.information(
            self, "Done",
            f"Fitted {n_ok}/{len(all_cals)} channels successfully.\n"
            f"Range: {self._span_lo:.1f}–{self._span_hi:.1f} keV  |  "
            f"Peak: {label:.1f} keV")

    # ------------------------------------------------------------------ #
    # Results table
    # ------------------------------------------------------------------ #

    def _add_result_row(self, r):
        # Remove existing row for same (channel, peak) if present
        for row in range(self.tbl_results.rowCount()):
            ch_item = self.tbl_results.item(row, 0)
            pk_item = self.tbl_results.item(row, 1)
            if ch_item and pk_item:
                if (int(ch_item.text()) == r.channel_id and
                        abs(float(pk_item.text()) - r.peak_energy) < 0.1):
                    self.tbl_results.removeRow(row)
                    break

        row = self.tbl_results.rowCount()
        self.tbl_results.insertRow(row)

        def cell(txt, color=None):
            item = QTableWidgetItem(str(txt))
            item.setTextAlignment(Qt.AlignCenter)
            if color:
                item.setBackground(QColor(color))
            return item

        if r.success:
            self.tbl_results.setItem(row, 0, cell(r.channel_id))
            self.tbl_results.setItem(row, 1, cell(f"{r.peak_energy:.1f}"))
            self.tbl_results.setItem(row, 2, cell(f"{r.mu:.3f}"))
            self.tbl_results.setItem(row, 3, cell(f"{r.fwhm:.3f} ± {r.fwhm_err:.3f}"))
            self.tbl_results.setItem(row, 4, cell(f"{r.resolution:.2f} ± {r.resolution_err:.2f}"))
            self.tbl_results.setItem(row, 5, cell(f"{r.chi2_ndf:.3f}"))
            self.tbl_results.setItem(row, 6, cell("✅ OK", "#E8F5E9"))
        else:
            self.tbl_results.setItem(row, 0, cell(r.channel_id))
            self.tbl_results.setItem(row, 1, cell(f"{r.peak_energy:.1f}"))
            for col in range(2, 6):
                self.tbl_results.setItem(row, col, cell("—"))
            self.tbl_results.setItem(row, 6,
                cell(f"❌ {r.fail_reason[:30]}", "#FFEBEE"))

        self.tbl_results.scrollToBottom()

    def _clear_channel_fits(self):
        if self._current_cal is None:
            return
        ch_id = self._current_cal.channel_id
        # Remove from calculator
        to_del = [k for k in self.calc.results if k[0] == ch_id]
        for k in to_del:
            del self.calc.results[k]
        # Remove rows from table
        rows_to_del = []
        for row in range(self.tbl_results.rowCount()):
            item = self.tbl_results.item(row, 0)
            if item and int(item.text()) == ch_id:
                rows_to_del.append(row)
        for row in reversed(rows_to_del):
            self.tbl_results.removeRow(row)
        # Replot without overlays
        self._fit_lines.clear()
        self._plot_spectrum(self._current_cal, clear_fits=True)
        self._update_trend_combo()
        self._update_trend()

    # ------------------------------------------------------------------ #
    # Trend plot
    # ------------------------------------------------------------------ #

    def _update_trend_combo(self):
        peaks = self.calc.peak_labels()
        self.cb_trend_peak.blockSignals(True)
        self.cb_trend_peak.clear()
        for p in peaks:
            self.cb_trend_peak.addItem(f"{p:.1f} keV", p)
        self.cb_trend_peak.blockSignals(False)
        if self.cb_trend_peak.count() > 0:
            self._update_trend()

    def _update_trend(self):
        self.ax_trend.clear()
        if self.cb_trend_peak.count() == 0:
            self.canvas_trend.draw()
            return

        peak  = self.cb_trend_peak.currentData()
        ttype = self.cb_trend_type.currentData()

        if ttype == "fwhm":
            channels, vals, errs = self.calc.fwhm_trend(peak)
            ylabel = "FWHM (keV)"
            color  = "#1565c0"
        else:
            channels, vals, errs = self.calc.resolution_trend(peak)
            ylabel = "Resolution R%"
            color  = "#2e7d32"

        if len(channels) == 0:
            self.ax_trend.text(0.5, 0.5, "No data yet",
                                ha="center", va="center",
                                transform=self.ax_trend.transAxes,
                                color="#9e9e9e")
            self.canvas_trend.draw()
            return

        self.ax_trend.errorbar(
            channels, vals, yerr=errs,
            fmt="o-", markersize=4, linewidth=1.0,
            color=color, ecolor="#90CAF9", capsize=3)
        mean_val = float(np.nanmean(vals))
        self.ax_trend.axhline(mean_val, color="#FF6F00", linestyle="--",
                               linewidth=0.9, alpha=0.8,
                               label=f"Mean = {mean_val:.3f}")
        self.ax_trend.set_xlabel("Channel ID", fontsize=9)
        self.ax_trend.set_ylabel(ylabel,        fontsize=9)
        self.ax_trend.set_title(
            f"{ylabel} vs Channel  |  Peak {peak:.1f} keV", fontsize=9)
        self.ax_trend.legend(fontsize=8)
        self.ax_trend.grid(True, alpha=0.25, linestyle="--")
        self.fig_trend.tight_layout(pad=1.5)
        self.canvas_trend.draw()

    # ------------------------------------------------------------------ #
    # Export
    # ------------------------------------------------------------------ #

    def _export(self):
        if not self.calc.results:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Resolution Results",
            "resolution_results.txt",
            "Text Files (*.txt);;All Files (*)")
        if not path:
            return
        try:
            src = self._current_cal.source if self._current_cal else ""
            self.calc.export(path, source_file=src)
            QMessageBox.information(self, "Exported",
                f"Resolution results saved to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    def _save_trend_plot(self):
        path, selected_filter = QFileDialog.getSaveFileName(
            self, "Save Trend Plot", "resolution_trend.png",
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)")
        if not path:
            return
        try:
            self.fig_trend.savefig(path, dpi=150, bbox_inches="tight")
            QMessageBox.information(self, "Saved", f"Plot saved to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))
