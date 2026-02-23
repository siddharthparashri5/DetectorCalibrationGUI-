
"""
CalibratedSpectrumTab
=====================
Tab widget showing calibrated energy spectra.
"""

from __future__ import annotations
import numpy as np
import os

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
    QFileDialog, QMessageBox, QCheckBox, QSizePolicy,
    QScrollArea, QGridLayout
)
from PyQt5.QtCore import Qt, pyqtSignal

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar
)
import matplotlib.pyplot as plt

from src.calib_spectrum import CalibSpectrumEngine, CalibratedSpectrum


class CalibratedSpectrumTab(QWidget):
    channel_for_resolution = pyqtSignal(int, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.engine      = CalibSpectrumEngine()
        self.loader      = None
        self.fit_results = {}
        self._cal_spectra = {}
        self._build_ui()

    def inject(self, loader, fit_results: dict):
        self.loader      = loader
        self.fit_results = fit_results

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        ctrl = QHBoxLayout()

        src_box = QGroupBox("Calibration Source")
        src_l   = QHBoxLayout(src_box)
        self.btn_from_memory = QPushButton("📋  From Session")
        self.btn_from_memory.clicked.connect(self._load_from_memory)
        self.btn_from_file = QPushButton("📂  Load coeffs.txt")
        self.btn_from_file.clicked.connect(self._load_from_file)
        self.lbl_calib_src = QLabel("No calibration loaded")
        self.lbl_calib_src.setStyleSheet("color:#555; font-size:11px;")
        src_l.addWidget(self.btn_from_memory)
        src_l.addWidget(self.btn_from_file)
        src_l.addWidget(self.lbl_calib_src)
        ctrl.addWidget(src_box)

        range_box = QGroupBox("Energy Range (keV)")
        range_l   = QHBoxLayout(range_box)
        range_l.addWidget(QLabel("Min:"))
        self.sb_emin = QDoubleSpinBox()
        self.sb_emin.setRange(0, 1e6); self.sb_emin.setValue(0)
        self.sb_emin.setSpecialValueText("Auto")
        range_l.addWidget(self.sb_emin)
        range_l.addWidget(QLabel("Max:"))
        self.sb_emax = QDoubleSpinBox()
        self.sb_emax.setRange(0, 1e6); self.sb_emax.setValue(0)
        self.sb_emax.setSpecialValueText("Auto")
        range_l.addWidget(self.sb_emax)
        range_l.addWidget(QLabel("Bins:"))
        self.sb_out_bins = QSpinBox()
        self.sb_out_bins.setRange(64, 16384); self.sb_out_bins.setValue(1024)
        range_l.addWidget(self.sb_out_bins)
        ctrl.addWidget(range_box)

        ch_box = QGroupBox("Channel")
        ch_l   = QHBoxLayout(ch_box)
        self.cb_channel = QComboBox()
        self.cb_channel.setMinimumWidth(120)
        self.cb_channel.currentIndexChanged.connect(self._on_channel_changed)
        self.btn_apply_all = QPushButton("Apply to All")
        self.btn_apply_all.clicked.connect(self._apply_all)
        ch_l.addWidget(self.cb_channel)
        ch_l.addWidget(self.btn_apply_all)
        ctrl.addWidget(ch_box)

        self.btn_send_res = QPushButton("Analyse Resolution")
        self.btn_send_res.clicked.connect(self._send_to_resolution)
        self.btn_send_res.setEnabled(False)
        ctrl.addWidget(self.btn_send_res)
        ctrl.addStretch()
        layout.addLayout(ctrl)

        self.fig, self.ax = plt.subplots(figsize=(9, 4))
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, stretch=1)

        self.lbl_info = QLabel("")
        self.lbl_info.setStyleSheet("font-size: 11px; color: #424242; padding: 4px;")
        layout.addWidget(self.lbl_info)

    def _load_from_memory(self):
        if not self.fit_results:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(self, "No Session Data", "Run calibration fitting first.")
            return
        self.engine.load_from_memory(self.fit_results)
        n = len(self.engine.calib_params)
        self.lbl_calib_src.setText(f"{n} channels calibrated")
        self._populate_channel_combo()

    def _load_from_file(self):
        from PyQt5.QtWidgets import QFileDialog, QMessageBox
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Calibration Coefficients", "", "Text Files (*.txt);;All Files (*)")
        if not path:
            return
        try:
            self.engine.load_from_file(path)
        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e))
            return
        n = len(self.engine.calib_params)
        self.lbl_calib_src.setText(f"{os.path.basename(path)}  ({n} channels)")
        self._populate_channel_combo()

    def _populate_channel_combo(self):
        self.cb_channel.blockSignals(True)
        self.cb_channel.clear()
        for ch_id in self.engine.calibrated_channels():
            self.cb_channel.addItem(f"Channel {ch_id}", ch_id)
        self.cb_channel.blockSignals(False)
        if self.cb_channel.count() > 0:
            self.cb_channel.setCurrentIndex(0)
            self._on_channel_changed(0)

    def _on_channel_changed(self, idx: int):
        if idx < 0 or not self.cb_channel.count():
            return
        ch_id = self.cb_channel.currentData()
        if ch_id is None:
            return
        self._show_channel(ch_id)

    def _show_channel(self, ch_id: int):
        if self.loader is None:
            return
        raw = self.loader.get_spectrum(ch_id)
        if raw is None:
            self.lbl_info.setText(f"No raw spectrum for channel {ch_id}.")
            return
        cal = self.engine.apply(ch_id, raw.bin_centers, raw.counts,
                                 out_bins=self.sb_out_bins.value(),
                                 e_min=self.sb_emin.value(),
                                 e_max=self.sb_emax.value())
        if cal is None:
            self.lbl_info.setText(f"No calibration available for channel {ch_id}.")
            return
        self._cal_spectra[ch_id] = cal
        self._plot_single(cal)
        self.btn_send_res.setEnabled(True)

    def _apply_all(self):
        if self.loader is None:
            return
        self._cal_spectra.clear()
        for ch_id in self.engine.calibrated_channels():
            raw = self.loader.get_spectrum(ch_id)
            if raw is None:
                continue
            cal = self.engine.apply(ch_id, raw.bin_centers, raw.counts,
                                     out_bins=self.sb_out_bins.value(),
                                     e_min=self.sb_emin.value(),
                                     e_max=self.sb_emax.value())
            if cal:
                self._cal_spectra[ch_id] = cal
        ch_id = self.cb_channel.currentData()
        if ch_id and ch_id in self._cal_spectra:
            self._plot_single(self._cal_spectra[ch_id])

    def _plot_single(self, cal):
        self.ax.clear()
        self.ax.step(cal.energy_centers, cal.counts, where="mid",
                      color="#1565c0", linewidth=0.9)
        self.ax.set_xlabel("Energy (keV)", fontsize=10)
        self.ax.set_ylabel("Counts",       fontsize=10)
        self.ax.set_title(
            f"Calibrated Spectrum — Channel {cal.channel_id}  [{cal.model}]",
            fontsize=9)
        self.ax.set_yscale("log")
        self.ax.set_ylim(bottom=0.5)
        self.ax.grid(True, alpha=0.25, linestyle="--")
        self.fig.tight_layout(pad=1.5)
        self.canvas.draw()
        total = int(cal.counts.sum())
        emin  = cal.energy_centers[cal.counts > 0].min() if (cal.counts > 0).any() else 0
        emax  = cal.energy_centers.max()
        self.lbl_info.setText(
            f"Channel {cal.channel_id}  |  Energy: {emin:.1f}–{emax:.1f} keV  |  "
            f"Total counts: {total:,}  |  Model: {cal.model}")

    def _send_to_resolution(self):
        ch_id = self.cb_channel.currentData()
        if ch_id is None or ch_id not in self._cal_spectra:
            return
        self.channel_for_resolution.emit(ch_id, self._cal_spectra[ch_id])

    def get_calibrated_spectrum(self, ch_id: int):
        return self._cal_spectra.get(ch_id)

    def all_calibrated_spectra(self) -> dict:
        return dict(self._cal_spectra)
