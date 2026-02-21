#!/usr/bin/env python3
"""
DetectorCalibGUI — Detector Energy Calibration Tool
Author: Siddharth Parashari
"""

import sys
import os

os.environ.setdefault("ROOT_IGNORE_COMMANDLINE", "1")

from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtCore import Qt
from src.main_window import MainWindow


def check_backend():
    """
    Try to detect a ROOT backend before launching the GUI.
    If neither PyROOT nor uproot is found, show a clear error dialog.
    """
    try:
        from src.root_loader import _detect_backend, _find_root_python_path
        backend = _detect_backend()
        print(f"[DetectorCalibGUI] ROOT backend: {backend}")
        return True
    except RuntimeError as e:
        app = QApplication.instance() or QApplication(sys.argv)
        msg = QMessageBox()
        msg.setWindowTitle("ROOT Backend Not Found")
        msg.setIcon(QMessageBox.Critical)
        msg.setText("<b>No ROOT backend could be found.</b>")
        msg.setInformativeText(
            "DetectorCalibGUI needs one of the following:<br><br>"
            "<b>Option A — PyROOT</b> (you have ROOT 6.36 installed):<br>"
            "Open a terminal and run:<br>"
            "<code>&nbsp;&nbsp;source /path/to/root/bin/thisroot.sh</code><br>"
            "Then relaunch:<br>"
            "<code>&nbsp;&nbsp;python3 main.py</code><br><br>"
            "<b>Option B — uproot</b> (no ROOT install needed):<br>"
            "<code>&nbsp;&nbsp;pip install uproot awkward</code><br>"
            "Then relaunch:<br>"
            "<code>&nbsp;&nbsp;python3 main.py</code><br><br>"
            "On macOS with Homebrew ROOT, try:<br>"
            "<code>&nbsp;&nbsp;source $(brew --prefix root)/bin/thisroot.sh</code>"
        )
        msg.setDetailedText(str(e))
        msg.exec_()
        return False


def main():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    # macOS rendering fix
    if sys.platform == "darwin":
        os.environ.setdefault("QT_MAC_WANTS_LAYER", "1")

    app = QApplication(sys.argv)
    app.setApplicationName("DetectorCalibGUI")
    app.setOrganizationName("NuclearPhysicsLab")
    app.setStyle("Fusion")

    if not check_backend():
        sys.exit(1)

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
