# app.py
#!/usr/bin/env python3
"""Spectrogram Studio Pro — Entry Point"""

import sys
from PySide6.QtWidgets import QApplication

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Spectrogram Studio Pro")
    app.setOrganizationName("SpectrogramStudio")
    app.setStyle("Fusion")
    
    from main_window import MainWindow
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()