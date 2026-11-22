# main_window.py
import os
import sys
import shutil
import tempfile
import time
import soundfile as sf
import numpy as np
import cv2
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFileDialog, QProgressBar, QGroupBox, QComboBox,
    QDoubleSpinBox, QCheckBox, QSpinBox, QLineEdit, QMenuBar, QTextEdit,
    QMessageBox, QTableWidgetItem, QPlainTextEdit, QStackedWidget,
    QScrollArea, QSplitter, QTabWidget, QSizePolicy,
)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QAction, QColor, QBrush, QTextCursor, QImage, QPixmap

from config import (ConfigManager, AUDIO_EXTS, DEFAULT_SETTINGS,
                    SettingsProxy, get_default_load_folder,
                    get_cwd_subfolder, FOLDER_INPUT, FOLDER_OUTPUT,
                    FOLDER_TTS_OUTPUT)
from utils import open_file_location, logger
from palettes import get_all_luts, get_default_palette_dir
from worker import RenderWorker, TTSWorker, TTSRenderWorker
from widgets import AudioTableWidget
from tts_backends import BACKEND_REGISTRY, EdgeTTSBackend, Qwen3Backend


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TTS Spectrogram Studio")
        self.resize(1100, 920)

        if not shutil.which("ffmpeg"):
            QMessageBox.critical(
                self, "Error",
                "FFmpeg not found! Please install FFmpeg and add it to PATH.")
            sys.exit(1)

        # ── Guard flag: prevent saves while loading ──
        self._loading_settings = False

        self.worker = None
        self.tts_worker = None
        self.tts_render_worker = None
        self.config_manager = ConfigManager()

        saved_cfg = self.config_manager.load()
        saved_settings = saved_cfg.get('settings', {})

        self._palette_folder = saved_settings.get('palette_folder', '').strip()
        if not self._palette_folder:
            self._palette_folder = get_default_palette_dir()
        self.available_luts = get_all_luts(self._palette_folder)

        # ── Folder paths (relative to CWD = TTSSpectrograms) ──
        self.input_dir = get_cwd_subfolder(FOLDER_INPUT)
        self.output_target_batch = get_cwd_subfolder(FOLDER_OUTPUT)
        self._tts_output_dir = get_cwd_subfolder(FOLDER_TTS_OUTPUT)
        self.output_target_merge = os.path.join(
            self.output_target_batch, "Merged_Output.mp4")
        self.output_target = self.output_target_batch

        saved_load = saved_settings.get('default_load_folder', '').strip()
        self.default_load_folder = (
            saved_load if saved_load and os.path.isdir(saved_load)
            else self.input_dir)

        saved_last = saved_settings.get('last_input_dir', '').strip()
        self.last_input_dir = (
            saved_last if saved_last and os.path.isdir(saved_last)
            else self.default_load_folder)

        self._preview_path = None
        self.error_count = 0

        # TTS settings dict + proxy
        self.tts_settings = {}
        self.tts_proxy = SettingsProxy(
            self.tts_settings, on_change=self._on_tts_setting_changed)
        self._tts_settings_widget = None
        self._current_tts_backend_key = "edge_tts"

        # ── Build UI, then load, then connect autosave ──
        self.init_ui()
        self.init_menu()
        self.load_settings()          # sets widget values (no saves yet)
        self._init_tts_backends()     # creates TTS widgets (saves blocked)
        self.setup_autosave()         # NOW connect save signals
        self._final_save_after_load() # one consolidated save

        if self.default_load_folder and os.path.isdir(self.default_load_folder):
            self.add_files_from_paths([self.default_load_folder])

    # ================================================================
    # CLOSE EVENT — guarantee settings are persisted on exit
    # ================================================================
    def closeEvent(self, event):
        """Save all settings before the window closes."""
        self.save_settings()
        # Wait briefly for any running workers
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(2000)
        if self.tts_render_worker and self.tts_render_worker.isRunning():
            self.tts_render_worker.cancel()
            self.tts_render_worker.wait(2000)
        event.accept()

    # ================================================================
    # UI INITIALIZATION
    # ================================================================
    def init_ui(self):
        self.setStyleSheet("""
            QWidget { background-color: #1e1e1e; color: #e0e0e0;
                      font-family: 'Segoe UI', Arial; font-size: 13px; }
            QGroupBox { border: 1px solid #3a3a3a; border-radius: 6px;
                        margin-top: 18px; font-weight: bold; padding-top: 18px; }
            QGroupBox::title { subcontrol-origin: margin; left: 12px;
                               padding: 0 6px; }
            QPushButton { background-color: #3c3c3c; border: 1px solid #555;
                          border-radius: 4px; padding: 7px 14px; }
            QPushButton:hover { background-color: #4d4d4d; }
            QPushButton:disabled { background-color: #2a2a2a; color: #666; }
            QProgressBar::chunk { background-color: #2ecc71;
                                  border-radius: 3px; }
            QComboBox { padding: 4px; border: 1px solid #555;
                        border-radius: 3px; }
            QSpinBox, QDoubleSpinBox, QLineEdit { padding: 4px;
                                                   border: 1px solid #555;
                                                   border-radius: 3px; }
            QTabWidget::pane { border: 1px solid #3a3a3a;
                               border-radius: 4px; top: -1px; }
            QTabBar::tab { background: #2d2d2d; padding: 10px 22px;
                           border: 1px solid #3a3a3a;
                           border-bottom: none;
                           border-top-left-radius: 6px;
                           border-top-right-radius: 6px;
                           margin-right: 2px; }
            QTabBar::tab:selected { background: #3c3c3c;
                                    border-bottom: 2px solid #2ecc71; }
            QTabBar::tab:hover { background: #383838; }
            QScrollArea { border: none; background: transparent; }
        """)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(6)

        # ── Tab Widget ──
        self.tab_widget = QTabWidget()
        self.tab_widget.addTab(
            self._create_files_tab(), "  Files & Output  ")
        self.tab_widget.addTab(
            self._create_spectrogram_tab(), "  Spectrogram  ")
        self.tab_widget.addTab(
            self._create_visualizer_tab(), "  Visualizer  ")
        self.tab_widget.addTab(
            self._create_tts_tab(), "  TTS  ")
        main_layout.addWidget(self.tab_widget, 1)

        # ── Bottom: Controls ──
        ctrl = QHBoxLayout()
        self.btn_start = QPushButton("▶  Start Render")
        self.btn_start.setStyleSheet(
            "background-color: #2ecc71; color: white; "
            "font-weight: bold; padding: 12px; font-size: 14px;")
        self.btn_start.clicked.connect(self.start_processing)

        self.btn_preview = QPushButton("👁  Preview (15s)")
        self.btn_preview.setStyleSheet(
            "background-color: #f1c40f; color: black; "
            "font-weight: bold; padding: 12px; font-size: 14px;")
        self.btn_preview.clicked.connect(self.start_preview)

        self.btn_cancel = QPushButton("✖  Cancel")
        self.btn_cancel.setStyleSheet(
            "background-color: #e74c3c; color: white; "
            "padding: 12px; font-size: 14px;")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self.cancel_processing)

        ctrl.addWidget(self.btn_start, 1)
        ctrl.addWidget(self.btn_preview, 1)
        ctrl.addWidget(self.btn_cancel, 1)
        main_layout.addLayout(ctrl)

        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        main_layout.addWidget(self.progress_bar)

        self.log_widget = QTextEdit()
        self.log_widget.setReadOnly(True)
        self.log_widget.setMaximumHeight(90)
        main_layout.addWidget(self.log_widget)

        self.update_palette_preview()
        self.toggle_auto_freq()
        self.toggle_auto_highpass()

    # ── Tab 1: Files & Output ─────────────────────────────────────
    def _create_files_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(8)

        g1 = QGroupBox("Source Files")
        l1 = QVBoxLayout(g1)

        r_def = QHBoxLayout()
        r_def.addWidget(QLabel("Default Folder:"))
        self.lbl_default_folder = QLabel(self.default_load_folder)
        self.lbl_default_folder.setStyleSheet(
            "color: #5dade2; font-weight: bold;")
        self.lbl_default_folder.setWordWrap(True)
        btn_change = QPushButton("Change...")
        btn_change.setFixedWidth(80)
        btn_change.clicked.connect(self.change_default_load_folder)
        r_def.addWidget(self.lbl_default_folder, 1)
        r_def.addWidget(btn_change)
        l1.addLayout(r_def)

        ctrl = QHBoxLayout()
        self.btn_add_files = QPushButton("Add Files")
        self.btn_add_files.clicked.connect(self.add_files)
        self.btn_load_folder = QPushButton("Load Folder")
        self.btn_load_folder.clicked.connect(self.load_all_from_folder)
        self.chk_recursive = QCheckBox("Recursive")
        self.chk_recursive.setChecked(True)
        ctrl.addWidget(self.btn_add_files)
        ctrl.addWidget(self.btn_load_folder)
        ctrl.addWidget(self.chk_recursive)
        ctrl.addStretch()
        l1.addLayout(ctrl)

        self.table_widget = AudioTableWidget()
        self.table_widget.files_dropped.connect(self.add_files_from_paths)
        self.table_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        l1.addWidget(self.table_widget, 1)
        layout.addWidget(g1, 3)

        g2 = QGroupBox("Output Settings")
        l2 = QVBoxLayout(g2)

        r_out = QHBoxLayout()
        self.btn_output = QPushButton("Save Location...")
        self.btn_output.clicked.connect(self.select_output)
        self.lbl_output = QLabel(self.output_target)
        self.lbl_output.setStyleSheet(
            "font-weight: bold; color: #5dade2;")
        self.lbl_output.setWordWrap(True)
        r_out.addWidget(self.btn_output)
        r_out.addWidget(self.lbl_output, 1)
        l2.addLayout(r_out)

        r_v1 = QHBoxLayout()
        r_v1.addWidget(QLabel("Resolution:"))
        self.combo_res = QComboBox()
        self.combo_res.addItems(['720p', '1080p', '1440p', '2160p'])
        self.combo_res.setCurrentIndex(1)
        r_v1.addWidget(self.combo_res)
        r_v1.addSpacing(12)
        r_v1.addWidget(QLabel("FPS:"))
        self.spin_fps = QSpinBox()
        self.spin_fps.setRange(10, 60)
        self.spin_fps.setValue(30)
        r_v1.addWidget(self.spin_fps)
        r_v1.addSpacing(12)
        self.chk_gpu = QCheckBox("GPU (NVENC)")
        self.chk_gpu.setChecked(True)
        r_v1.addWidget(self.chk_gpu)
        r_v1.addStretch()
        l2.addLayout(r_v1)

        r_v2 = QHBoxLayout()
        self.chk_merge = QCheckBox("Merge into One Video")
        self.chk_merge.stateChanged.connect(self.toggle_output_mode)
        r_v2.addWidget(self.chk_merge)
        r_v2.addSpacing(12)
        r_v2.addWidget(QLabel("Filename Suffix:"))
        self.txt_suffix = QLineEdit()
        self.txt_suffix.setMaximumWidth(150)
        self.txt_suffix.setPlaceholderText("_spectrogram")
        r_v2.addWidget(self.txt_suffix)
        r_v2.addStretch()
        l2.addLayout(r_v2)
        layout.addWidget(g2, 1)

        return page

    # ── Tab 2: Spectrogram ────────────────────────────────────────
    def _create_spectrogram_tab(self) -> QWidget:
        inner = QWidget()
        inner_layout = QVBoxLayout(inner)
        inner_layout.setSpacing(8)

        g3 = QGroupBox("Spectrogram Engine")
        l3 = QVBoxLayout(g3)

        r_algo = QHBoxLayout()
        r_algo.addWidget(QLabel("Algorithm:"))
        self.combo_spec_type = QComboBox()
        self.combo_spec_type.addItems(
            ['Standard (STFT)', 'Mel-Scaled', 'CQT'])
        r_algo.addWidget(self.combo_spec_type)
        r_algo.addSpacing(16)
        r_algo.addWidget(QLabel("Display:"))
        self.combo_vis_mode = QComboBox()
        self.combo_vis_mode.addItems(['Standard', 'Classic', 'Radial'])
        r_algo.addWidget(self.combo_vis_mode)
        r_algo.addStretch()
        l3.addLayout(r_algo)

        r_c = QHBoxLayout()
        r_c.addWidget(QLabel("Palette:"))
        self.combo_cmap = QComboBox()
        self.combo_cmap.addItems(sorted(self.available_luts.keys()))
        self.combo_cmap.currentIndexChanged.connect(
            self.update_palette_preview)
        r_c.addWidget(self.combo_cmap)
        self.palette_preview = QLabel()
        self.palette_preview.setFixedHeight(22)
        self.palette_preview.setMinimumWidth(180)
        self.palette_preview.setStyleSheet(
            "border: 1px solid #555; border-radius: 3px;")
        r_c.addWidget(self.palette_preview, 1)
        r_c.addSpacing(12)
        r_c.addWidget(QLabel("Color Cycle:"))
        self.combo_color_mode = QComboBox()
        self.combo_color_mode.addItems(['Static', 'Rainbow Cycle'])
        r_c.addWidget(self.combo_color_mode)
        r_c.addStretch()
        l3.addLayout(r_c)

        r_gran = QHBoxLayout()
        r_gran.addWidget(QLabel("FFT:"))
        self.combo_nfft = QComboBox()
        self.combo_nfft.addItems(['2048', '4096', '8192', '16384'])
        self.combo_nfft.setCurrentIndex(1)
        r_gran.addWidget(self.combo_nfft)
        r_gran.addSpacing(12)
        r_gran.addWidget(QLabel("Hop:"))
        self.spin_hop = QSpinBox()
        self.spin_hop.setRange(64, 4096)
        self.spin_hop.setValue(512)
        r_gran.addWidget(self.spin_hop)
        r_gran.addSpacing(12)
        r_gran.addWidget(QLabel("Interp:"))
        self.combo_interp = QComboBox()
        self.combo_interp.addItems(['Linear', 'Cubic', 'Lanczos'])
        self.combo_interp.setCurrentIndex(2)
        r_gran.addWidget(self.combo_interp)
        r_gran.addSpacing(12)
        self.chk_sharpen = QCheckBox("Sharpen")
        self.chk_sharpen.setChecked(True)
        r_gran.addWidget(self.chk_sharpen)
        r_gran.addStretch()
        l3.addLayout(r_gran)

        r_freq = QHBoxLayout()
        self.chk_auto_freq = QCheckBox("Auto Hz")
        self.chk_auto_freq.setChecked(True)
        self.chk_auto_freq.stateChanged.connect(self.toggle_auto_freq)
        r_freq.addWidget(self.chk_auto_freq)
        r_freq.addSpacing(8)
        r_freq.addWidget(QLabel("Min:"))
        self.spin_freq_min = QSpinBox()
        self.spin_freq_min.setRange(1, 20000)
        self.spin_freq_min.setValue(20)
        r_freq.addWidget(self.spin_freq_min)
        r_freq.addSpacing(8)
        r_freq.addWidget(QLabel("Max:"))
        self.spin_freq_max = QSpinBox()
        self.spin_freq_max.setRange(20, 22000)
        self.spin_freq_max.setValue(20000)
        r_freq.addWidget(self.spin_freq_max)
        r_freq.addStretch()
        l3.addLayout(r_freq)

        r_fx = QHBoxLayout()
        self.chk_overlay = QCheckBox("Time Overlay")
        self.chk_overlay.setChecked(True)
        r_fx.addWidget(self.chk_overlay)
        r_fx.addSpacing(12)
        r_fx.addWidget(QLabel("Window:"))
        self.spin_window = QDoubleSpinBox()
        self.spin_window.setRange(1, 60)
        self.spin_window.setValue(5.0)
        r_fx.addWidget(self.spin_window)
        r_fx.addSpacing(12)
        r_fx.addWidget(QLabel("Dyn:"))
        self.spin_dyn_range = QSpinBox()
        self.spin_dyn_range.setRange(20, 150)
        self.spin_dyn_range.setValue(80)
        r_fx.addWidget(self.spin_dyn_range)
        r_fx.addSpacing(12)
        r_fx.addWidget(QLabel("Gamma:"))
        self.spin_gamma = QDoubleSpinBox()
        self.spin_gamma.setRange(0.1, 5.0)
        self.spin_gamma.setValue(1.0)
        self.spin_gamma.setSingleStep(0.1)
        r_fx.addWidget(self.spin_gamma)
        r_fx.addStretch()
        l3.addLayout(r_fx)
        inner_layout.addWidget(g3)

        g4 = QGroupBox("Audio Processing")
        r_audio = QHBoxLayout(g4)

        self.chk_normalize = QCheckBox("Normalize")
        self.chk_normalize.setChecked(True)
        r_audio.addWidget(self.chk_normalize)
        r_audio.addSpacing(10)

        self.chk_auto_hp = QCheckBox("Auto Clean Sub-Bass")
        self.chk_auto_hp.setChecked(True)
        self.chk_auto_hp.setToolTip(
            "Automatically detects and removes inaudible sub-bass rumble.")
        self.chk_auto_hp.stateChanged.connect(self.toggle_auto_highpass)
        r_audio.addWidget(self.chk_auto_hp)
        r_audio.addSpacing(10)

        r_audio.addWidget(QLabel("High-Pass:"))
        self.spin_highpass = QSpinBox()
        self.spin_highpass.setRange(0, 200)
        self.spin_highpass.setValue(20)
        self.spin_highpass.setSuffix(" Hz")
        r_audio.addWidget(self.spin_highpass)
        r_audio.addSpacing(10)

        r_audio.addWidget(QLabel("Bass:"))
        self.spin_bass = QDoubleSpinBox()
        self.spin_bass.setRange(-12, 12)
        self.spin_bass.setSuffix(" dB")
        r_audio.addWidget(self.spin_bass)
        r_audio.addSpacing(10)

        r_audio.addWidget(QLabel("Treble:"))
        self.spin_treble = QDoubleSpinBox()
        self.spin_treble.setRange(-12, 12)
        self.spin_treble.setSuffix(" dB")
        r_audio.addWidget(self.spin_treble)
        r_audio.addStretch()
        inner_layout.addWidget(g4)

        inner_layout.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidget(inner)
        return scroll

    # ── Tab 3: Visualizer ─────────────────────────────────────────
    def _create_visualizer_tab(self) -> QWidget:
        inner = QWidget()
        inner_layout = QVBoxLayout(inner)
        inner_layout.setSpacing(8)

        g5 = QGroupBox("Center Visualizer (Spectrum Analyzer)")
        l5 = QVBoxLayout(g5)

        r_vis_en = QHBoxLayout()
        self.chk_center_line = QCheckBox("Enable Visualizer")
        self.chk_center_line.setChecked(True)
        r_vis_en.addWidget(self.chk_center_line)
        r_vis_en.addSpacing(12)

        self.chk_draw_center_line = QCheckBox("Draw Center Line")
        self.chk_draw_center_line.setChecked(True)
        r_vis_en.addWidget(self.chk_draw_center_line)
        r_vis_en.addSpacing(12)

        r_vis_en.addWidget(QLabel("Position:"))
        self.combo_analyzer_pos = QComboBox()
        self.combo_analyzer_pos.addItems(['Center', 'Edges'])
        r_vis_en.addWidget(self.combo_analyzer_pos)
        r_vis_en.addSpacing(12)

        r_vis_en.addWidget(QLabel("Color Style:"))
        self.combo_analyzer_color = QComboBox()
        self.combo_analyzer_color.addItems(
            ['Match Video', 'Neon Gradient', 'Sunset Gradient',
             'White', 'Rainbow'])
        r_vis_en.addWidget(self.combo_analyzer_color)
        r_vis_en.addStretch()
        l5.addLayout(r_vis_en)

        r_vp = QHBoxLayout()
        r_vp.addWidget(QLabel("Bars:"))
        self.spin_bar_count = QSpinBox()
        self.spin_bar_count.setRange(2, 256)
        self.spin_bar_count.setValue(64)
        r_vp.addWidget(self.spin_bar_count)
        r_vp.addSpacing(12)
        r_vp.addWidget(QLabel("Base Width:"))
        self.spin_line_width = QSpinBox()
        self.spin_line_width.setRange(1, 200)
        self.spin_line_width.setValue(6)
        r_vp.addWidget(self.spin_line_width)
        r_vp.addSpacing(12)
        r_vp.addWidget(QLabel("Sensitivity:"))
        self.spin_line_sensitivity = QDoubleSpinBox()
        self.spin_line_sensitivity.setRange(0, 2000)
        self.spin_line_sensitivity.setValue(800)
        self.spin_line_sensitivity.setSingleStep(50)
        r_vp.addWidget(self.spin_line_sensitivity)
        r_vp.addStretch()
        l5.addLayout(r_vp)

        r_vp2 = QHBoxLayout()
        r_vp2.addWidget(QLabel("Glow:"))
        self.spin_line_glow = QDoubleSpinBox()
        self.spin_line_glow.setRange(0, 50)
        self.spin_line_glow.setValue(15)
        self.spin_line_glow.setSingleStep(1)
        r_vp2.addWidget(self.spin_line_glow)
        r_vp2.addSpacing(12)
        r_vp2.addWidget(QLabel("Bloom:"))
        self.spin_line_bloom = QDoubleSpinBox()
        self.spin_line_bloom.setRange(0, 50)
        self.spin_line_bloom.setValue(8)
        self.spin_line_bloom.setSingleStep(1)
        r_vp2.addWidget(self.spin_line_bloom)
        r_vp2.addStretch()
        l5.addLayout(r_vp2)

        inner_layout.addWidget(g5)
        inner_layout.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidget(inner)
        return scroll

    # ── Tab 4: TTS ────────────────────────────────────────────────
    def _create_tts_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(8)

        g6 = QGroupBox("Text-to-Speech")
        l6 = QVBoxLayout(g6)

        r_tts_top = QHBoxLayout()
        r_tts_top.addWidget(QLabel("TTS Engine:"))
        self.combo_tts_backend = QComboBox()
        self._populate_tts_combo()
        self.combo_tts_backend.currentIndexChanged.connect(
            self._on_tts_backend_changed)
        r_tts_top.addWidget(self.combo_tts_backend, 1)
        r_tts_top.addStretch()
        l6.addLayout(r_tts_top)

        self.txt_tts_input = QPlainTextEdit()
        self.txt_tts_input.setPlaceholderText(
            "Enter text to convert to speech…")
        self.txt_tts_input.setMinimumHeight(100)
        l6.addWidget(self.txt_tts_input, 1)

        # ── Action Buttons Row 1: Speech Only ──
        btn_row1 = QHBoxLayout()
        self.btn_tts_generate = QPushButton("▶  Generate Speech")
        self.btn_tts_generate.setStyleSheet(
            "background-color: #3498db; color: white; "
            "font-weight: bold; padding: 10px;")
        self.btn_tts_generate.clicked.connect(self._on_tts_generate)
        btn_row1.addWidget(self.btn_tts_generate)

        self.btn_tts_to_table = QPushButton("Add to Source Files")
        self.btn_tts_to_table.setStyleSheet("padding: 10px;")
        self.btn_tts_to_table.clicked.connect(self._on_tts_add_to_table)
        btn_row1.addWidget(self.btn_tts_to_table)

        self.lbl_tts_status = QLabel("Ready")
        self.lbl_tts_status.setStyleSheet(
            "color: #5dade2; font-size: 12px; padding: 6px;")
        btn_row1.addWidget(self.lbl_tts_status)
        btn_row1.addStretch()
        l6.addLayout(btn_row1)

        # ── Action Buttons Row 2: Combined TTS → MP4 Pipeline ──
        btn_row2 = QHBoxLayout()
        self.btn_tts_to_mp4 = QPushButton(
            "🎬  TTS → Spectrogram MP4  (One-Click Pipeline)")
        self.btn_tts_to_mp4.setStyleSheet(
            "background-color: #8e44ad; color: white; "
            "font-weight: bold; padding: 14px; font-size: 14px; "
            "border: 2px solid #a569bd;")
        self.btn_tts_to_mp4.setToolTip(
            "Generate speech from text, then immediately render a "
            "spectrogram video — all in one step.")
        self.btn_tts_to_mp4.clicked.connect(self._on_tts_to_mp4)
        btn_row2.addWidget(self.btn_tts_to_mp4, 1)
        l6.addLayout(btn_row2)

        self.tts_settings_stack = QStackedWidget()
        self.tts_settings_stack.setMinimumHeight(120)
        l6.addWidget(self.tts_settings_stack, 1)

        layout.addWidget(g6)
        return page

    # ================================================================
    # MENU
    # ================================================================
    def init_menu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        file_menu.addAction("Reset Settings", self.reset_settings)
        file_menu.addAction("Set Palette Folder...",
                            self.change_palette_folder)
        file_menu.addAction(
            "Open Palette Folder",
            lambda: open_file_location(self._palette_folder))
        file_menu.addSeparator()
        file_menu.addAction(
            "Open Input Folder",
            lambda: open_file_location(self.input_dir))
        file_menu.addAction(
            "Open Output Folder",
            lambda: open_file_location(self.output_target))
        file_menu.addAction(
            "Open TTS Output Folder",
            lambda: open_file_location(self._tts_output_dir))
        file_menu.addSeparator()
        file_menu.addAction("Exit", self.close)

    # ================================================================
    # TTS BACKEND MANAGEMENT
    # ================================================================
    def _populate_tts_combo(self):
        self.combo_tts_backend.clear()
        for key, info in BACKEND_REGISTRY.items():
            self.combo_tts_backend.addItem(info['name'], key)

    def _init_tts_backends(self):
        while self.tts_settings_stack.count() > 0:
            widget = self.tts_settings_stack.widget(0)
            self.tts_settings_stack.removeWidget(widget)
            widget.deleteLater()

        # Block saves while constructing TTS widgets
        was_loading = self._loading_settings
        self._loading_settings = True

        for key, info in BACKEND_REGISTRY.items():
            backend_cls = info['class']
            try:
                mode = None
                if hasattr(backend_cls, 'QWEN3_MODES'):
                    mode = "voice_custom"
                widget = backend_cls.get_settings_widget(
                    mode, self.tts_proxy, parent=self)
            except Exception as e:
                widget = QLabel(f"Error loading settings: {e}")
            self.tts_settings_stack.addWidget(widget)

        # Set saved backend — block signals to avoid premature save
        self.combo_tts_backend.blockSignals(True)
        saved_backend = self.tts_settings.get(
            'tts_backend',
            DEFAULT_SETTINGS.get('tts_backend', 'edge_tts'))
        idx = self.combo_tts_backend.findData(saved_backend)
        if idx >= 0:
            self.combo_tts_backend.setCurrentIndex(idx)
            keys = list(BACKEND_REGISTRY.keys())
            if saved_backend in keys:
                self.tts_settings_stack.setCurrentIndex(
                    keys.index(saved_backend))
        self.combo_tts_backend.blockSignals(False)

        self._loading_settings = was_loading

    def _on_tts_backend_changed(self, index):
        key = self.combo_tts_backend.currentData()
        if key is None:
            return
        self._current_tts_backend_key = key
        self.tts_proxy.set('tts_backend', key)

        keys = list(BACKEND_REGISTRY.keys())
        if key in keys:
            self.tts_settings_stack.setCurrentIndex(keys.index(key))
        self.log_message(f"TTS backend: {BACKEND_REGISTRY[key]['name']}")

    def _on_tts_setting_changed(self, key, value):
        self.save_settings()

    def _create_tts_backend(self):
        key = self._current_tts_backend_key
        info = BACKEND_REGISTRY.get(key)
        if info is None:
            return None
        backend_cls = info['class']
        try:
            return backend_cls(self.tts_settings)
        except RuntimeError as e:
            QMessageBox.warning(self, "TTS Error", str(e))
            return None

    # ── Generate Speech Only ──────────────────────────────────────
    def _on_tts_generate(self):
        text = self.txt_tts_input.toPlainText().strip()
        if not text:
            QMessageBox.warning(
                self, "No Text",
                "Please enter text to convert to speech.")
            return

        backend = self._create_tts_backend()
        if backend is None:
            return

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(
            self._tts_output_dir,
            f"tts_{self._current_tts_backend_key}_{timestamp}.wav")

        self._set_busy(True)
        self.btn_tts_generate.setEnabled(False)
        self.btn_tts_to_mp4.setEnabled(False)
        self.lbl_tts_status.setText("Generating speech…")
        self.lbl_tts_status.setStyleSheet(
            "color: #f1c40f; font-size: 12px; padding: 6px;")

        self.tts_worker = TTSWorker(backend, text, output_path)
        self.tts_worker.log.connect(
            lambda m: self.log_message(m, "INFO"))
        self.tts_worker.error.connect(self._on_tts_error)
        self.tts_worker.tts_finished.connect(self._on_tts_finished)
        self.tts_worker.start()

    def _on_tts_error(self, msg):
        self._set_busy(False)
        self.btn_tts_generate.setEnabled(True)
        self.btn_tts_to_mp4.setEnabled(True)
        self.lbl_tts_status.setText("Error")
        self.lbl_tts_status.setStyleSheet(
            "color: #e74c3c; font-size: 12px; padding: 6px;")
        self.log_message(f"TTS Error: {msg}", "ERROR")
        QMessageBox.warning(self, "TTS Error", msg)
        self.tts_worker = None

    def _on_tts_finished(self, output_path):
        self._set_busy(False)
        self.btn_tts_generate.setEnabled(True)
        self.btn_tts_to_mp4.setEnabled(True)
        self.lbl_tts_status.setText("Done!")
        self.lbl_tts_status.setStyleSheet(
            "color: #2ecc71; font-size: 12px; padding: 6px;")
        self.log_message(f"TTS output: {output_path}", "SUCCESS")
        open_file_location(output_path)
        self.tts_worker = None

    def _on_tts_add_to_table(self):
        if not os.path.isdir(self._tts_output_dir):
            return
        wav_files = sorted([
            os.path.join(self._tts_output_dir, f)
            for f in os.listdir(self._tts_output_dir)
            if f.lower().endswith('.wav')
        ], key=lambda p: os.path.getmtime(p), reverse=True)
        if wav_files:
            self.add_files_from_paths(wav_files[:10])
        else:
            self.log_message("No TTS output files found.", "WARN")

    # ── Combined TTS → MP4 Pipeline ───────────────────────────────
    def _on_tts_to_mp4(self):
        text = self.txt_tts_input.toPlainText().strip()
        if not text:
            QMessageBox.warning(
                self, "No Text",
                "Please enter text to generate speech and video from.")
            return

        backend = self._create_tts_backend()
        if backend is None:
            return

        self.save_settings()

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_name = f"tts_{self._current_tts_backend_key}_{timestamp}"

        audio_path = os.path.join(self._tts_output_dir, f"{base_name}.wav")

        suffix = self.txt_suffix.text().strip()
        video_name = base_name
        if suffix:
            video_name = f"{base_name}{suffix}"
        video_path = os.path.join(
            self.output_target_batch, f"{video_name}.mp4")

        counter = 1
        while os.path.exists(video_path):
            video_path = os.path.join(
                self.output_target_batch,
                f"{video_name}_v{counter}.mp4")
            counter += 1

        render_settings = self.get_common_settings()

        self._set_busy(True)
        self.btn_tts_generate.setEnabled(False)
        self.btn_tts_to_mp4.setEnabled(False)
        self.btn_start.setEnabled(False)
        self.btn_preview.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.progress_bar.setValue(0)

        self.lbl_tts_status.setText("Pipeline: TTS → MP4…")
        self.lbl_tts_status.setStyleSheet(
            "color: #a569bd; font-size: 12px; padding: 6px;")

        self.log_message(
            f"Starting TTS → MP4 pipeline for "
            f"{len(text)} chars of text…", "INFO")

        self.tts_render_worker = TTSRenderWorker(
            backend, text, audio_path, video_path, render_settings)

        self.tts_render_worker.progress.connect(self.progress_bar.setValue)
        self.tts_render_worker.status.connect(
            lambda s: self.lbl_tts_status.setText(s))
        self.tts_render_worker.log.connect(
            lambda m: self.log_message(m, "INFO"))
        self.tts_render_worker.error.connect(self._on_tts_render_error)
        self.tts_render_worker.tts_render_finished.connect(
            self._on_tts_render_finished)

        self.tts_render_worker.start()

    def _on_tts_render_error(self, msg):
        self._set_busy(False)
        self.btn_tts_generate.setEnabled(True)
        self.btn_tts_to_mp4.setEnabled(True)
        self.btn_start.setEnabled(True)
        self.btn_preview.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.progress_bar.setValue(0)
        self.lbl_tts_status.setText("Pipeline Error")
        self.lbl_tts_status.setStyleSheet(
            "color: #e74c3c; font-size: 12px; padding: 6px;")
        self.log_message(f"Pipeline Error: {msg}", "ERROR")
        QMessageBox.warning(self, "TTS → MP4 Error", msg)
        self.tts_render_worker = None

    def _on_tts_render_finished(self, audio_path, video_path):
        self._set_busy(False)
        self.btn_tts_generate.setEnabled(True)
        self.btn_tts_to_mp4.setEnabled(True)
        self.btn_start.setEnabled(True)
        self.btn_preview.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.progress_bar.setValue(100)
        self.lbl_tts_status.setText("Pipeline Complete ✓")
        self.lbl_tts_status.setStyleSheet(
            "color: #2ecc71; font-size: 12px; padding: 6px;")

        self.log_message(
            f"Pipeline complete!\n"
            f"  Audio → {audio_path}\n"
            f"  Video → {video_path}", "SUCCESS")

        if os.path.exists(audio_path):
            self.add_files_from_paths([audio_path])

        if os.path.exists(video_path):
            open_file_location(video_path)

        self.tts_render_worker = None

    # ================================================================
    # UI HELPERS
    # ================================================================
    def _set_busy(self, busy: bool):
        self.btn_add_files.setEnabled(not busy)
        self.btn_load_folder.setEnabled(not busy)

    def toggle_auto_freq(self):
        is_auto = self.chk_auto_freq.isChecked()
        self.spin_freq_min.setEnabled(not is_auto)
        self.spin_freq_max.setEnabled(not is_auto)

    def toggle_auto_highpass(self):
        is_auto = self.chk_auto_hp.isChecked()
        self.spin_highpass.setEnabled(not is_auto)

    def update_palette_preview(self):
        name = self.combo_cmap.currentText()
        lut = self.available_luts.get(name)
        if lut is None:
            self.palette_preview.setText("(no palette)")
            return
        try:
            grad = np.arange(256, dtype=np.uint8).reshape(1, 256)
            colored_rgb = cv2.cvtColor(
                cv2.applyColorMap(grad, lut), cv2.COLOR_BGR2RGB)
            display = cv2.resize(
                colored_rgb, (256, 20), interpolation=cv2.INTER_LINEAR)
            h, w, ch = display.shape
            self._preview_data = display.copy()
            qimg = QImage(
                self._preview_data.data, w, h, ch * w,
                QImage.Format_RGB888)
            self.palette_preview.setPixmap(
                QPixmap.fromImage(qimg).scaled(
                    max(1, self.palette_preview.width()), 20,
                    Qt.IgnoreAspectRatio, Qt.SmoothTransformation))
        except Exception:
            self.palette_preview.setText("(error)")

    def change_palette_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Palette Folder", self._palette_folder)
        if folder:
            self._palette_folder = folder
            self.available_luts = get_all_luts(folder)
            prev_text = self.combo_cmap.currentText()
            self.combo_cmap.blockSignals(True)
            self.combo_cmap.clear()
            self.combo_cmap.addItems(sorted(self.available_luts.keys()))
            idx = self.combo_cmap.findText(prev_text)
            self.combo_cmap.setCurrentIndex(idx if idx >= 0 else 0)
            self.combo_cmap.blockSignals(False)
            self.update_palette_preview()
            self.save_settings()
            self.log_message(f"Palette folder set to: {folder}", "SUCCESS")

    def change_default_load_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Set Default Load Folder", self.default_load_folder)
        if folder:
            self.default_load_folder = folder
            self.last_input_dir = folder
            self.lbl_default_folder.setText(folder)
            self.save_settings()
            self.table_widget.setRowCount(0)
            self.log_message(f"Auto-loading from: {folder}", "SUCCESS")
            self.add_files_from_paths([folder])

    def setup_autosave(self):
        """Connect all UI widget signals to save_settings().
        Called AFTER load_settings() so loading doesn't trigger spurious saves.
        """
        widgets = [
            self.combo_res, self.spin_fps, self.chk_gpu, self.chk_merge,
            self.txt_suffix, self.combo_spec_type, self.combo_vis_mode,
            self.combo_cmap, self.combo_color_mode, self.combo_nfft,
            self.spin_hop, self.combo_interp, self.chk_sharpen,
            self.chk_overlay, self.spin_window, self.chk_auto_freq,
            self.spin_freq_min, self.spin_freq_max, self.spin_dyn_range,
            self.spin_gamma, self.chk_normalize, self.chk_auto_hp,
            self.spin_highpass, self.spin_bass, self.spin_treble,
            self.chk_center_line, self.combo_analyzer_color,
            self.spin_line_width, self.spin_line_sensitivity,
            self.spin_line_glow, self.spin_line_bloom,
            self.spin_bar_count, self.combo_analyzer_pos,
            self.chk_draw_center_line,
        ]
        for w in widgets:
            try:
                if isinstance(w, (QSpinBox, QDoubleSpinBox)):
                    w.valueChanged.connect(self._delayed_save)
                elif isinstance(w, QComboBox):
                    w.currentIndexChanged.connect(self._delayed_save)
                elif isinstance(w, QCheckBox):
                    w.stateChanged.connect(self._delayed_save)
                elif isinstance(w, QLineEdit):
                    w.textChanged.connect(self._delayed_save)
            except Exception:
                pass

    # ── Debounced save: coalesce rapid changes into one write ─────
    def _delayed_save(self, *_args):
        """Coalesce rapid widget changes into a single disk write
        (50 ms debounce)."""
        if self._loading_settings:
            return
        if not hasattr(self, '_save_timer'):
            self._save_timer = QTimer(self)
            self._save_timer.setSingleShot(True)
            self._save_timer.setInterval(50)
            self._save_timer.timeout.connect(self.save_settings)
        self._save_timer.start()

    def _final_save_after_load(self):
        """One consolidated save after all loading is done."""
        self._loading_settings = False
        self.save_settings()

    def log_message(self, text, level="INFO"):
        timestamp = time.strftime("%H:%M:%S")
        color_map = {
            "INFO": "white", "WARN": "#F39C12",
            "ERROR": "#E74C3C", "SUCCESS": "#2ECC71"
        }
        color = color_map.get(level, "white")
        self.log_widget.append(
            f'<span style="color:{color};">[{timestamp}] {text}</span>')
        self.log_widget.moveCursor(QTextCursor.End)

    # ================================================================
    # SETTINGS PERSISTENCE — Save & Load
    # ================================================================
    def save_settings(self):
        """Persist ALL current UI + TTS proxy values to disk."""
        if getattr(self, '_loading_settings', False):
            return

        s = {
            'resolution': self.combo_res.currentIndex(),
            'fps': self.spin_fps.value(),
            'gpu': self.chk_gpu.isChecked(),
            'merge': self.chk_merge.isChecked(),
            'suffix': self.txt_suffix.text(),
            'spec_type': self.combo_spec_type.currentText(),
            'visual_mode': self.combo_vis_mode.currentText(),
            'colormap': self.combo_cmap.currentText(),
            'color_mode': self.combo_color_mode.currentText(),
            'n_fft': int(self.combo_nfft.currentText()),
            'hop_length': self.spin_hop.value(),
            'interpolation': self.combo_interp.currentText().lower(),
            'sharpen': self.chk_sharpen.isChecked(),
            'overlay': self.chk_overlay.isChecked(),
            'window': self.spin_window.value(),
            'auto_freq': self.chk_auto_freq.isChecked(),
            'min_freq': self.spin_freq_min.value(),
            'max_freq': self.spin_freq_max.value(),
            'dyn_range': self.spin_dyn_range.value(),
            'gamma': self.spin_gamma.value(),
            'normalize': self.chk_normalize.isChecked(),
            'auto_highpass': self.chk_auto_hp.isChecked(),
            'highpass_freq': self.spin_highpass.value(),
            'bass': self.spin_bass.value(),
            'treble': self.spin_treble.value(),
            'center_line': self.chk_center_line.isChecked(),
            'analyzer_color_mode': self.combo_analyzer_color.currentText(),
            'line_base_width': self.spin_line_width.value(),
            'line_magnitude_sensitivity':
                self.spin_line_sensitivity.value(),
            'line_glow_intensity': self.spin_line_glow.value(),
            'line_bloom': self.spin_line_bloom.value(),
            'analyzer_bar_count': self.spin_bar_count.value(),
            'analyzer_position': self.combo_analyzer_pos.currentText(),
            'analyzer_show_center_line':
                self.chk_draw_center_line.isChecked(),
            'last_input_dir': self.last_input_dir,
            'default_load_folder': self.default_load_folder,
            'palette_folder': self._palette_folder,
            'tts_backend': self._current_tts_backend_key,
        }
        # Merge TTS proxy settings (voice, rate, pitch, qwen3, etc.)
        s.update(self.tts_settings)

        config = {
            'output_batch': self.output_target_batch,
            'output_merge': self.output_target_merge,
            'settings': s
        }
        self.config_manager.save(config)

    def load_settings(self):
        """Read persisted settings from disk and apply to widgets.
        Sets _loading_settings=True so intermediate saves are skipped.
        """
        self._loading_settings = True

        config = self.config_manager.load()
        if config.get('output_batch'):
            self.output_target_batch = config['output_batch']
        if config.get('output_merge'):
            self.output_target_merge = config['output_merge']
        self.toggle_output_mode()

        s = config.get('settings', DEFAULT_SETTINGS)

        self.combo_res.setCurrentIndex(s.get('resolution', 1))
        self.spin_fps.setValue(s.get('fps', 30))
        self.chk_gpu.setChecked(s.get('gpu', True))
        self.chk_merge.setChecked(s.get('merge', False))
        self.txt_suffix.setText(s.get('suffix', ''))

        self._set_combo_text(
            self.combo_spec_type, s.get('spec_type', 'Standard (STFT)'))
        self._set_combo_text(
            self.combo_vis_mode, s.get('visual_mode', 'Standard'))
        self._set_combo_text(
            self.combo_cmap, s.get('colormap', 'Vibrant (Turbo)'))
        self._set_combo_text(
            self.combo_color_mode, s.get('color_mode', 'Static'))
        self._set_combo_text(
            self.combo_nfft, str(s.get('n_fft', 4096)))
        self._set_combo_text(
            self.combo_interp, s.get('interpolation', 'Lanczos'))

        self.spin_hop.setValue(s.get('hop_length', 512))
        self.chk_sharpen.setChecked(s.get('sharpen', True))
        self.chk_overlay.setChecked(s.get('overlay', True))
        self.spin_window.setValue(s.get('window', 5.0))
        self.chk_auto_freq.setChecked(s.get('auto_freq', True))
        self.toggle_auto_freq()
        self.spin_freq_min.setValue(s.get('min_freq', 20))
        self.spin_freq_max.setValue(s.get('max_freq', 20000))
        self.spin_dyn_range.setValue(s.get('dyn_range', 80))
        self.spin_gamma.setValue(s.get('gamma', 1.0))

        self.chk_normalize.setChecked(s.get('normalize', True))
        self.chk_auto_hp.setChecked(s.get('auto_highpass', True))
        self.toggle_auto_highpass()
        self.spin_highpass.setValue(s.get('highpass_freq', 20))
        self.spin_bass.setValue(s.get('bass', 0.0))
        self.spin_treble.setValue(s.get('treble', 0.0))

        self.chk_center_line.setChecked(s.get('center_line', True))
        self.chk_draw_center_line.setChecked(
            s.get('analyzer_show_center_line', True))
        self._set_combo_text(
            self.combo_analyzer_color,
            s.get('analyzer_color_mode', 'Match Video'))
        self._set_combo_text(
            self.combo_analyzer_pos,
            s.get('analyzer_position', 'Center'))
        self.spin_line_width.setValue(s.get('line_base_width', 6))
        self.spin_line_sensitivity.setValue(
            s.get('line_magnitude_sensitivity', 800.0))
        self.spin_line_glow.setValue(
            s.get('line_glow_intensity', 15.0))
        self.spin_line_bloom.setValue(s.get('line_bloom', 8.0))
        self.spin_bar_count.setValue(s.get('analyzer_bar_count', 64))

        saved_default = s.get('default_load_folder', '').strip()
        self.default_load_folder = (
            saved_default if saved_default and os.path.isdir(saved_default)
            else self.input_dir)
        self.lbl_default_folder.setText(self.default_load_folder)

        saved_last = s.get('last_input_dir', '').strip()
        self.last_input_dir = (
            saved_last if saved_last and os.path.isdir(saved_last)
            else self.default_load_folder)

        if s.get('palette_folder', '').strip():
            self._palette_folder = s['palette_folder']
        self.update_palette_preview()

        # Load TTS settings into proxy dict (but don't save yet)
        tts_keys = [k for k in DEFAULT_SETTINGS if k.startswith((
            'edge_', 'qwen3_', 'tts_', 'enable_text_chunking',
            'chunk_', 'ffmpeg_path', 'voice_references_root', 'hf_cache_dir'))]
        for key in tts_keys:
            self.tts_settings[key] = s.get(
                key, DEFAULT_SETTINGS.get(key, ''))

        self._current_tts_backend_key = s.get(
            'tts_backend',
            DEFAULT_SETTINGS.get('tts_backend', 'edge_tts'))

        # _loading_settings stays True until _final_save_after_load()
        # is called after _init_tts_backends() completes.

    @staticmethod
    def _set_combo_text(combo, text):
        idx = combo.findText(text)
        if idx >= 0:
            combo.setCurrentIndex(idx)

    def reset_settings(self):
        self.config_manager.save({'settings': DEFAULT_SETTINGS})
        self.load_settings()
        self._final_save_after_load()
        self.log_message("Settings reset to defaults.", "WARN")

    # ================================================================
    # FILE MANAGEMENT
    # ================================================================
    def load_all_from_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Folder", self.last_input_dir)
        if folder:
            self.add_files_from_paths([folder])
            self.last_input_dir = folder
            self.save_settings()

    def add_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Audio", self.last_input_dir,
            "Audio (*.wav *.mp3 *.ogg *.flac *.aac *.m4a)")
        if files:
            self.add_files_from_paths(files)
            self.last_input_dir = os.path.dirname(files[0])
            self.save_settings()

    def add_files_from_paths(self, paths):
        count = 0
        for path in paths:
            if os.path.isdir(path):
                walker = (os.walk(path) if self.chk_recursive.isChecked()
                          else [(path, [], os.listdir(path))])
                for root, _dirs, files in walker:
                    for f in files:
                        if f.lower().endswith(AUDIO_EXTS):
                            count += self.add_valid_item(
                                os.path.join(root, f))
                    if not self.chk_recursive.isChecked():
                        break
            elif os.path.isfile(path) and path.lower().endswith(AUDIO_EXTS):
                count += self.add_valid_item(path)
        if count > 0:
            self.log_message(f"Added {count} file(s).")

    def add_valid_item(self, filepath):
        for row in range(self.table_widget.rowCount()):
            item = self.table_widget.item(row, 0)
            if item and item.data(Qt.UserRole) == filepath:
                return 0
        try:
            info = sf.info(filepath)
            row = self.table_widget.rowCount()
            self.table_widget.insertRow(row)
            name_item = QTableWidgetItem(os.path.basename(filepath))
            name_item.setData(Qt.UserRole, filepath)
            name_item.setToolTip(filepath)
            self.table_widget.setItem(row, 0, name_item)
            dur_str = (f"{int(info.duration // 60):02}:"
                       f"{int(info.duration % 60):02}")
            self.table_widget.setItem(row, 1, QTableWidgetItem(dur_str))
            self.table_widget.setItem(row, 2, QTableWidgetItem("Pending"))
            return 1
        except Exception as e:
            self.log_message(
                f"Cannot read {os.path.basename(filepath)}: {e}", "WARN")
            return 0

    def update_table_status(self, filepath, text, color):
        for row in range(self.table_widget.rowCount()):
            item = self.table_widget.item(row, 0)
            if item and item.data(Qt.UserRole) == filepath:
                status_item = self.table_widget.item(row, 2)
                if status_item:
                    status_item.setText(text)
                    status_item.setForeground(QBrush(QColor(color)))
                break

    def toggle_output_mode(self):
        is_merge = self.chk_merge.isChecked()
        self.output_target = (
            self.output_target_merge if is_merge
            else self.output_target_batch)
        self.lbl_output.setText(self.output_target)

    def select_output(self):
        if self.chk_merge.isChecked():
            f, _ = QFileDialog.getSaveFileName(
                self, "Save Video", self.output_target, "MP4 (*.mp4)")
            if f:
                if not f.lower().endswith('.mp4'):
                    f += '.mp4'
                self.output_target_merge = f
                self.output_target = f
                self.lbl_output.setText(f)
        else:
            d = QFileDialog.getExistingDirectory(
                self, "Select Folder", self.output_target)
            if d:
                self.output_target_batch = d
                self.output_target = d
                self.lbl_output.setText(d)
        self.save_settings()

    # ================================================================
    # RENDER PIPELINE
    # ================================================================
    def get_common_settings(self, preview=False):
        spec_type_map = {
            'Standard (STFT)': 'stft', 'Mel-Scaled': 'mel', 'CQT': 'cqt'}
        vis_mode_map = {
            'Standard': 'standard', 'Classic': 'classic', 'Radial': 'radial'}
        return {
            'merge_mode': self.chk_merge.isChecked(),
            'colormap': self.combo_cmap.currentText(),
            'luts': self.available_luts,
            'color_mode': self.combo_color_mode.currentText().lower().replace(
                ' ', '_'),
            'visual_mode': vis_mode_map.get(
                self.combo_vis_mode.currentText(), 'standard'),
            'spec_type': spec_type_map.get(
                self.combo_spec_type.currentText(), 'stft'),
            'use_gpu': self.chk_gpu.isChecked(),
            'width': [1280, 1920, 2560, 3840][self.combo_res.currentIndex()],
            'height': [720, 1080, 1440, 2160][self.combo_res.currentIndex()],
            'fps': self.spin_fps.value(),
            'preview_mode': preview,
            'draw_overlay': self.chk_overlay.isChecked(),
            'n_fft': int(self.combo_nfft.currentText()),
            'hop_length': self.spin_hop.value(),
            'sharpen': self.chk_sharpen.isChecked(),
            'interpolation': self.combo_interp.currentText().lower(),
            'window_size': self.spin_window.value(),
            'auto_freq': self.chk_auto_freq.isChecked(),
            'min_freq': self.spin_freq_min.value(),
            'max_freq': self.spin_freq_max.value(),
            'dynamic_range_db': self.spin_dyn_range.value(),
            'gamma': self.spin_gamma.value(),
            'normalize': self.chk_normalize.isChecked(),
            'auto_highpass': self.chk_auto_hp.isChecked(),
            'highpass_freq': self.spin_highpass.value(),
            'bass_gain': self.spin_bass.value(),
            'treble_gain': self.spin_treble.value(),
            'filename_suffix': self.txt_suffix.text(),
            'center_line': self.chk_center_line.isChecked(),
            'analyzer_color_mode': self.combo_analyzer_color.currentText(),
            'line_base_width': self.spin_line_width.value(),
            'line_magnitude_sensitivity':
                self.spin_line_sensitivity.value(),
            'line_glow_intensity': self.spin_line_glow.value(),
            'line_bloom': self.spin_line_bloom.value(),
            'analyzer_bar_count': self.spin_bar_count.value(),
            'analyzer_position': self.combo_analyzer_pos.currentText(),
            'analyzer_show_center_line':
                self.chk_draw_center_line.isChecked(),
        }

    def start_processing(self):
        files = self.get_files_to_process()
        if not files:
            self.log_message("No files to process.", "WARN")
            return
        settings = self.get_common_settings()
        self.save_settings()
        self._preview_path = None
        self.run_worker(files, self.output_target, settings)

    def start_preview(self):
        files = self.get_files_to_process()
        if not files:
            self.log_message("No files to preview.", "WARN")
            return
        preview_output = os.path.join(
            tempfile.gettempdir(), "spectrogram_preview.mp4")
        self._preview_path = preview_output
        settings = self.get_common_settings(preview=True)
        self.run_worker(files, preview_output, settings)

    def get_files_to_process(self):
        sel = []
        for i in self.table_widget.selectedItems():
            item = self.table_widget.item(i.row(), 0)
            if item:
                sel.append(item.data(Qt.UserRole))
        sel = list(set(sel))
        if sel:
            return sel
        all_files = []
        for row in range(self.table_widget.rowCount()):
            item = self.table_widget.item(row, 0)
            if item:
                all_files.append(item.data(Qt.UserRole))
        return all_files

    def run_worker(self, files, output_target, settings):
        self._set_busy(True)
        self.btn_start.setEnabled(False)
        self.btn_preview.setEnabled(False)
        self.btn_tts_to_mp4.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.error_count = 0
        self.progress_bar.setValue(0)

        self.worker = RenderWorker(files, output_target, settings)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.status.connect(
            lambda s: self.log_message(s, "INFO"))
        self.worker.log.connect(lambda m: self.log_message(m))
        self.worker.file_started.connect(
            lambda f: self.update_table_status(f, "Rendering", "yellow"))
        self.worker.file_finished.connect(
            lambda f: self.update_table_status(f, "Done", "#2ECC71"))
        self.worker.batch_error.connect(self._on_batch_error)
        self.worker.error.connect(self._on_fatal_error)
        self.worker.render_finished.connect(self.on_render_finished)
        self.worker.start()

    def _on_batch_error(self, filepath, msg):
        self.update_table_status(filepath, "Error", "red")
        self.error_count += 1
        self.log_message(
            f"Error: {os.path.basename(filepath)} — {msg}", "ERROR")

    def _on_fatal_error(self, msg):
        self.log_message(f"Fatal: {msg}", "ERROR")

    def cancel_processing(self):
        cancelled = False
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            cancelled = True
        if self.tts_render_worker and self.tts_render_worker.isRunning():
            self.tts_render_worker.cancel()
            cancelled = True
        if cancelled:
            self.log_message("Cancelling…", "WARN")

    def on_render_finished(self):
        self._set_busy(False)
        self.btn_start.setEnabled(True)
        self.btn_preview.setEnabled(True)
        self.btn_tts_to_mp4.setEnabled(True)
        self.btn_cancel.setEnabled(False)

        if self.worker and self.worker.is_cancelled:
            self.progress_bar.setValue(0)
            self.log_message("Render cancelled.", "WARN")
        else:
            self.progress_bar.setValue(100)
            if self.error_count > 0:
                self.log_message(
                    f"Batch complete with {self.error_count} error(s).",
                    "WARN")
            else:
                self.log_message("Batch complete!", "SUCCESS")

            if (self._preview_path
                    and os.path.exists(self._preview_path)):
                self.log_message(
                    f"Preview saved: {self._preview_path}", "SUCCESS")
                open_file_location(self._preview_path)
                self._preview_path = None

        self.worker = None