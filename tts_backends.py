# tts_backends.py
"""
Consolidated TTS Backend module.
Contains the base class, Edge TTS backend, and Qwen3 backend,
plus the global BACKEND_REGISTRY for the main window to use.
"""

import os
import gc
import logging
import asyncio
import wave
import tempfile
import struct
import shutil
import glob
import subprocess
from datetime import datetime
from typing import List, Tuple, Optional, Dict
from abc import ABC, abstractmethod

# ==============================================================================
# LOGGING
# ==============================================================================
logger = logging.getLogger("SpectrogramStudio")

# ==============================================================================
# OPTIONAL TTS LIBRARY IMPORTS
# ==============================================================================
try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    edge_tts = None
    EDGE_TTS_AVAILABLE = False

try:
    from qwen_tts import Qwen3TTSModel
    QWEN_AVAILABLE = True
except ImportError:
    Qwen3TTSModel = None
    QWEN_AVAILABLE = False

try:
    import soundfile as sf
except ImportError:
    sf = None

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False

# ==============================================================================
# UI IMPORTS
# ==============================================================================
from PySide6.QtWidgets import (
    QWidget, QFormLayout, QComboBox, QLineEdit,
    QPlainTextEdit, QHBoxLayout, QLabel, QPushButton, QFileDialog,
    QMessageBox, QSlider, QStackedWidget,
)
from PySide6.QtCore import Qt

# Local imports
from text_chunker import TextChunker, ChunkingConfig, get_chunker_for_language
from utils import logger as _logger


# =========================================================================
# BASE CLASS
# =========================================================================
class BaseTTS_backend(ABC):
    """
    Abstract Base Class for all TTS Backends.
    All backend plugins must inherit from this and implement defined methods.
    """

    def __init__(self, config: dict):
        self.config = config
        self._progress_callback: Optional[callable] = None

    def set_progress_callback(self, callback: Optional[callable]):
        self._progress_callback = callback

    def _report_progress(self, stage: str, current: int, total: int,
                         message: str = ""):
        if self._progress_callback:
            self._progress_callback(stage, current, total, message)

    @abstractmethod
    def initialize(self):
        """Load models into memory (VRAM/RAM)."""
        pass

    @abstractmethod
    def generate_batch(self, texts: List[str],
                       output_paths: List[str]) -> Tuple[bool, List[str]]:
        """
        Generate audio for a list of texts.
        Returns: (success_bool, list_of_error_messages)
        """
        pass

    @abstractmethod
    def cleanup(self):
        """Unload models and free resources."""
        pass


# =========================================================================
# EDGE TTS BACKEND
# =========================================================================
class EdgeTTSBackend(BaseTTS_backend):
    """
    Concrete implementation for Microsoft Edge TTS.
    Uses the edge-tts package to access Microsoft's neural TTS voices.
    """

    EDGE_VOICES = {
        "English": [
            ("en-US-AvaMultilingualNeural", "Ava (US, Multilingual)"),
            ("en-US-AndrewMultilingualNeural", "Andrew (US, Multilingual)"),
            ("en-US-EmmaMultilingualNeural", "Emma (US, Multilingual)"),
            ("en-US-BrianMultilingualNeural", "Brian (US, Multilingual)"),
            ("en-US-AvaNeural", "Ava (US, Female)"),
            ("en-US-AndrewNeural", "Andrew (US, Male)"),
            ("en-US-EmmaNeural", "Emma (US, Female)"),
            ("en-US-BrianNeural", "Brian (US, Male)"),
            ("en-US-JennyNeural", "Jenny (US, Female)"),
            ("en-US-GuyNeural", "Guy (US, Male)"),
            ("en-GB-SoniaNeural", "Sonia (UK, Female)"),
            ("en-GB-RyanNeural", "Ryan (UK, Male)"),
            ("en-AU-NatashaNeural", "Natasha (AU, Female)"),
            ("en-AU-WilliamNeural", "William (AU, Male)"),
            ("en-CA-ClaraNeural", "Clara (CA, Female)"),
            ("en-CA-LiamNeural", "Liam (CA, Male)"),
            ("en-IN-NeerjaNeural", "Neerja (IN, Female)"),
            ("en-IN-PrabhatNeural", "Prabhat (IN, Male)"),
        ],
        "Chinese": [
            ("zh-CN-XiaoxiaoNeural", "Xiaoxiao (Female)"),
            ("zh-CN-YunxiNeural", "Yunxi (Male)"),
            ("zh-CN-YunjianNeural", "Yunjian (Male)"),
            ("zh-CN-XiaoyiNeural", "Xiaoyi (Female)"),
            ("zh-CN-YunyangNeural", "Yunyang (Male)"),
        ],
        "Japanese": [
            ("ja-JP-NanamiNeural", "Nanami (Female)"),
            ("ja-JP-KeitaNeural", "Keita (Male)"),
        ],
        "Korean": [
            ("ko-KR-SunHiNeural", "SunHi (Female)"),
            ("ko-KR-InJoonNeural", "InJoon (Male)"),
        ],
        "German": [
            ("de-DE-KatjaNeural", "Katja (Female)"),
            ("de-DE-ConradNeural", "Conrad (Male)"),
        ],
        "French": [
            ("fr-FR-DeniseNeural", "Denise (Female)"),
            ("fr-FR-HenriNeural", "Henri (Male)"),
        ],
        "Russian": [
            ("ru-RU-SvetlanaNeural", "Svetlana (Female)"),
            ("ru-RU-DmitryNeural", "Dmitry (Male)"),
        ],
        "Spanish": [
            ("es-ES-ElviraNeural", "Elvira (ES, Female)"),
            ("es-ES-AlvaroNeural", "Alvaro (ES, Male)"),
        ],
        "Italian": [
            ("it-IT-ElsaNeural", "Elsa (Female)"),
            ("it-IT-DiegoNeural", "Diego (Male)"),
        ],
    }

    EDGE_LANGUAGES = list(EDGE_VOICES.keys())

    @staticmethod
    def get_ui_options() -> Dict:
        return {
            "languages": EdgeTTSBackend.EDGE_LANGUAGES,
            "voices": EdgeTTSBackend.EDGE_VOICES
        }

    @staticmethod
    def get_settings_widget(mode: str, config, parent=None) -> QWidget:
        widget = QWidget(parent)
        layout = QFormLayout(widget)

        layout.addRow(QLabel("<b>Edge TTS Configuration</b>"))
        layout.addRow(QLabel(
            "<i>Free online TTS using Microsoft Edge neural voices</i>"))

        lang_combo = QComboBox()
        lang_combo.addItems(EdgeTTSBackend.EDGE_LANGUAGES)
        current_lang = config.get("edge_language", "English")
        if current_lang in EdgeTTSBackend.EDGE_LANGUAGES:
            lang_combo.setCurrentText(current_lang)

        def on_language_changed(lang):
            config.set("edge_language", lang)
            voice_combo = widget.findChild(QComboBox, "voice_combo")
            if voice_combo:
                voice_combo.clear()
                voices = EdgeTTSBackend.EDGE_VOICES.get(lang, [])
                for voice_id, voice_name in voices:
                    voice_combo.addItem(voice_name, voice_id)
                if voices:
                    current_voice = config.get("edge_voice", "")
                    found = False
                    for i in range(voice_combo.count()):
                        if voice_combo.itemData(i) == current_voice:
                            voice_combo.setCurrentIndex(i)
                            found = True
                            break
                    if not found:
                        voice_combo.setCurrentIndex(0)
                        config.set("edge_voice", voices[0][0])

        lang_combo.currentTextChanged.connect(on_language_changed)
        layout.addRow("Language:", lang_combo)

        voice_combo = QComboBox()
        voice_combo.setObjectName("voice_combo")
        voices = EdgeTTSBackend.EDGE_VOICES.get(current_lang, [])
        for voice_id, voice_name in voices:
            voice_combo.addItem(voice_name, voice_id)

        current_voice = config.get("edge_voice", "")
        if current_voice:
            for i in range(voice_combo.count()):
                if voice_combo.itemData(i) == current_voice:
                    voice_combo.setCurrentIndex(i)
                    break

        def on_voice_changed(index):
            if index >= 0:
                voice_id = voice_combo.itemData(index)
                config.set("edge_voice", voice_id)

        voice_combo.currentIndexChanged.connect(on_voice_changed)
        layout.addRow("Voice:", voice_combo)

        rate_layout = QHBoxLayout()
        rate_slider = QSlider(Qt.Orientation.Horizontal)
        rate_slider.setMinimum(-100)
        rate_slider.setMaximum(100)
        rate_raw = config.get("edge_rate", "+0%")
        try:
            rate_val = int(str(rate_raw).replace("%", "").replace("+", ""))
        except ValueError:
            rate_val = 0
        rate_slider.setValue(rate_val)
        rate_label = QLabel(f"{rate_slider.value()}%")
        rate_label.setMinimumWidth(50)
        rate_slider.valueChanged.connect(
            lambda v: rate_label.setText(f"{v}%"))
        rate_slider.sliderReleased.connect(
            lambda: config.set("edge_rate", f"{rate_slider.value()}%"))
        rate_layout.addWidget(rate_slider)
        rate_layout.addWidget(rate_label)
        layout.addRow("Speech Rate:", rate_layout)

        pitch_layout = QHBoxLayout()
        pitch_slider = QSlider(Qt.Orientation.Horizontal)
        pitch_slider.setMinimum(-50)
        pitch_slider.setMaximum(50)
        pitch_raw = config.get("edge_pitch", "+0Hz")
        try:
            pitch_val = int(str(pitch_raw).replace("+", "").replace("Hz", ""))
        except ValueError:
            pitch_val = 0
        pitch_slider.setValue(pitch_val)
        pitch_label = QLabel(f"{pitch_slider.value():+d}Hz")
        pitch_label.setMinimumWidth(50)
        pitch_slider.valueChanged.connect(
            lambda v: pitch_label.setText(f"{v:+d}Hz"))
        pitch_slider.sliderReleased.connect(
            lambda: config.set("edge_pitch", f"{pitch_slider.value():+d}Hz"))
        pitch_layout.addWidget(pitch_slider)
        pitch_layout.addWidget(pitch_label)
        layout.addRow("Pitch:", pitch_layout)

        volume_layout = QHBoxLayout()
        volume_slider = QSlider(Qt.Orientation.Horizontal)
        volume_slider.setMinimum(-100)
        volume_slider.setMaximum(100)
        vol_raw = config.get("edge_volume", "+0%")
        try:
            vol_val = int(str(vol_raw).replace("%", "").replace("+", ""))
        except ValueError:
            vol_val = 0
        volume_slider.setValue(vol_val)
        volume_label = QLabel(f"{volume_slider.value():+d}%")
        volume_label.setMinimumWidth(50)
        volume_slider.valueChanged.connect(
            lambda v: volume_label.setText(f"{v:+d}%"))
        volume_slider.sliderReleased.connect(
            lambda: config.set("edge_volume", f"{volume_slider.value():+d}%"))
        volume_layout.addWidget(volume_slider)
        volume_layout.addWidget(volume_label)
        layout.addRow("Volume:", volume_layout)

        on_language_changed(current_lang)
        return widget

    def __init__(self, config: dict):
        super().__init__(config)
        if not EDGE_TTS_AVAILABLE:
            raise RuntimeError(
                "edge-tts library is not installed. "
                "Install via: pip install edge-tts")

    def initialize(self):
        _logger.info("[EdgeTTSBackend] Initializing (online service)")
        voice = self.config.get("edge_voice", "")
        if not voice:
            language = self.config.get("edge_language", "English")
            voices = self.EDGE_VOICES.get(language, [])
            if voices:
                self.config["edge_voice"] = voices[0][0]

    def cleanup(self):
        gc.collect()

    def generate_batch(self, texts: List[str],
                       output_paths: List[str]) -> Tuple[bool, List[str]]:
        errors = []
        success_count = 0

        valid_texts = [t for t in texts if t and t.strip()]
        if not valid_texts:
            raise ValueError("Cannot generate audio: all texts are empty")

        enable_chunking = self.config.get("enable_text_chunking", True)
        chunker = self._get_chunker() if enable_chunking else None
        temp_dir = tempfile.mkdtemp(prefix="edge_tts_chunks_")

        for i, text in enumerate(texts):
            _logger.info(
                f"[EdgeTTSBackend] Generating {i + 1}/{len(texts)}...")
            needs_chunking = chunker and chunker.needs_chunking(text)

            if needs_chunking:
                language = self.config.get("edge_language", "English")
                chunks = chunker.chunk_text(text, language)
                chunk_paths = []
                chunk_errors = []

                for chunk_idx, (chunk_text, _) in enumerate(chunks):
                    chunk_path = os.path.join(
                        temp_dir, f"chunk_{i}_{chunk_idx}.mp3")
                    self._report_progress(
                        "chunk", chunk_idx + 1, len(chunks),
                        f"Chunk {chunk_idx + 1}/{len(chunks)}")
                    try:
                        success = self._generate_single(
                            chunk_text, chunk_path)
                        if success:
                            chunk_paths.append(chunk_path)
                        else:
                            chunk_errors.append(
                                f"Failed chunk {chunk_idx + 1}")
                    except Exception as e:
                        chunk_errors.append(f"Chunk {chunk_idx + 1}: {e}")

                if len(chunk_paths) == len(chunks):
                    if self._concatenate_audio_files(
                            chunk_paths, output_paths[i]):
                        success_count += 1
                    else:
                        errors.append(f"Concat failed for slide {i + 1}")
                else:
                    errors.extend(chunk_errors)
            else:
                try:
                    if self._generate_single(text, output_paths[i]):
                        success_count += 1
                    else:
                        errors.append(f"Failed slide {i + 1}")
                except Exception as e:
                    errors.append(str(e))

        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass

        return (True, []) if success_count == len(texts) else (False, errors)

    def _generate_single(self, text: str, output_path: str) -> bool:
        if not text.strip():
            raise ValueError("Text is empty")

        voice = self.config.get("edge_voice", "en-US-JennyNeural")
        rate = self.config.get("edge_rate", "+0%")
        pitch = self.config.get("edge_pitch", "+0Hz")
        volume_raw = self.config.get("edge_volume", "+0%")

        volume = volume_raw
        if volume_raw and not volume_raw.startswith(('+', '-')):
            try:
                volume = f"{int(volume_raw.replace('%', '')) - 100:+d}%"
            except ValueError:
                volume = "+0%"

        try:
            communicate = edge_tts.Communicate(
                text=text.strip(), voice=voice,
                rate=rate, pitch=pitch, volume=volume)

            out_dir = os.path.dirname(output_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)

            try:
                loop = asyncio.get_running_loop()
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run, communicate.save(output_path))
                    future.result()
            except RuntimeError:
                asyncio.run(communicate.save(output_path))

            if output_path.endswith('.mp3'):
                wav_path = output_path.replace('.mp3', '.wav')
                if self._convert_mp3_to_wav(output_path, wav_path):
                    os.remove(output_path)
                    os.rename(wav_path, output_path)

            _logger.info(f"[EdgeTTSBackend] Generated: {output_path}")
            return True

        except Exception as e:
            _logger.error(f"[EdgeTTSBackend] Generation failed: {e}")
            return False

    def _convert_mp3_to_wav(self, mp3_path: str, wav_path: str) -> bool:
        try:
            ffmpeg_path = self.config.get("ffmpeg_path", "ffmpeg")
            result = subprocess.run(
                [ffmpeg_path, "-y", "-i", mp3_path,
                 "-ar", "24000", "-ac", "1", wav_path],
                capture_output=True, timeout=30)
            return result.returncode == 0
        except Exception as e:
            _logger.warning(f"Could not convert MP3 to WAV: {e}")
            return False

    def _concatenate_audio_files(self, audio_files: List[str],
                                 output_path: str) -> bool:
        if not audio_files:
            return False
        if len(audio_files) == 1:
            try:
                shutil.copy(audio_files[0], output_path)
                return True
            except Exception:
                return False

        try:
            with tempfile.NamedTemporaryFile(
                    mode='w', suffix='.txt', delete=False) as f:
                for af in audio_files:
                    abs_path = os.path.abspath(af).replace(os.sep, '/')
                    f.write(f"file '{abs_path}'\n")
                concat_file = f.name

            ffmpeg_path = self.config.get("ffmpeg_path", "ffmpeg")
            cmd = [
                ffmpeg_path, "-y", "-f", "concat", "-safe", "0",
                "-i", concat_file, "-ar", "24000", "-ac", "1", output_path
            ]
            result = subprocess.run(cmd, capture_output=True, timeout=60)

            try:
                os.remove(concat_file)
            except OSError:
                pass

            return result.returncode == 0
        except Exception as e:
            _logger.error(f"Failed to concatenate audio files: {e}")
            return False

    def _get_chunker(self) -> TextChunker:
        config = ChunkingConfig(
            max_chars=self.config.get("chunk_max_chars", 500),
            max_sentences=self.config.get("chunk_max_sentences", 5),
            min_chunk_chars=self.config.get("chunk_min_chars", 50),
            language=self.config.get("edge_language", "English")
        )
        return TextChunker(config)

    @staticmethod
    async def list_available_voices(language: str = None) -> List[Dict]:
        try:
            voices = await edge_tts.list_voices()
            if language:
                voices = [
                    v for v in voices
                    if v['Locale'].startswith(language)]
            return voices
        except Exception as e:
            _logger.error(f"Failed to list voices: {e}")
            return []


# =========================================================================
# QWEN3 TTS BACKEND
# =========================================================================
class Qwen3Backend(BaseTTS_backend):
    """
    Concrete implementation for Qwen3.
    Supports 3 modes: Voice Custom, Voice Clone, Voice Design.
    """

    QWEN3_LANGUAGES = [
        "Chinese", "English", "Japanese", "Korean",
        "German", "French", "Russian", "Portuguese",
        "Spanish", "Italian"
    ]

    QWEN3_MODES = [
        ("voice_custom", "Voice Custom (Preset Speakers)"),
        ("voice_clone", "Voice Clone (User Reference)"),
        ("voice_design", "Voice Design (Generate Reference)"),
    ]

    @staticmethod
    def get_ui_options() -> Dict:
        return {
            "modes": Qwen3Backend.QWEN3_MODES,
            "languages": Qwen3Backend.QWEN3_LANGUAGES
        }

    def _get_model_id(self) -> str:
        mode = self.config.get("qwen3_mode", "voice_custom")
        size = self.config.get("qwen3_size", "1.7B")
        if mode == "voice_design":
            suffix = "VoiceDesign"
        elif mode == "voice_clone":
            suffix = "Base"
        else:
            suffix = "CustomVoice"
        return f"Qwen/Qwen3-TTS-12Hz-{size}-{suffix}"

    @staticmethod
    def _find_local_model_path(model_id: str, cache_dir: str) -> str:
        if not cache_dir or not os.path.exists(cache_dir):
            return model_id

        model_author, model_name = (
            model_id.split("/", 1) if "/" in model_id else ("", model_id))

        candidates = []
        hf_folder = f"models--{model_author}--{model_name}"
        hf_path = os.path.join(cache_dir, hf_folder)
        if os.path.isdir(hf_path):
            snapshots = os.path.join(hf_path, "snapshots")
            if os.path.isdir(snapshots):
                try:
                    snaps = os.listdir(snapshots)
                    if snaps:
                        candidates.append(
                            os.path.join(snapshots, snaps[0]))
                except Exception:
                    pass

        candidates.append(os.path.join(cache_dir, model_name))
        candidates.append(
            os.path.join(cache_dir, model_id.replace("/", "_")))

        for path in candidates:
            if os.path.isdir(path):
                has_config = os.path.exists(
                    os.path.join(path, "config.json"))
                has_weights = (
                    glob.glob(os.path.join(path, "*.safetensors"))
                    or glob.glob(
                        os.path.join(path, "pytorch_model*.bin")))
                if has_config and has_weights:
                    return path

        return model_id

    @staticmethod
    def _get_smart_attn_implementation(device: str,
                                        requested_attn: str) -> str:
        if not device.startswith("cuda"):
            return "eager"

        if requested_attn == "flash_attention_2":
            is_hardware_supported = False
            try:
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    idx = int(device.split(":")[1]) if ":" in device else 0
                    major, minor = torch.cuda.get_device_capability(idx)
                    if major >= 8:
                        is_hardware_supported = True
                    else:
                        _logger.warning(
                            f"GPU compute capability {major}.{minor} < 8.0. "
                            f"Flash Attention 2 requires Ampere+.")
            except Exception:
                pass

            if is_hardware_supported:
                if FLASH_ATTN_AVAILABLE:
                    return "flash_attention_2"
                else:
                    _logger.warning(
                        "flash_attn library not installed. "
                        "Falling back to Eager.")
            return "eager"

        return requested_attn

    @staticmethod
    def get_settings_widget(mode: str, config, parent=None) -> QWidget:
        widget = QWidget(parent)
        layout = QFormLayout(widget)

        mode_layout = QHBoxLayout()
        mode_combo = QComboBox()
        for mode_key, mode_name in Qwen3Backend.QWEN3_MODES:
            mode_combo.addItem(mode_name, mode_key)

        current_mode = config.get("qwen3_mode", "voice_custom")
        for i in range(mode_combo.count()):
            if mode_combo.itemData(i) == current_mode:
                mode_combo.setCurrentIndex(i)
                break
        mode_layout.addWidget(mode_combo)
        layout.addRow("Mode:", mode_layout)

        mode_stack = QStackedWidget()
        layout.addRow(mode_stack)

        mode_widgets = {}
        for mode_key, mode_name in Qwen3Backend.QWEN3_MODES:
            mode_w = Qwen3Backend._build_mode_widget(
                mode_key, config, parent)
            mode_widgets[mode_key] = mode_w
            mode_stack.addWidget(mode_w)

        def on_mode_changed(index):
            selected = mode_combo.currentData()
            config.set("qwen3_mode", selected)
            keys = [k for k, _ in Qwen3Backend.QWEN3_MODES]
            if selected in keys:
                mode_stack.setCurrentIndex(keys.index(selected))

        mode_combo.currentIndexChanged.connect(on_mode_changed)
        on_mode_changed(mode_combo.currentIndex())

        # Common settings
        layout.addRow(QLabel(""))  # Spacer
        
        device_edit = QLineEdit()
        device_edit.setText(config.get("qwen3_device_map", "cuda:0"))
        device_edit.editingFinished.connect(
            lambda: config.set("qwen3_device_map", device_edit.text()))
        layout.addRow("Device:", device_edit)

        size_combo = QComboBox()
        size_combo.addItems(["1.7B", "3B", "7B"])
        size_combo.setCurrentText(config.get("qwen3_size", "1.7B"))
        size_combo.currentTextChanged.connect(
            lambda t: config.set("qwen3_size", t))
        layout.addRow("Model Size:", size_combo)

        attn_combo = QComboBox()
        attn_combo.addItems(["flash_attention_2", "eager", "sdpa"])
        attn_combo.setCurrentText(
            config.get("qwen3_attn_implementation", "flash_attention_2"))
        attn_combo.currentTextChanged.connect(
            lambda t: config.set("qwen3_attn_implementation", t))
        layout.addRow("Attention:", attn_combo)

        dtype_combo = QComboBox()
        dtype_combo.addItems(["float16", "bfloat16", "float32"])
        dtype_combo.setCurrentText(
            config.get("qwen3_dtype", "float16"))
        dtype_combo.currentTextChanged.connect(
            lambda t: config.set("qwen3_dtype", t))
        layout.addRow("Dtype:", dtype_combo)

        cache_edit = QLineEdit()
        cache_edit.setText(config.get("hf_cache_dir", ""))
        cache_edit.setPlaceholderText("Leave empty for default HuggingFace cache")
        cache_edit.editingFinished.connect(
            lambda: config.set("hf_cache_dir", cache_edit.text()))
        layout.addRow("HF Cache Dir:", cache_edit)

        return widget

    @staticmethod
    def _build_mode_widget(mode: str, config, parent=None) -> QWidget:
        widget = QWidget(parent)
        layout = QFormLayout(widget)

        if mode == "voice_custom":
            layout.addRow(QLabel("<b>Voice Custom Configuration</b>"))
            layout.addRow(QLabel(
                "<i>Model: Qwen3-TTS-12Hz-1.7B-CustomVoice</i>"))

            speaker_edit = QLineEdit()
            speaker_edit.setText(config.get("qwen3_speaker", "Vivian"))
            speaker_edit.setPlaceholderText("e.g., Vivian, Ryan")
            speaker_edit.editingFinished.connect(
                lambda: config.set("qwen3_speaker",
                                   speaker_edit.text()))
            layout.addRow("Speaker Name:", speaker_edit)

            instruct_edit = QLineEdit()
            instruct_edit.setText(config.get("qwen3_instruct", ""))
            instruct_edit.setPlaceholderText(
                "e.g., 'Speak slowly and sadly'")
            instruct_edit.editingFinished.connect(
                lambda: config.set("qwen3_instruct",
                                   instruct_edit.text()))
            layout.addRow("Style Instruction:", instruct_edit)

            lang_combo = QComboBox()
            lang_combo.addItems(Qwen3Backend.QWEN3_LANGUAGES)
            lang_combo.setCurrentText(
                config.get("qwen3_language", "English"))
            lang_combo.currentTextChanged.connect(
                lambda t: config.set("qwen3_language", t))
            layout.addRow("Language:", lang_combo)

        elif mode == "voice_clone":
            layout.addRow(QLabel("<b>Voice Clone Configuration</b>"))
            layout.addRow(QLabel(
                "<i>Model: Qwen3-TTS-12Hz-1.7B-Base</i>"))

            ref_row = QHBoxLayout()
            ref_audio_edit = QLineEdit()
            ref_audio_edit.setText(config.get("qwen3_ref_audio", ""))
            ref_audio_edit.setPlaceholderText(
                "Path to reference .wav file")
            ref_audio_edit.editingFinished.connect(
                lambda: config.set("qwen3_ref_audio",
                                   ref_audio_edit.text()))
            ref_text_edit_ref = [None]

            btn_browse = QPushButton("...")
            btn_browse.setMaximumWidth(30)

            def _browse():
                fname, _ = QFileDialog.getOpenFileName(
                    None, "Select Reference Audio", "",
                    "Audio Files (*.wav *.mp3 *.flac)")
                if fname:
                    ref_audio_edit.setText(fname)
                    config.set("qwen3_ref_audio", fname)
                    te = ref_text_edit_ref[0]
                    if te:
                        txt_path = os.path.splitext(fname)[0] + ".txt"
                        if os.path.exists(txt_path):
                            try:
                                with open(txt_path, 'r',
                                          encoding='utf-8') as f:
                                    te.setText(f.read().strip())
                                    config.set("qwen3_ref_text",
                                               te.text())
                            except Exception:
                                pass

            btn_browse.clicked.connect(_browse)
            ref_row.addWidget(ref_audio_edit)
            ref_row.addWidget(btn_browse)
            layout.addRow("Reference Audio:", ref_row)

            ref_text_edit = QLineEdit()
            ref_text_edit_ref[0] = ref_text_edit
            ref_text_edit.setText(config.get("qwen3_ref_text", ""))
            ref_text_edit.setPlaceholderText(
                "Transcript of reference audio")
            ref_text_edit.editingFinished.connect(
                lambda: config.set("qwen3_ref_text",
                                   ref_text_edit.text()))
            layout.addRow("Reference Text:", ref_text_edit)

            lang_combo = QComboBox()
            lang_combo.addItems(Qwen3Backend.QWEN3_LANGUAGES)
            lang_combo.setCurrentText(
                config.get("qwen3_language", "English"))
            lang_combo.currentTextChanged.connect(
                lambda t: config.set("qwen3_language", t))
            layout.addRow("Language:", lang_combo)

        elif mode == "voice_design":
            layout.addRow(QLabel("<b>Voice Design Configuration</b>"))
            layout.addRow(QLabel(
                "<i>Model: Qwen3-TTS-12Hz-1.7B-VoiceDesign</i>"))

            desc_edit = QPlainTextEdit()
            desc_edit.setPlaceholderText(
                "Describe the voice "
                "(e.g., 'A deep male voice, speaking slowly')")
            desc_edit.setMaximumHeight(80)
            desc_edit.setPlainText(
                config.get("qwen3_vd_description", ""))
            desc_edit.textChanged.connect(
                lambda: config.set("qwen3_vd_description",
                                   desc_edit.toPlainText()))
            layout.addRow("Voice Description:", desc_edit)

            save_name_edit = QLineEdit()
            save_name_edit.setText(
                config.get("qwen3_vd_save_name",
                           "my_designed_voice"))
            save_name_edit.editingFinished.connect(
                lambda: config.set("qwen3_vd_save_name",
                                   save_name_edit.text()))
            layout.addRow("Save Reference As:", save_name_edit)

            lang_combo = QComboBox()
            lang_combo.addItems(Qwen3Backend.QWEN3_LANGUAGES)
            lang_combo.setCurrentText(
                config.get("qwen3_language", "English"))
            lang_combo.currentTextChanged.connect(
                lambda t: config.set("qwen3_language", t))
            layout.addRow("Language:", lang_combo)

        return widget

    def __init__(self, config: dict):
        super().__init__(config)
        self.model = None
        self.device = config.get("qwen3_device_map", "cuda:0")

        if not QWEN_AVAILABLE:
            raise RuntimeError(
                "Qwen3 library (qwen-tts) is not installed.")

    def initialize(self):
        if self.model is not None:
            self.cleanup()

        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        mode = self.config.get("qwen3_mode")
        _logger.info(f"[Qwen3Backend] Initializing for mode: {mode}")

        model_id = self._get_model_id()
        cache_dir = self.config.get("hf_cache_dir")
        model_path = self._find_local_model_path(model_id, cache_dir)

        requested_attn = self.config.get(
            "qwen3_attn_implementation", "flash_attention_2")
        actual_attn = self._get_smart_attn_implementation(
            self.device, requested_attn)

        dtype_str = self.config.get("qwen3_dtype", "float16")
        dtype = getattr(torch, dtype_str, torch.float16)

        try:
            _logger.info(
                f"[Qwen3Backend] Loading model from: {model_path}")

            is_local = os.path.isdir(model_path)

            self.model = Qwen3TTSModel.from_pretrained(
                model_path,
                local_files_only=is_local,
                device_map=self.device,
                torch_dtype=dtype,
                attn_implementation=actual_attn
            )
            _logger.info("[Qwen3Backend] Model loaded successfully.")

        except Exception as e:
            _logger.error(f"[Qwen3Backend] Failed to load model: {e}")
            self.model = None
            raise

    def cleanup(self):
        if self.model is not None:
            del self.model
            self.model = None
        gc.collect()
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def generate_batch(self, texts: List[str],
                       output_paths: List[str]) -> Tuple[bool, List[str]]:
        valid_texts = [t for t in texts if t and t.strip()]
        if not valid_texts:
            raise ValueError("Cannot generate audio: all texts are empty")

        if self.model is None:
            try:
                self.initialize()
            except Exception as e:
                return False, [f"Initialization failed: {e}"]

        mode = self.config.get("qwen3_mode")
        try:
            if mode == "voice_custom":
                return self._gen_voice_custom(texts, output_paths)
            elif mode == "voice_clone":
                return self._gen_voice_clone(texts, output_paths)
            elif mode == "voice_design":
                return self._gen_voice_design(texts, output_paths)
            else:
                return False, [f"Unknown mode: {mode}"]
        except Exception as e:
            _logger.exception("[Qwen3Backend] Generation failed")
            return False, [str(e) for _ in texts]

    def _save_wav(self, wav, sr, path):
        try:
            if hasattr(wav, 'cpu'):
                wav = wav.cpu().numpy()
            out_dir = os.path.dirname(path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            if sf is not None:
                sf.write(path, wav, sr)
            else:
                import numpy as _np
                wav_int = (wav * 32767).astype(_np.int16)
                with wave.open(path, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(sr)
                    wf.writeframes(wav_int.tobytes())
            return True
        except Exception as e:
            _logger.error(f"Failed to save audio file {path}: {e}")
            return False

    def _concatenate_wav_files(self, wav_files: List[str],
                               output_path: str,
                               add_silence_ms: int = 100) -> bool:
        if not wav_files:
            return False
        if len(wav_files) == 1:
            try:
                shutil.copy(wav_files[0], output_path)
                return True
            except Exception:
                return False

        try:
            audio_data = []
            sample_rate = None
            n_channels = None
            sampwidth = None

            for wav_path in wav_files:
                if not os.path.exists(wav_path):
                    return False

                with wave.open(wav_path, 'rb') as wf:
                    if sample_rate is None:
                        sample_rate = wf.getframerate()
                        n_channels = wf.getnchannels()
                        sampwidth = wf.getsampwidth()

                    frames = wf.readframes(wf.getnframes())
                    audio_data.append(frames)

                    if wav_path != wav_files[-1] and add_silence_ms > 0:
                        silence_samples = int(
                            sample_rate * add_silence_ms / 1000)
                        silence_frames = (
                            b'\x00'
                            * (silence_samples * sampwidth * n_channels))
                        audio_data.append(silence_frames)

            out_dir = os.path.dirname(output_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)

            with wave.open(output_path, 'wb') as out_wf:
                out_wf.setnchannels(n_channels)
                out_wf.setsampwidth(sampwidth)
                out_wf.setframerate(sample_rate)
                for data in audio_data:
                    out_wf.writeframes(data)

            return True

        except Exception as e:
            _logger.error(f"Failed to concatenate WAV files: {e}")
            return False

    def _get_chunker(self) -> TextChunker:
        config = ChunkingConfig(
            max_chars=self.config.get("chunk_max_chars", 500),
            max_sentences=self.config.get("chunk_max_sentences", 5),
            min_chunk_chars=self.config.get("chunk_min_chars", 50),
            language=self.config.get("qwen3_language", "English")
        )
        return TextChunker(config)

    def _gen_voice_custom(self, texts, output_paths):
        errors = []
        success_count = 0
        enable_chunking = self.config.get("enable_text_chunking", True)
        chunker = self._get_chunker() if enable_chunking else None
        temp_dir = tempfile.mkdtemp(prefix="qwen3_chunks_")

        for i, text in enumerate(texts):
            needs_chunking = chunker and chunker.needs_chunking(text)

            if needs_chunking:
                language = self.config.get("qwen3_language", "English")
                chunks = chunker.chunk_text(text, language)
                chunk_paths = []

                for chunk_idx, (chunk_text, _) in enumerate(chunks):
                    chunk_path = os.path.join(
                        temp_dir, f"chunk_{i}_{chunk_idx}.wav")
                    try:
                        wav = self.model.synthesize(
                            text=chunk_text,
                            speaker=self.config.get(
                                "qwen3_speaker", "Vivian"),
                            instruct=self.config.get("qwen3_instruct", ""),
                            language=language
                        )
                        if self._save_wav(wav, 24000, chunk_path):
                            chunk_paths.append(chunk_path)
                    except Exception as e:
                        errors.append(f"Chunk {chunk_idx}: {e}")

                if len(chunk_paths) == len(chunks):
                    if self._concatenate_wav_files(
                            chunk_paths, output_paths[i]):
                        success_count += 1
                    else:
                        errors.append(f"Concat failed for slide {i + 1}")
            else:
                try:
                    wav = self.model.synthesize(
                        text=text,
                        speaker=self.config.get(
                            "qwen3_speaker", "Vivian"),
                        instruct=self.config.get("qwen3_instruct", ""),
                        language=self.config.get(
                            "qwen3_language", "English")
                    )
                    if self._save_wav(wav, 24000, output_paths[i]):
                        success_count += 1
                    else:
                        errors.append(f"Save failed for slide {i + 1}")
                except Exception as e:
                    errors.append(str(e))

        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass

        return (True, []) if success_count == len(texts) else (False, errors)

    def _gen_voice_clone(self, texts, output_paths):
        errors = []
        success_count = 0
        ref_audio = self.config.get("qwen3_ref_audio", "")
        ref_text = self.config.get("qwen3_ref_text", "")
        
        if not ref_audio or not os.path.exists(ref_audio):
            return False, ["Reference audio file not found or not set."]

        for i, text in enumerate(texts):
            try:
                wav = self.model.clone_voice(
                    text=text,
                    ref_audio=ref_audio,
                    ref_text=ref_text,
                    language=self.config.get("qwen3_language", "English")
                )
                if self._save_wav(wav, 24000, output_paths[i]):
                    success_count += 1
                else:
                    errors.append(f"Save failed for slide {i + 1}")
            except Exception as e:
                errors.append(str(e))

        return (True, []) if success_count == len(texts) else (False, errors)

    def _gen_voice_design(self, texts, output_paths):
        errors = []
        success_count = 0
        description = self.config.get("qwen3_vd_description", "")
        save_name = self.config.get("qwen3_vd_save_name", "my_voice")

        if not description:
            return False, ["Voice description is empty."]

        # Generate reference audio from description
        ref_dir = self.config.get("voice_references_root", "voicereferences")
        os.makedirs(ref_dir, exist_ok=True)
        ref_path = os.path.join(ref_dir, f"{save_name}.wav")
        ref_text_path = os.path.join(ref_dir, f"{save_name}.txt")

        try:
            # Create the designed voice reference
            designed_wav = self.model.design_voice(
                description=description,
                language=self.config.get("qwen3_language", "English")
            )
            self._save_wav(designed_wav, 24000, ref_path)
            
            # Save description as transcript
            with open(ref_text_path, 'w', encoding='utf-8') as f:
                f.write(description)

            # Now use it as a clone reference
            for i, text in enumerate(texts):
                try:
                    wav = self.model.clone_voice(
                        text=text,
                        ref_audio=ref_path,
                        ref_text=description,
                        language=self.config.get(
                            "qwen3_language", "English")
                    )
                    if self._save_wav(wav, 24000, output_paths[i]):
                        success_count += 1
                    else:
                        errors.append(f"Save failed for slide {i + 1}")
                except Exception as e:
                    errors.append(str(e))

        except Exception as e:
            return False, [f"Voice design failed: {e}"]

        return (True, []) if success_count == len(texts) else (False, errors)


# =========================================================================
# REGISTRY
# =========================================================================
BACKEND_REGISTRY = {
    'edge_tts': {
        'name': 'Edge TTS (Online)',
        'class': EdgeTTSBackend,
    },
    'qwen3': {
        'name': 'Qwen3 TTS (Local)',
        'class': Qwen3Backend,
    },
}