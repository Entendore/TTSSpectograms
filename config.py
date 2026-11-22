# config.py
import os
import json
from typing import Dict, Any
from PySide6.QtCore import QStandardPaths

CONFIG_FILE_NAME = "spectrogram_settings.json"
AUDIO_EXTS = ('.wav', '.ogg', '.mp3', '.flac', '.aac', '.m4a')

# Folder names relative to CWD (TTSSpectrograms)
FOLDER_INPUT = "Input"
FOLDER_OUTPUT = "Output"
FOLDER_TTS_OUTPUT = "TTSOutput"

DEFAULT_SETTINGS = {
    'auto_highpass': True,
    'highpass_freq': 20,
    'resolution': 1,
    'fps': 30,
    'gpu': True,
    'merge': False,
    'suffix': '',
    'colormap': 'Vibrant (Turbo)',
    'color_mode': 'Static',
    'overlay': True,
    'window': 5.0,
    'auto_freq': True,
    'min_freq': 20,
    'max_freq': 20000,
    'dyn_range': 80,
    'gamma': 1.0,
    'blur': 0.0,
    'normalize': True,
    'bass': 0.0,
    'treble': 0.0,
    'last_input_dir': '',
    'default_load_folder': '',
    'visual_mode': 'Standard',
    'spec_type': 'Standard (STFT)',
    'n_fft': 4096,
    'hop_length': 512,
    'sharpen': True,
    'interpolation': 'Lanczos',
    'center_line': True,
    'analyzer_color_mode': 'Match Video',
    'line_base_width': 6,
    'line_magnitude_sensitivity': 800.0,
    'line_glow_intensity': 15.0,
    'line_bloom': 8.0,
    'analyzer_bar_count': 64,
    'analyzer_position': 'Center',
    'analyzer_show_center_line': True,
    'palette_folder': '',
    # TTS settings
    'tts_backend': 'edge_tts',
    'edge_language': 'English',
    'edge_voice': 'en-US-JennyNeural',
    'edge_rate': '+0%',
    'edge_pitch': '+0Hz',
    'edge_volume': '+0%',
    'qwen3_mode': 'voice_custom',
    'qwen3_speaker': 'Vivian',
    'qwen3_instruct': '',
    'qwen3_language': 'English',
    'qwen3_ref_audio': '',
    'qwen3_ref_text': '',
    'qwen3_vd_description': '',
    'qwen3_vd_save_name': 'my_designed_voice',
    'qwen3_device_map': 'cuda:0',
    'qwen3_size': '1.7B',
    'qwen3_attn_implementation': 'flash_attention_2',
    'qwen3_dtype': 'float16',
    'enable_text_chunking': True,
    'chunk_max_chars': 500,
    'chunk_max_sentences': 5,
    'chunk_min_chars': 50,
    'ffmpeg_path': 'ffmpeg',
    'voice_references_root': 'voicereferences',
    'hf_cache_dir': '',
}


def get_cwd_subfolder(folder_name: str) -> str:
    """Return an absolute path to a subfolder of CWD, creating it if needed."""
    path = os.path.join(os.getcwd(), folder_name)
    os.makedirs(path, exist_ok=True)
    return path


def get_default_load_folder() -> str:
    """Return the Input folder under CWD, falling back to standard locations."""
    cwd_input = get_cwd_subfolder(FOLDER_INPUT)
    if cwd_input and os.path.isdir(cwd_input):
        return cwd_input
    music = QStandardPaths.writableLocation(QStandardPaths.MusicLocation)
    if music and os.path.isdir(music):
        return music
    home = QStandardPaths.writableLocation(QStandardPaths.HomeLocation)
    if home and os.path.isdir(home):
        return home
    return os.path.expanduser("~")


class ConfigManager:
    def __init__(self):
        config_dir = QStandardPaths.writableLocation(QStandardPaths.AppDataLocation)
        os.makedirs(config_dir, exist_ok=True)
        self.path = os.path.join(config_dir, CONFIG_FILE_NAME)

    def load(self) -> Dict[str, Any]:
        if not os.path.exists(self.path):
            return {'settings': DEFAULT_SETTINGS}
        try:
            with open(self.path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if 'settings' not in data:
                    data['settings'] = {}
                for key, value in DEFAULT_SETTINGS.items():
                    if key not in data['settings']:
                        data['settings'][key] = value
                return data
        except (json.JSONDecodeError, IOError):
            return {'settings': DEFAULT_SETTINGS}

    def save(self, config: Dict[str, Any]):
        try:
            with open(self.path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4)
        except IOError as e:
            print(f"Error saving settings: {e}")


class SettingsProxy:
    """Dict wrapper with .get()/.set() for TTS backend UI widgets."""

    def __init__(self, data: dict, on_change=None):
        self._data = data
        self._on_change = on_change

    def get(self, key, default=None):
        return self._data.get(key, default)

    def set(self, key, value):
        self._data[key] = value
        if self._on_change:
            self._on_change(key, value)

    def raw(self) -> dict:
        return self._data