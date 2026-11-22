# renderer.py
# Fixed: Added actual Base Width, Glow, and Bloom logic to the visualizer overlay.

import os
import time
import subprocess
import threading
import numpy as np
import librosa
import cv2
from typing import Dict, Tuple, Callable, Optional

try:
    from scipy import signal as scipy_signal
except ImportError:
    scipy_signal = None

try:
    from numba import cuda
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


class StreamReader(threading.Thread):
    def __init__(self, stream):
        super().__init__()
        self.stream = stream
        self.output = []
        self.daemon = True

    def run(self):
        try:
            for line in iter(self.stream.readline, b''):
                self.output.append(line.decode('utf-8', errors='replace'))
        finally:
            self.stream.close()

    def get_output(self) -> str:
        return ''.join(self.output)


class VideoEncoder:
    def __init__(self, target_w: int, target_h: int, fps: int,
                 audio_path: str, output_path: str, use_gpu: bool):
        self.target_w = target_w
        self.target_h = target_h
        self.fps = fps
        self.audio_path = audio_path
        self.output_path = output_path
        self.use_gpu = use_gpu
        self.process = None
        self.stderr_reader = None

    def _build_cmd(self):
        if self.use_gpu:
            codec, preset = 'h264_nvenc', 'p7'
            extra_flags = [
                '-rc', 'vbr', '-cq', '18', '-b:v', '15M', '-maxrate', '20M',
                '-profile:v', 'high', '-movflags', '+faststart'
            ]
        else:
            codec, preset = 'libx264', 'medium'
            extra_flags = [
                '-crf', '20', '-profile:v', 'high', '-level', '4.2',
                '-movflags', '+faststart'
            ]
        return [
            'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-s', f'{self.target_w}x{self.target_h}', '-pix_fmt', 'rgb24',
            '-r', str(self.fps), '-i', '-',
            '-i', self.audio_path,
            '-map', '0:v:0', '-map', '1:a:0',
            '-c:v', codec, '-preset', preset,
            *extra_flags,
            '-pix_fmt', 'yuv420p',
            '-c:a', 'aac', '-b:a', '320k',
            '-shortest', self.output_path
        ]

    def start(self):
        cmd = self._build_cmd()
        self.process = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE,
            stdout=subprocess.PIPE
        )
        self.stderr_reader = StreamReader(self.process.stderr)
        self.stderr_reader.start()

    def write(self, data: bytes) -> bool:
        try:
            self.process.stdin.write(data)
            return True
        except (BrokenPipeError, OSError):
            return False

    def finish(self) -> str:
        try:
            if self.process.stdin:
                self.process.stdin.close()
        except Exception:
            pass
        try:
            self.process.wait(timeout=300)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            self.process.wait()
        if self.stderr_reader:
            self.stderr_reader.join(timeout=5)
            return self.stderr_reader.get_output()
        return ""


class SpectrogramProcessor:
    def __init__(self, settings: Dict):
        self.settings = settings
        self._filter_cache = {}
        self._remap_cache = {}
        self._gamma_lut = None
        self._gamma_cache_key = None

    def process_audio(self, y: np.ndarray, sr: int) -> np.ndarray:
        if len(y) == 0:
            return y

        hp_freq = 0
        if self.settings.get('auto_highpass', True):
            hp_freq = self._detect_sub_bass_noise(y, sr)
        else:
            hp_freq = self.settings.get('highpass_freq', 20)

        if scipy_signal and hp_freq > 0 and hp_freq < sr / 2:
            try:
                b, a = scipy_signal.butter(
                    5, hp_freq / (sr / 2), btype='high', analog=False)
                y = scipy_signal.filtfilt(b, a, y)
            except Exception:
                pass

        if self.settings.get('normalize', True):
            peak = np.max(np.abs(y))
            if peak > 1e-5:
                y = librosa.util.normalize(y)

        if scipy_signal:
            if self.settings.get('bass_gain', 0) != 0:
                b, a = self._get_filter_coefficients(
                    sr, 200, self.settings['bass_gain'], 'low')
                y = scipy_signal.lfilter(b, a, y)
            if self.settings.get('treble_gain', 0) != 0:
                b, a = self._get_filter_coefficients(
                    sr, 4000, self.settings['treble_gain'], 'high')
                y = scipy_signal.lfilter(b, a, y)
        return y

    def _detect_sub_bass_noise(self, y: np.ndarray, sr: int) -> float:
        try:
            analysis_len = int(min(len(y), sr * 5))
            y_analysis = y[:analysis_len]
            n_fft = 4096
            D = np.abs(librosa.stft(y_analysis, n_fft=n_fft, hop_length=1024))
            freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

            sub_band_indices = np.where((freqs >= 0) & (freqs <= 60))[0]
            if len(sub_band_indices) == 0:
                return 20.0

            sub_band_energy = D[sub_band_indices, :]
            mean_energy = np.mean(sub_band_energy, axis=0) + 1e-10
            geo_mean_energy = np.exp(
                np.mean(np.log(sub_band_energy + 1e-10), axis=0))
            flatness = np.mean(geo_mean_energy / mean_energy)

            if flatness > 0.6:
                return 40.0
            elif flatness > 0.35:
                idx_20hz = np.argmin(np.abs(freqs - 20))
                idx_40hz = np.argmin(np.abs(freqs - 40))
                if np.mean(D[idx_20hz, :]) > (np.mean(D[idx_40hz, :]) * 0.8):
                    return 30.0
                return 20.0
            else:
                return 15.0
        except Exception:
            return 20.0

    def _get_filter_coefficients(self, sr, freq, gain_db, ftype):
        key = (sr, freq, gain_db, ftype)
        if key not in self._filter_cache:
            self._filter_cache[key] = self._design_shelf_filter(
                sr, freq, gain_db, ftype)
        return self._filter_cache[key]

    def _design_shelf_filter(self, sr, freq, gain_db, ftype):
        if sr <= 0 or freq <= 0 or freq >= sr / 2:
            return [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]
        A = 10 ** (gain_db / 40.0)
        w0 = 2 * np.pi * freq / sr
        beta = np.sqrt(A + (1 / A))

        if ftype == 'low':
            b0 = A * ((A + 1) - (A - 1) * np.cos(w0) + beta * np.sin(w0))
            b1 = 2 * A * ((A - 1) - (A + 1) * np.cos(w0))
            b2 = A * ((A + 1) - (A - 1) * np.cos(w0) - beta * np.sin(w0))
            a0 = (A + 1) + (A - 1) * np.cos(w0) + beta * np.sin(w0)
            a1 = -2 * ((A - 1) + (A + 1) * np.cos(w0))
            a2 = (A + 1) + (A - 1) * np.cos(w0) - beta * np.sin(w0)
        else:
            b0 = A * ((A + 1) + (A - 1) * np.cos(w0) + beta * np.sin(w0))
            b1 = -2 * A * ((A - 1) + (A + 1) * np.cos(w0))
            b2 = A * ((A + 1) + (A - 1) * np.cos(w0) - beta * np.sin(w0))
            a0 = (A + 1) - (A - 1) * np.cos(w0) + beta * np.sin(w0)
            a1 = 2 * ((A - 1) - (A + 1) * np.cos(w0))
            a2 = (A + 1) - (A - 1) * np.cos(w0) - beta * np.sin(w0)

        return [b0/a0, b1/a0, b2/a0], [1.0, a1/a0, a2/a0]

    def _get_gamma_lut(self, gamma: float):
        if self._gamma_cache_key != gamma or self._gamma_lut is None:
            inv_gamma = 1.0 / gamma
            table = np.array(
                [((i / 255.0) ** inv_gamma) * 255 for i in range(256)],
                dtype=np.uint8)
            self._gamma_lut = table
            self._gamma_cache_key = gamma
        return self._gamma_lut

    def generate_spectrogram_image(self, y: np.ndarray, sr: int,
                                   duration: float) -> Tuple[np.ndarray, int]:
        spec_type = self.settings.get('spec_type', 'stft')
        n_fft = self.settings.get('n_fft', 4096)
        hop_length = self.settings.get('hop_length', 512)
        dyn_range = self.settings.get('dynamic_range_db', 80)
        min_freq = self.settings.get('min_freq', 20.0)
        max_freq = self.settings.get('max_freq', 20000.0)

        S_db = None
        if spec_type == 'mel':
            n_mels = min(256, self.settings.get('height', 1080))
            S = librosa.feature.melspectrogram(
                y=y, sr=sr, n_mels=n_mels, n_fft=n_fft,
                hop_length=hop_length, fmin=min_freq, fmax=max_freq)
            S_db = librosa.power_to_db(S, ref=np.max)
            if self.settings.get('auto_freq', True):
                min_freq, max_freq = self._detect_bounds_from_mel(
                    S_db, sr, n_mels, min_freq, max_freq)
        elif spec_type == 'cqt':
            S = np.abs(librosa.cqt(
                y, sr=sr, hop_length=hop_length,
                fmin=librosa.note_to_hz('C1'),
                n_bins=84, bins_per_octave=12))
            S_db = librosa.amplitude_to_db(S, ref=np.max)
            if self.settings.get('auto_freq', True):
                min_freq, max_freq = self._detect_bounds_from_cqt(S_db, sr)
        else:  # stft
            D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
            S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
            if self.settings.get('auto_freq', True):
                min_freq, max_freq = self._detect_bounds_from_stft(
                    S_db, sr, n_fft)

        S_norm_float = (S_db + dyn_range) * (255.0 / dyn_range)
        gray_img = np.clip(S_norm_float, 0, 255).astype(np.uint8)

        gamma = self.settings.get('gamma', 1.0)
        if gamma != 1.0:
            gray_img = cv2.LUT(gray_img, self._get_gamma_lut(gamma))

        vis_mode = self.settings.get('visual_mode', 'standard')
        if vis_mode == 'radial':
            log_img = self._create_radial(
                gray_img, sr, n_fft, spec_type, min_freq, max_freq)
        elif vis_mode == 'classic':
            log_img = self._create_linear(gray_img)
        else:
            if spec_type == 'stft':
                log_img = self._create_log_freq(
                    gray_img, sr, n_fft, min_freq, max_freq)
            else:
                log_img = self._create_linear(gray_img)

        if self.settings.get('sharpen', True):
            gaussian = cv2.GaussianBlur(log_img, (0, 0), 3.0)
            log_img = cv2.addWeighted(log_img, 1.5, gaussian, -0.5, 0)

        return log_img, hop_length

    def _detect_bounds_from_stft(self, S_db, sr, n_fft):
        threshold_db = -60.0
        max_db = np.max(S_db)
        if not np.isfinite(max_db):
            return 20.0, float(sr / 2)
        freq_maxes = np.max(S_db, axis=1)
        freq_mask = freq_maxes > (max_db + threshold_db)
        if not np.any(freq_mask):
            return 20.0, float(sr / 2)
        indices = np.where(freq_mask)[0]
        bin_freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        return (
            float(bin_freqs[max(0, indices[0] - 2)]),
            float(bin_freqs[min(len(freq_mask) - 1, indices[-1] + 2)])
        )

    def _detect_bounds_from_mel(self, S_db, sr, n_mels, fmin, fmax):
        threshold_db = -60.0
        max_db = np.max(S_db)
        if not np.isfinite(max_db):
            return fmin, fmax
        freq_maxes = np.max(S_db, axis=1)
        freq_mask = freq_maxes > (max_db + threshold_db)
        if not np.any(freq_mask):
            return fmin, fmax
        indices = np.where(freq_mask)[0]
        min_mel = librosa.hz_to_mel(fmin)
        max_mel = librosa.hz_to_mel(fmax)
        mels = np.linspace(min_mel, max_mel, n_mels + 2)
        min_idx = max(0, indices[0])
        max_idx = min(n_mels - 1, indices[-1])
        return (
            max(20.0, librosa.mel_to_hz(mels[min_idx])),
            min(float(sr/2), librosa.mel_to_hz(mels[max_idx + 2]))
        )

    def _detect_bounds_from_cqt(self, S_db, sr):
        threshold_db = -60.0
        max_db = np.max(S_db)
        if not np.isfinite(max_db):
            return 20.0, float(sr / 2)
        freq_maxes = np.max(S_db, axis=1)
        freq_mask = freq_maxes > (max_db + threshold_db)
        if not np.any(freq_mask):
            return 20.0, float(sr / 2)
        indices = np.where(freq_mask)[0]
        freqs = librosa.cqt_frequencies(
            S_db.shape[0], fmin=librosa.note_to_hz('C1'))
        return (
            float(freqs[max(0, indices[0])]),
            float(freqs[min(len(freqs) - 1, indices[-1])])
        )

    def _create_linear(self, img):
        target_h = self.settings.get('height', 1080)
        h, w = img.shape[:2]
        if w > 15000:
            img = cv2.resize(img, (15000, h), interpolation=cv2.INTER_AREA)
        return cv2.resize(img, (img.shape[1], target_h),
                          interpolation=cv2.INTER_LANCZOS4)

    def _create_log_freq(self, img, sr, n_fft, min_freq, max_freq):
        target_h = self.settings.get('height', 1080)

        cache_key = (sr, n_fft, min_freq, max_freq, target_h, img.shape[1])
        if cache_key in self._remap_cache:
            map_x_grid, map_y_grid = self._remap_cache[cache_key]
            if map_x_grid.shape[1] == img.shape[1]:
                return cv2.remap(img, map_x_grid, map_y_grid,
                                 interpolation=cv2.INTER_LANCZOS4)

        if img.shape[1] > 20000:
            img = cv2.resize(img, (20000, img.shape[0]),
                             interpolation=cv2.INTER_AREA)

        out_freqs = np.logspace(
            np.log10(max(1, min_freq)), np.log10(max_freq), target_h)
        bin_freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        y_indices = np.interp(
            out_freqs, bin_freqs, np.arange(len(bin_freqs))).astype(np.float32)

        map_x = np.arange(img.shape[1], dtype=np.float32)
        map_x_grid, map_y_grid = np.meshgrid(map_x, y_indices)
        self._remap_cache[cache_key] = (map_x_grid, map_y_grid)

        log_img = cv2.remap(img, map_x_grid, map_y_grid,
                            interpolation=cv2.INTER_LANCZOS4)
        if log_img.shape[1] > 16000:
            log_img = cv2.resize(log_img, (16000, target_h),
                                 interpolation=cv2.INTER_AREA)
        return log_img

    def _create_radial(self, img, sr, n_fft, spec_type, min_freq, max_freq):
        dim = min(self.settings.get('width', 1920),
                  self.settings.get('height', 1080))
        if dim < 100:
            dim = 800
        nyquist = sr / 2

        if spec_type == 'stft':
            bin_freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        elif spec_type == 'mel':
            bin_freqs = np.linspace(0, nyquist, img.shape[0])
        else:
            bin_freqs = librosa.cqt_frequencies(
                n_bins=84, fmin=librosa.note_to_hz('C1'))

        out_freqs = np.logspace(
            np.log10(max(1, min_freq)), np.log10(max_freq), dim // 2)
        y_indices = np.interp(
            out_freqs, bin_freqs, np.arange(len(bin_freqs))).astype(np.float32)

        if img.shape[1] > 20000:
            img = cv2.resize(img, (20000, img.shape[0]),
                             interpolation=cv2.INTER_AREA)

        map_x = np.arange(img.shape[1], dtype=np.float32)
        map_x_grid, map_y_grid = np.meshgrid(map_x, y_indices)
        src_log_freq = cv2.remap(img, map_x_grid, map_y_grid,
                                 interpolation=cv2.INTER_LANCZOS4)
        src_polar = cv2.resize(src_log_freq, (3600, dim // 2),
                               interpolation=cv2.INTER_LANCZOS4)

        center = (dim // 2, dim // 2)
        max_radius = dim // 2
        flags = (cv2.WARP_POLAR_LINEAR + cv2.WARP_INVERSE_MAP
                 + cv2.INTER_LANCZOS4)
        return cv2.warpPolar(
            src_polar, (dim, dim), center, max_radius, flags)


class VideoRenderer:
    def __init__(self, settings: Dict):
        self.settings = settings
        self.luts = settings.get('luts', {})
        self.use_gpu = CUPY_AVAILABLE and settings.get('use_gpu', False)

        if self.use_gpu:
            try:
                cp.cuda.Device(0).use()
                print("[Renderer] GPU Acceleration Active (CuPy/CUDA)")
            except Exception as e:
                print(f"[Renderer] GPU Init Failed: {e}. CPU fallback.")
                self.use_gpu = False
        else:
            print("[Renderer] Running on CPU")

    def _get_lut(self, name: str) -> np.ndarray:
        lut = self.luts.get(name)
        if lut is not None:
            return lut
        for fb in ['Vibrant (Turbo)', 'Jet (Classic)']:
            if fb in self.luts:
                return self.luts[fb]
        return np.tile(
            np.arange(256, dtype=np.uint8).reshape(256, 1, 1), (1, 1, 3))

    def render_video_ffmpeg(self, audio_path, out_path, spec_img,
                            y_audio, sr, dur, seg_info, progress_cb,
                            cancel_check):
        fps = self.settings['fps']
        window_sec = self.settings['window_size']
        tw, th = self.settings['width'], self.settings['height']

        if tw % 2 != 0:
            tw -= 1
        if th % 2 != 0:
            th -= 1

        total_frames = int(dur * fps)
        if total_frames == 0:
            raise ValueError("Zero duration")

        img_h, img_w = spec_img.shape[:2]
        px_per_sec = img_w / dur
        win_px = (window_sec / 2.0) * px_per_sec

        lut_data = self._get_lut(
            self.settings.get('colormap', 'Vibrant (Turbo)'))
        lut_rgb = cv2.cvtColor(lut_data, cv2.COLOR_BGR2RGB).reshape(256, 3)

        gpu_spec, gpu_lut = None, None
        if self.use_gpu:
            gpu_spec = cp.asarray(spec_img, dtype=cp.float32)
            gpu_lut = cp.asarray(lut_rgb, dtype=cp.float32) / 255.0
            batch_size = 30
        else:
            batch_size = 15

        encoder = VideoEncoder(tw, th, fps, audio_path, out_path, self.use_gpu)
        encoder.start()
        start_time = time.time()

        try:
            for batch_start in range(0, total_frames, batch_size):
                if cancel_check():
                    return

                batch_end = min(batch_start + batch_size, total_frames)

                if self.use_gpu:
                    frames = self._render_batch_gpu(
                        batch_start, batch_end, gpu_spec, gpu_lut,
                        tw, th, img_w, img_h, px_per_sec, win_px)
                    frames = frames.get()
                else:
                    frames = self._render_batch_cpu(
                        batch_start, batch_end, spec_img, lut_rgb,
                        tw, th, img_w, img_h, px_per_sec, win_px)

                for i, f_idx in enumerate(range(batch_start, batch_end)):
                    frame = frames[i]
                    if self.settings.get('draw_overlay', True):
                        self._draw_overlay(frame, f_idx, fps, dur)
                    if self.settings.get('center_line', True):
                        time_sec = f_idx / fps
                        cx_spec = int(time_sec * px_per_sec)
                        cx = max(0, min(img_w - 1, cx_spec))
                        spec_col = spec_img[:, cx]
                        self._draw_vertical_analyzer(frame, spec_col)

                if not encoder.write(frames.tobytes()):
                    raise Exception("FFmpeg process died.")

                if batch_start % (batch_size * 2) == 0:
                    elapsed = time.time() - start_time
                    speed = batch_end / elapsed if elapsed > 0 else 0
                    eta = int((total_frames - batch_end) / speed) if speed > 0 else 0
                    m, s = divmod(eta, 60)
                    progress_cb(
                        int((batch_end / total_frames) * 100),
                        f"Rendering — ETA: {m:02d}:{s:02d}")

        finally:
            err = encoder.finish()
            if cancel_check():
                raise Exception("Cancelled")
            if encoder.process.returncode != 0:
                raise Exception(f"FFmpeg Error:\n{err[-500:]}")

    def _render_batch_cpu(self, start_idx, end_idx, spec_img, lut_rgb,
                          tw, th, sw, sh, pps, win_px):
        count = end_idx - start_idx
        results = np.zeros((count, th, tw, 3), dtype=np.uint8)
        fps = self.settings['fps']

        for i, frame_idx in enumerate(range(start_idx, end_idx)):
            time_sec = frame_idx / fps
            center_x = int(time_sec * pps)

            x1 = int(max(0, center_x - win_px))
            x2 = int(min(sw, center_x + win_px))
            if x2 <= x1:
                x2 = x1 + 1

            img_slice = spec_img[:, x1:x2]
            colored = lut_rgb[img_slice].astype(np.uint8)

            interp = (cv2.INTER_LANCZOS4
                      if self.settings.get('interpolation') == 'lanczos'
                      else cv2.INTER_CUBIC)
            results[i] = cv2.resize(colored, (tw, th), interpolation=interp)
        return results

    def _render_batch_gpu(self, start_idx, end_idx, g_spec, g_lut,
                          tw, th, sw, sh, pps, win_px):
        count = end_idx - start_idx
        g_output = cp.zeros((count, th, tw, 3), dtype=cp.uint8)

        threads_per_block = (8, 8, 1)
        blocks_per_grid = (
            (tw + threads_per_block[0] - 1) // threads_per_block[0],
            (th + threads_per_block[1] - 1) // threads_per_block[1],
            count
        )

        fps = self.settings['fps']

        _cuda_render_kernel[blocks_per_grid, threads_per_block](
            g_spec, g_lut, g_output,
            start_idx, fps,
            tw, th, sw, sh, pps, win_px
        )
        return g_output

    def _draw_overlay(self, frame, frame_idx, fps, dur):
        current_time = frame_idx / fps
        m, s = divmod(int(current_time), 60)
        ms = int((current_time - int(current_time)) * 100)
        text = f"{m:02}:{s:02}.{ms:02}"
        cv2.putText(frame, text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                    cv2.LINE_AA)

    def _draw_vertical_analyzer(self, frame, spec_column):
        h, w = frame.shape[:2]
        position = self.settings.get('analyzer_position', 'Center')
        bar_count = self.settings.get('analyzer_bar_count', 64)
        sensitivity = self.settings.get('line_magnitude_sensitivity', 800.0)
        base_width = self.settings.get('line_base_width', 6)
        glow_intensity = self.settings.get('line_glow_intensity', 15.0)
        bloom = self.settings.get('line_bloom', 8.0)
        show_center_line = self.settings.get('analyzer_show_center_line', True)

        x_old = np.linspace(0, 1, len(spec_column))
        x_new = np.linspace(0, 1, bar_count)
        bar_magnitudes = np.interp(x_new, x_old, spec_column.astype(float))
        bar_h = h / bar_count

        mode = self.settings.get('analyzer_color_mode', 'Match Video')
        if mode == 'Match Video':
            color = frame[h // 2, w // 2].tolist()
        elif mode == 'Neon Gradient':
            color = (255, 0, 255)
        elif mode == 'Sunset Gradient':
            color = (0, 165, 255)
        else:
            color = (255, 255, 255)

        # Create an overlay to apply bloom/glow
        overlay = np.zeros_like(frame, dtype=np.float32)

        for i in range(bar_count):
            mag = bar_magnitudes[i]
            bar_len = int((mag / 255.0) * sensitivity)
            y1 = int(i * bar_h)
            y2 = int((i + 1) * bar_h) - 1

            if position == 'Center':
                x_c = w // 2
                # Base center line
                if show_center_line:
                    cv2.rectangle(overlay,
                                  (x_c - base_width // 2, y1),
                                  (x_c + base_width // 2, y2),
                                  color, -1)
                # Magnitude bars
                if bar_len > base_width:
                    cv2.rectangle(overlay,
                                  (x_c - bar_len // 2, y1),
                                  (x_c + bar_len // 2, y2),
                                  color, -1)
            else:  # Edges
                cv2.rectangle(overlay, (0, y1), (bar_len, y2), color, -1)
                cv2.rectangle(overlay,
                              (w - bar_len, y1), (w, y2), color, -1)

        # Apply Bloom / Glow
        if bloom > 0 or glow_intensity > 0:
            bloom_val = max(1, int(bloom))
            blurred = cv2.GaussianBlur(overlay, (0, 0), bloom_val)
            alpha = min(1.0, glow_intensity / 30.0)
            overlay = cv2.addWeighted(overlay, 1.0, blurred, alpha, 0)

        # Merge overlay with frame
        frame[:] = cv2.add(frame, overlay.astype(np.uint8))


# =========================================================================
# CUDA KERNELS
# =========================================================================
if CUPY_AVAILABLE:
    @cuda.jit
    def _cuda_render_kernel(spec_img, lut, output, start_frame, fps,
                            tw, th, sw, sh, pps, win_px):
        tx = cuda.threadIdx.x
        ty = cuda.threadIdx.y
        bx = cuda.blockIdx.x
        by = cuda.blockIdx.y
        bz = cuda.blockIdx.z

        x = tx + bx * cuda.blockDim.x
        y = ty + by * cuda.blockDim.y
        batch_idx = bz

        if x >= tw or y >= th or batch_idx >= output.shape[0]:
            return

        frame_idx = start_frame + batch_idx
        time_sec = frame_idx / fps

        center_x = time_sec * pps
        norm_x = x / (tw - 1) if tw > 1 else 0.5
        src_x_f = center_x - win_px + (norm_x * 2 * win_px)

        norm_y = y / (th - 1) if th > 1 else 0.5
        src_y_f = norm_y * (sh - 1)

        if src_x_f < 0 or src_x_f >= sw or src_y_f < 0 or src_y_f >= sh:
            output[batch_idx, y, x, 0] = 0
            output[batch_idx, y, x, 1] = 0
            output[batch_idx, y, x, 2] = 0
            return

        x0 = int(src_x_f)
        y0 = int(src_y_f)
        x1 = x0 + 1
        y1 = y0 + 1

        x_w = src_x_f - x0
        y_w = src_y_f - y0

        if x1 >= sw:
            x1 = x0
        if y1 >= sh:
            y1 = y0

        v00 = spec_img[y0, x0]
        v01 = spec_img[y0, x1]
        v10 = spec_img[y1, x0]
        v11 = spec_img[y1, x1]

        val = (v00 * (1 - x_w) * (1 - y_w)
               + v01 * x_w * (1 - y_w)
               + v10 * (1 - x_w) * y_w
               + v11 * x_w * y_w)

        idx = int(val)
        if idx > 255:
            idx = 255
        if idx < 0:
            idx = 0

        output[batch_idx, y, x, 0] = int(lut[idx, 0] * 255)
        output[batch_idx, y, x, 1] = int(lut[idx, 1] * 255)
        output[batch_idx, y, x, 2] = int(lut[idx, 2] * 255)