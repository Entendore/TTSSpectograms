# worker.py
import os
import tempfile
import subprocess
import traceback
import soundfile as sf
import numpy as np
from PySide6.QtCore import QThread, Signal
from typing import List, Dict

from renderer import SpectrogramProcessor, VideoRenderer
from utils import logger


class RenderWorker(QThread):
    progress = Signal(int)
    status = Signal(str)
    log = Signal(str)
    error = Signal(str)
    batch_error = Signal(str, str)
    # FIX: renamed from 'finished' to avoid shadowing QThread.finished
    render_finished = Signal()
    file_started = Signal(str)
    file_finished = Signal(str)

    def __init__(self, file_list, output_target, settings):
        super().__init__()
        self.file_list = file_list
        self.output_target = output_target
        self.settings = settings
        self.is_cancelled = False
        self.processor = SpectrogramProcessor(settings)
        self.renderer = VideoRenderer(settings)

        if os.path.isfile(output_target):
            base_dir = os.path.dirname(output_target)
        else:
            base_dir = output_target
        self.local_temp_dir = os.path.join(
            base_dir if base_dir else os.getcwd(), "_temp_proc")
        try:
            os.makedirs(self.local_temp_dir, exist_ok=True)
        except OSError:
            self.local_temp_dir = None

    def run(self):
        try:
            if self.settings.get('merge_mode', False):
                self.process_concatenated()
            else:
                self.process_batch()
        except Exception as e:
            self.error.emit(f"Fatal Error:\n{traceback.format_exc()}")
        finally:
            self.progress.emit(100)
            self.render_finished.emit()

    def cancel(self):
        self.is_cancelled = True

    def _get_temp_dir(self):
        return self.local_temp_dir or tempfile.gettempdir()

    def _build_output_path(self, base_name, out_dir, index):
        suffix = self.settings.get('filename_suffix', '').strip()
        name = base_name
        if suffix:
            name = f"{name}{suffix}"
        out = os.path.join(out_dir, f"{name}.mp4")
        if os.path.exists(out):
            out = os.path.join(out_dir, f"{name}_{index}.mp4")
        counter = 1
        final_out = out
        while os.path.exists(final_out):
            stem, ext = os.path.splitext(out)
            final_out = f"{stem}_v{counter}{ext}"
            counter += 1
        return final_out

    def _cleanup_temp(self, *paths):
        for tmp in paths:
            try:
                if tmp and os.path.exists(tmp):
                    os.unlink(tmp)
            except OSError:
                pass

    def process_concatenated(self):
        self.log.emit("Merging files...")
        valid_files, total_dur, seg_info = [], 0.0, []
        for fpath in self.file_list:
            if self.is_cancelled:
                return
            try:
                info = sf.info(fpath)
                if info.duration > 0:
                    seg_info.append({
                        'start': total_dur,
                        'end': total_dur + info.duration,
                        'name': os.path.basename(fpath)
                    })
                    valid_files.append(fpath)
                    total_dur += info.duration
            except Exception:
                self.batch_error.emit(fpath, "Read Error")

        if not valid_files:
            self.log.emit("No valid files to merge.")
            return

        temp_base = self._get_temp_dir()
        list_path = os.path.join(temp_base, "concat.txt")
        temp_wav = os.path.join(temp_base, "merged.wav")
        temp_proc_wav = os.path.join(temp_base, "merged_processed.wav")

        try:
            with open(list_path, 'w', encoding='utf-8') as wf:
                for p in valid_files:
                    escaped = p.replace('\\', '/').replace("'", "'\\''")
                    wf.write(f"file '{escaped}'\n")

            cmd = [
                'ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', list_path,
                '-ac', '1', '-ar', '44100', '-c:a', 'pcm_f32le', temp_wav
            ]
            result = subprocess.run(cmd, stderr=subprocess.PIPE)
            if result.returncode != 0:
                err_msg = result.stderr.decode('utf-8', errors='replace')[-500:]
                raise Exception(f"FFmpeg concat failed: {err_msg}")

            y, sr = sf.read(temp_wav, dtype='float32')
            dur = len(y) / sr
            if dur == 0:
                raise Exception("Merged audio is empty")

            if self.settings.get('preview_mode', False):
                y = y[:int(min(15, dur) * sr)]
                dur = len(y) / sr

            y_proc = self.processor.process_audio(y, sr)
            sf.write(temp_proc_wav, y_proc, sr)

            spec, _ = self.processor.generate_spectrogram_image(y_proc, sr, dur)

            out = (self.output_target if os.path.isfile(self.output_target)
                   else os.path.join(self.output_target, "Merged.mp4"))

            self.log.emit(f"Rendering merged video ({dur:.1f}s)...")

            self.renderer.render_video_ffmpeg(
                temp_proc_wav, out, spec, y_proc, sr, dur, seg_info,
                lambda p, s: (self.progress.emit(p), self.status.emit(s)),
                lambda: self.is_cancelled
            )

            if not self.is_cancelled:
                self.log.emit(f"Merged output saved: {out}")

        finally:
            self._cleanup_temp(list_path, temp_wav, temp_proc_wav)

    def process_batch(self):
        out_dir = self.output_target
        if not os.path.isdir(out_dir):
            out_dir = os.path.dirname(out_dir) or os.getcwd()
            try:
                os.makedirs(out_dir, exist_ok=True)
            except OSError:
                out_dir = os.getcwd()

        total = len(self.file_list)
        for i, fpath in enumerate(self.file_list):
            if self.is_cancelled:
                return

            self.file_started.emit(fpath)
            self.log.emit(
                f"Processing ({i + 1}/{total}): {os.path.basename(fpath)}")

            temp_proc_wav = None
            try:
                y, sr = sf.read(fpath, dtype='float32')
                if y.ndim > 1:
                    y = y.mean(axis=1)
                dur = len(y) / sr
                if dur == 0:
                    raise Exception("Empty audio")

                y_proc = self.processor.process_audio(y, sr)

                if self.settings.get('preview_mode', False):
                    preview_dur = min(15, dur)
                    y_proc = y_proc[:int(preview_dur * sr)]
                    dur = len(y_proc) / sr

                temp_proc_wav = tempfile.NamedTemporaryFile(
                    suffix=".wav", delete=False,
                    dir=self._get_temp_dir()).name
                sf.write(temp_proc_wav, y_proc, sr)

                spec, _ = self.processor.generate_spectrogram_image(
                    y_proc, sr, dur)

                base_name = os.path.splitext(os.path.basename(fpath))[0]
                out = self._build_output_path(base_name, out_dir, i)

                self.log.emit(f"Rendering {base_name} ({dur:.1f}s)...")

                self.renderer.render_video_ffmpeg(
                    temp_proc_wav, out, spec, y_proc, sr, dur,
                    [{'start': 0, 'end': dur}],
                    lambda p, s: (self.progress.emit(p), self.status.emit(s)),
                    lambda: self.is_cancelled
                )

                if not self.is_cancelled:
                    self.file_finished.emit(fpath)
                    self.log.emit(f"Finished: {os.path.basename(out)}")

            except Exception as e:
                self.batch_error.emit(fpath, str(e))
            finally:
                self._cleanup_temp(temp_proc_wav)

            if not self.is_cancelled and total > 0:
                batch_pct = int(((i + 1) / total) * 100)
                self.progress.emit(batch_pct)

        try:
            if self.local_temp_dir and os.path.isdir(self.local_temp_dir):
                remaining = os.listdir(self.local_temp_dir)
                if not remaining:
                    os.rmdir(self.local_temp_dir)
        except OSError:
            pass


class TTSWorker(QThread):
    """Background worker for TTS generation."""
    progress = Signal(int)
    log = Signal(str)
    error = Signal(str)
    # FIX: renamed from 'finished' to avoid shadowing QThread.finished
    tts_finished = Signal(str)

    def __init__(self, backend, text: str, output_path: str):
        super().__init__()
        self.backend = backend
        self.text = text
        self.output_path = output_path
        self.is_cancelled = False

    def run(self):
        try:
            self.log.emit("Initializing TTS backend...")
            self.backend.initialize()
            self.log.emit("Generating speech...")
            success, errors = self.backend.generate_batch(
                [self.text], [self.output_path])
            if success and os.path.exists(self.output_path):
                self.log.emit("Speech generated successfully!")
                self.tts_finished.emit(self.output_path)
            else:
                self.error.emit(
                    "\n".join(errors) if errors else "Unknown generation error")
        except Exception as e:
            self.error.emit(str(e))
        finally:
            try:
                self.backend.cleanup()
            except Exception:
                pass

    def cancel(self):
        self.is_cancelled = True


# =========================================================================
# COMBINED TTS + VIDEO RENDER WORKER
# =========================================================================
class TTSRenderWorker(QThread):
    """
    One-shot pipeline: Text → TTS Audio → Spectrogram Video.
    Generates speech from text, then immediately renders a spectrogram
    video for the generated audio, all in a single background thread.
    """
    progress = Signal(int)
    status = Signal(str)
    log = Signal(str)
    error = Signal(str)
    # Emits (audio_path, video_path) on success
    tts_render_finished = Signal(str, str)

    def __init__(self, backend, text: str, audio_output_path: str,
                 video_output_path: str, render_settings: Dict):
        super().__init__()
        self.backend = backend
        self.text = text
        self.audio_output_path = audio_output_path
        self.video_output_path = video_output_path
        self.render_settings = render_settings
        self.is_cancelled = False

    def run(self):
        temp_proc_wav = None
        try:
            # ── Phase 1: TTS Generation (0% → 40%) ──
            self.status.emit("Phase 1/2: Generating speech…")
            self.log.emit("Initializing TTS backend…")
            self.progress.emit(2)

            self.backend.initialize()
            self.progress.emit(10)

            self.log.emit("Generating speech from text…")
            success, errors = self.backend.generate_batch(
                [self.text], [self.audio_output_path])

            if not success:
                msg = "\n".join(errors) if errors else "TTS generation failed"
                self.error.emit(msg)
                return

            if not os.path.exists(self.audio_output_path):
                self.error.emit("TTS output file was not created")
                return

            self.progress.emit(40)
            self.log.emit("Speech generated successfully!")
            self.status.emit("Phase 1/2: Speech done ✓")

            if self.is_cancelled:
                return

            # ── Phase 2: Video Rendering (40% → 100%) ──
            self.status.emit("Phase 2/2: Rendering spectrogram video…")
            self.log.emit("Loading generated audio…")
            self.progress.emit(42)

            y, sr = sf.read(self.audio_output_path, dtype='float32')
            if y.ndim > 1:
                y = y.mean(axis=1)
            dur = len(y) / sr

            if dur == 0:
                self.error.emit("Generated audio is empty (0 seconds)")
                return

            self.log.emit(f"Audio duration: {dur:.1f}s — processing…")
            self.progress.emit(45)

            processor = SpectrogramProcessor(self.render_settings)
            renderer = VideoRenderer(self.render_settings)

            y_proc = processor.process_audio(y, sr)

            # Write processed audio to temp file for FFmpeg muxing
            audio_dir = os.path.dirname(self.audio_output_path)
            temp_proc_wav = os.path.join(audio_dir, "_temp_tts_render.wav")
            sf.write(temp_proc_wav, y_proc, sr)
            self.progress.emit(50)

            spec, _ = processor.generate_spectrogram_image(y_proc, sr, dur)
            self.progress.emit(55)

            self.log.emit(
                f"Rendering video ({dur:.1f}s) → "
                f"{os.path.basename(self.video_output_path)}…")

            seg_info = [{'start': 0, 'end': dur}]

            # Scale inner progress (0-100 from renderer) to outer (55-100)
            def on_render_progress(p, s):
                outer = int(55 + p * 0.45)
                self.progress.emit(outer)
                self.status.emit(f"Phase 2/2: {s}")

            renderer.render_video_ffmpeg(
                temp_proc_wav, self.video_output_path,
                spec, y_proc, sr, dur, seg_info,
                on_render_progress,
                lambda: self.is_cancelled
            )

            if not self.is_cancelled:
                self.progress.emit(100)
                self.status.emit("Done ✓")
                self.log.emit(
                    f"Pipeline complete!\n"
                    f"  Audio: {self.audio_output_path}\n"
                    f"  Video: {self.video_output_path}")
                self.tts_render_finished.emit(
                    self.audio_output_path, self.video_output_path)

        except Exception as e:
            self.error.emit(f"Pipeline error: {e}")
        finally:
            # Cleanup
            try:
                self.backend.cleanup()
            except Exception:
                pass
            if temp_proc_wav:
                try:
                    os.unlink(temp_proc_wav)
                except OSError:
                    pass

    def cancel(self):
        self.is_cancelled = True