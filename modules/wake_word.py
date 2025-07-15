import pvporcupine
import pyaudio
import struct
import threading
import logging
from typing import Callable, List, Optional


class WakeWordDetector:
    """Porcupine wake-word detector that runs in a background thread.

    Parameters
    ----------
    keyword_paths : List[str]
        List of .ppn custom keyword model paths.
    sensitivities : Optional[List[float]]
        Sensitivity for each keyword (0-1).  If None, 0.5 is used for all.
    audio_device_index : Optional[int]
        Index of the input device for PyAudio.  If None, default device is used.
    detected_callback : Callable[[], None]
        Function invoked when the keyword is detected.
    """

    def __init__(self,
                 keyword_paths: List[str],
                 sensitivities: Optional[List[float]] = None,
                 audio_device_index: Optional[int] = None,
                 detected_callback: Optional[Callable[[], None]] = None,
                 access_key: Optional[str] = None):
        if not keyword_paths:
            raise ValueError("keyword_paths must contain at least one .ppn file")

        self.logger = logging.getLogger(__name__)
        self._keyword_paths = keyword_paths
        self._sensitivities = sensitivities or [0.5] * len(keyword_paths)
        self._audio_device_index = audio_device_index
        self._detected_callback = detected_callback or (lambda: None)
        self._access_key = access_key

        # Internal state
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._porcupine = None
        self._pa = None
        self._pa_stream = None

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self.logger.info("Wake-word detector started")

    def stop(self):
        """Signal the detector thread to stop and wait for it to finish.

        The heavy resource cleanup is handled in the thread's *finally* block,
        so we avoid calling _cleanup() here to prevent double-free situations
        (Porcupine.delete() is *not* idempotent).
        """
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def _run(self):
        try:
            kwargs = {
                "keyword_paths": self._keyword_paths,
                "sensitivities": self._sensitivities,
            }
            if self._access_key:
                kwargs["access_key"] = self._access_key

            self._porcupine = pvporcupine.create(**kwargs)
            self._pa = pyaudio.PyAudio()
            self._pa_stream = self._pa.open(
                rate=self._porcupine.sample_rate,
                channels=1,
                format=pyaudio.paInt16,
                input=True,
                frames_per_buffer=self._porcupine.frame_length,
                input_device_index=self._audio_device_index,
            )

            while self._running:
                pcm = self._pa_stream.read(self._porcupine.frame_length, exception_on_overflow=False)
                pcm_unpacked = struct.unpack_from("%dh" % self._porcupine.frame_length, pcm)
                result = self._porcupine.process(pcm_unpacked)
                if result >= 0:
                    # Keyword detected
                    self.logger.info("Wake-word detected (index %d)", result)
                    try:
                        self._detected_callback()
                    except Exception as exc:
                        self.logger.error("Wake-word callback raised: %s", exc)
        except Exception as exc:
            self.logger.error("Wake-word detector error: %s", exc)
        finally:
            self._cleanup()

    def _cleanup(self):
        try:
            if self._pa_stream is not None:
                self._pa_stream.stop_stream()
                self._pa_stream.close()
                self._pa_stream = None

            if self._pa is not None:
                self._pa.terminate()
                self._pa = None

            if self._porcupine is not None:
                # .delete() frees native memory; calling it twice is UB.
                self._porcupine.delete()
                self._porcupine = None
        except Exception:
            pass 