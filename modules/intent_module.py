#!/usr/bin/env python3
"""intent_module.py
Just there to exist

Example
-------
>>> from modules.intent_module import RhinoIntentRecognizer
>>> def on_intent(inf):
...     print(inf)
>>> rec = RhinoIntentRecognizer(
...     context_path="contexts/smart_lighting_linux.rhn",
...     access_key="<PICOVOICE_ACCESS_KEY>",
...     inference_callback=on_intent)
>>> rec.start()  # will run until rec.stop() is called

The design purposefully keeps the public surface **very small**:
    • start() / stop()
    • is_running
    • context_info / version properties
    • Optional reset() 

"""

from __future__ import annotations

import logging
import threading
import struct
from typing import Callable, Optional, Dict, Any

import pyaudio

try:
    import pvrhino as rhino
    RHINO_AVAILABLE = True
except ImportError:  # pragma: no cover – runtime fallback
    rhino = None  # type: ignore
    RHINO_AVAILABLE = False

logger = logging.getLogger(__name__)


InferenceDict = Dict[str, Any]
InferenceCallback = Callable[[InferenceDict], None]


class RhinoIntentRecognizer:
    """Background microphone listener that emits Rhino inferences.

    Parameters
    ----------
    context_path: str
        Absolute path to the `.rhn` context file.
    access_key: str
        Picovoice *AccessKey* obtained from the console.
    model_path: Optional[str]
        Path to a non-default Rhino model (e.g. for other languages).
    sensitivity: float, default 0.5
        Trade-off between miss rate (0) and false alarms (1).
    endpoint_duration_sec: float, default 1.0
        Amount of trailing silence to detect endpoint.
    require_endpoint: bool, default True
        If *False* Rhino returns an inference even without silence.
    audio_device_index: Optional[int]
        PyAudio device index to use (None = system default).
    inference_callback: Callable[[dict], None]
        Callback invoked **from the background thread** whenever an
        inference is finalised. Receives a dictionary with keys:
            • is_understood: bool
            • intent: str | None
            • slots: dict[str, str]
    """

    def __init__(
        self,
        *,
        context_path: str,
        access_key: str,
        model_path: Optional[str] = None,
        sensitivity: float = 0.5,
        endpoint_duration_sec: float = 1.0,
        require_endpoint: bool = True,
        audio_device_index: Optional[int] = None,
        inference_callback: Optional[InferenceCallback] = None,
    ) -> None:
        if not RHINO_AVAILABLE:
            raise ImportError(
                "pvrhino is not installed. Install with `pip install pvrhino`."
            )

        self._inference_cb = inference_callback or (lambda _: None)
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Create Rhino handle
        logger.info("Initialising Rhino (context=%s)…", context_path)
        self._rhino = rhino.create(
            access_key=access_key,
            context_path=context_path,
            model_path=model_path,
            sensitivity=sensitivity,
            endpoint_duration_sec=endpoint_duration_sec,
            require_endpoint=require_endpoint,
        )

        # PyAudio stream config
        self._pa = pyaudio.PyAudio()
        self._pa_stream = self._pa.open(
            rate=self._rhino.sample_rate,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=self._rhino.frame_length,
            input_device_index=audio_device_index,
        )

    @property
    def context_info(self) -> str:
        """Human-readable description of the loaded context."""
        return getattr(self._rhino, "context_info", "")

    @property
    def version(self) -> str:
        """Rhino library version string."""
        return getattr(self._rhino, "version", "unknown")

    @property
    def is_running(self) -> bool:  # noqa: D401 – property style
        """True while the background capture thread is active."""
        return self._running

    def start(self) -> None:
        """Begin microphone capture and intent processing in background."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("RhinoIntentRecognizer started")

    def stop(self) -> None:
        """Stop background capture and free resources."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._cleanup()
        logger.info("RhinoIntentRecognizer stopped")

    def reset(self) -> None:
        """Reset internal Rhino state (discard current utterance)."""
        try:
            self._rhino.reset()
        except Exception as exc:  # pragma: no cover
            logger.warning("Rhino reset failed: %s", exc)

    def _run(self) -> None:  # pragma: no cover – IO loop
        try:
            while self._running:
                pcm = self._pa_stream.read(
                    self._rhino.frame_length, exception_on_overflow=False
                )
                pcm_int16 = struct.unpack_from(
                    f"{self._rhino.frame_length}h", pcm
                )
                try:
                    finalized = self._rhino.process(pcm_int16)
                except Exception as exc:
                    logger.error("Rhino process error: %s", exc)
                    continue

                if finalized:
                    try:
                        inf = self._rhino.get_inference()
                        data: InferenceDict = {
                            "is_understood": inf.is_understood,
                            "intent": inf.intent if inf.is_understood else None,
                            "slots": inf.slots if inf.is_understood else {},
                        }
                        self._inference_cb(data)
                    except Exception as exc:  # pragma: no cover – edge
                        logger.error("Failed to fetch inference: %s", exc)
                    finally:
                        # Prepare for next utterance
                        self._rhino.reset()
        except Exception as exc:  # pragma: no cover
            logger.error("Rhino thread crashed: %s", exc)
        finally:
            self._cleanup()

    def _cleanup(self):
        try:
            if self._pa_stream is not None:
                self._pa_stream.stop_stream()
                self._pa_stream.close()
            if self._pa is not None:
                self._pa.terminate()
            if self._rhino is not None:
                try:
                    self._rhino.delete()  # pyright: ignore [reportUnknownMemberType]
                except AttributeError:
                    pass 