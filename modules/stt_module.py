#!/usr/bin/env python3


import os
import wave
import json
import logging
import threading
import queue
import tempfile
import base64
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from abc import ABC, abstractmethod

# Optional imports
try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False
    logging.warning("sounddevice not available. Install with: pip install sounddevice")

try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except ImportError:
    SR_AVAILABLE = False
    logging.warning("speech_recognition not available. Install with: pip install SpeechRecognition")

try:
    import groq
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    logging.warning("groq package not available. Install with: pip install groq")

logger = logging.getLogger(__name__)

@dataclass
class AudioConfig:
    """Audio configuration settings"""
    sample_rate: int = 16000
    channels: int = 1
    dtype: str = 'int16'
    blocksize: int = 8000
    device: Optional[int] = None
    input_gain: float = 1.0
    silence_threshold: float = 0.03
    silence_duration: float = 0.5

@dataclass
class TranscriptionResult:
    """Container for transcription results"""
    text: str
    confidence: float
    language: Optional[str] = None
    speaker: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    words: Optional[List[Dict[str, Any]]] = None

class AudioProcessor:
    """Handles audio recording and preprocessing"""
    
    def __init__(self, config: AudioConfig):
        self.config = config
        self.stream = None
        self.audio_queue = queue.Queue()
        self.is_recording = False
        
        # Check for sounddevice availability
        if not SOUNDDEVICE_AVAILABLE:
            logger.warning("sounddevice not available. Install with: pip install sounddevice")
    
    def start_recording(self):
        """Start audio recording"""
        if not SOUNDDEVICE_AVAILABLE:
            raise ImportError("sounddevice package required for audio recording")
            
        def audio_callback(indata, frames, time, status):
            """Callback for audio stream"""
            if status:
                logger.warning(f"Audio callback status: {status}")
            # Process audio data (apply gain, noise reduction, etc.)
            processed = indata * self.config.input_gain
            self.audio_queue.put(processed.copy())

        self.stream = sd.InputStream(
            samplerate=self.config.sample_rate,
            channels=self.config.channels,
            dtype=self.config.dtype,
            blocksize=self.config.blocksize,
            device=self.config.device,
            callback=audio_callback
        )
        self.stream.start()
        self.is_recording = True
        logger.info("Started audio recording")
    
    def stop_recording(self):
        """Stop audio recording"""
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.is_recording = False
        logger.info("Stopped audio recording")
    
    def get_audio_chunk(self, timeout: Optional[float] = None) -> Optional[np.ndarray]:
        """Get next audio chunk from queue"""
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None

class BaseSTTEngine(ABC):
    """Base class for STT engines"""
    
    @abstractmethod
    def transcribe_file(self, audio_file: str) -> TranscriptionResult:
        """Transcribe audio file"""
        pass
    
    @abstractmethod
    def transcribe_stream(self, audio_chunk: np.ndarray) -> Optional[TranscriptionResult]:
        """Transcribe audio stream chunk"""
        pass

class GoogleSRSTTEngine(BaseSTTEngine):
    """Google Speech Recognition engine using speech_recognition library"""
    def __init__(self):
        if not SR_AVAILABLE:
            raise ImportError("speech_recognition package required for GoogleSRSTTEngine")
        self.recognizer = sr.Recognizer()
        self.sample_rate = 16000
        self.channels = 1

    def transcribe_file(self, audio_file: str) -> TranscriptionResult:
        with sr.AudioFile(audio_file) as source:
            audio = self.recognizer.record(source)
            try:
                text = self.recognizer.recognize_google(audio)
                return TranscriptionResult(text=text, confidence=1.0)
            except Exception as e:
                logger.error(f"GoogleSRSTTEngine file error: {e}")
                return TranscriptionResult(text="", confidence=0.0)

    def transcribe_stream(self, audio_chunk: np.ndarray) -> Optional[TranscriptionResult]:
        import io
        import numpy as np
        import wave
        # Print debug info about audio_chunk
        logger.debug(f"GoogleSRSTTEngine: audio_chunk shape={getattr(audio_chunk, 'shape', None)}, dtype={getattr(audio_chunk, 'dtype', None)}, min={np.min(audio_chunk) if audio_chunk is not None else 'n/a'}, max={np.max(audio_chunk) if audio_chunk is not None else 'n/a'}")
        try:
            with io.BytesIO() as wav_io:
                with wave.open(wav_io, 'wb') as wf:
                    wf.setnchannels(self.channels)
                    wf.setsampwidth(2)  # int16
                    wf.setframerate(self.sample_rate)
                    wf.writeframes((audio_chunk * 32767).astype(np.int16).tobytes())
                wav_io.seek(0)
                audio = sr.AudioFile(wav_io)
                with audio as source:
                    audio_data = self.recognizer.record(source)
                    try:
                        text = self.recognizer.recognize_google(audio_data)
                        return TranscriptionResult(text=text, confidence=1.0)
                    except Exception as e:
                        logger.error(f"GoogleSRSTTEngine stream error: {type(e).__name__}: {e}")
                        return None
        except Exception as e:
            logger.error(f"GoogleSRSTTEngine stream conversion error: {type(e).__name__}: {e}")
            return None

class SphinxSTTEngine(BaseSTTEngine):
    """Offline STT using CMU Sphinx via speech_recognition"""
    def __init__(self):
        if not SR_AVAILABLE:
            raise ImportError("speech_recognition package required for SphinxSTTEngine")
        self.recognizer = sr.Recognizer()
        self.sample_rate = 16000
        self.channels = 1

    def transcribe_file(self, audio_file: str) -> TranscriptionResult:
        with sr.AudioFile(audio_file) as source:
            audio = self.recognizer.record(source)
            try:
                text = self.recognizer.recognize_sphinx(audio)
                return TranscriptionResult(text=text, confidence=1.0)
            except Exception as e:
                logger.error(f"SphinxSTTEngine file error: {e}")
                return TranscriptionResult(text="", confidence=0.0)

    def transcribe_stream(self, audio_chunk: np.ndarray) -> Optional[TranscriptionResult]:
        import io
        import numpy as np
        import wave
        logger.debug(f"SphinxSTTEngine: audio_chunk shape={getattr(audio_chunk, 'shape', None)}, dtype={getattr(audio_chunk, 'dtype', None)}, min={np.min(audio_chunk) if audio_chunk is not None else 'n/a'}, max={np.max(audio_chunk) if audio_chunk is not None else 'n/a'}")
        try:
            with io.BytesIO() as wav_io:
                with wave.open(wav_io, 'wb') as wf:
                    wf.setnchannels(self.channels)
                    wf.setsampwidth(2)  # int16
                    wf.setframerate(self.sample_rate)
                    wf.writeframes((audio_chunk * 32767).astype(np.int16).tobytes())
                wav_io.seek(0)
                audio = sr.AudioFile(wav_io)
                with audio as source:
                    audio_data = self.recognizer.record(source)
                    try:
                        text = self.recognizer.recognize_sphinx(audio_data)
                        return TranscriptionResult(text=text, confidence=1.0)
                    except Exception as e:
                        logger.error(f"SphinxSTTEngine stream error: {type(e).__name__}: {e}")
                        return None
        except Exception as e:
            logger.error(f"SphinxSTTEngine stream conversion error: {type(e).__name__}: {e}")
            return None

class GroqSTTEngine(BaseSTTEngine):
    """Groq-based STT engine using Whisper model"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "whisper-large-v3"):
        if not GROQ_AVAILABLE:
            raise ImportError("groq package required for GroqSTTEngine")
        
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("Groq API key required. Set GROQ_API_KEY environment variable or pass api_key.")
        
        self.client = Groq(api_key=self.api_key)
        self.model = model
    
    def _encode_audio(self, audio_file: str) -> str:
        """Encode audio file to base64"""
        with open(audio_file, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    
    def transcribe_file(self, audio_file: str) -> TranscriptionResult:
        """Transcribe audio file using Groq"""
        try:
            # Open file in binary mode for Groq API
            with open(audio_file, "rb") as audio:
                # Call Groq API
                response = self.client.audio.transcriptions.create(
                    model=self.model,
                    file=audio,
                    response_format="verbose_json"
                )
            
            # Parse response (assuming standard Whisper API response format)
            return TranscriptionResult(
                text=response.text,
                confidence=1.0,  # Groq API doesn't provide confidence scores
                language=None,  # Language detection not supported yet
                words=None  # Word-level timing not supported yet
            )
        except Exception as e:
            logger.error(f"Groq transcription error: {e}")
            raise
    
    def transcribe_stream(self, audio_chunk: np.ndarray) -> Optional[TranscriptionResult]:
        """Transcribe audio stream chunk using Groq"""
        # Save chunk to temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as temp_file:
            with wave.open(temp_file.name, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(audio_chunk.tobytes())
            
            return self.transcribe_file(temp_file.name)

class STTManager:
    """
    Speech-to-Text manager supporting multiple engines and modes.
    
    Features:
    - Multiple STT engine support
    - Real-time transcription
    - File transcription
    - Audio device management
    - Event callbacks
    """
    
    def __init__(self, 
                 engine: str = "groq",
                 model: str = "whisper-large-v3",
                 api_key: Optional[str] = None,
                 audio_config: Optional[AudioConfig] = None):
        """
        Initialize STT manager.
        
        Args:
            engine: STT engine to use ('groq', 'whisper', or 'vosk')
            model: Model to use (for Groq/Whisper)
            api_key: Groq API key
            audio_config: Audio configuration settings
        """
        self.engine_name = engine
        self.model = model
        self.api_key = api_key
        self.audio_config = audio_config or AudioConfig()
        self.callbacks: List[Callable[[TranscriptionResult], None]] = []
        self.running = False
        self.transcription_thread = None
        self.audio_processor = AudioProcessor(self.audio_config)
        # Choose engine
        if engine == "groq":
            self.engine = GroqSTTEngine(api_key=api_key, model=model)
        elif engine == "whisper":
            from faster_whisper import WhisperModel
            self.engine = WhisperSTTEngine(model=model)
        elif engine == "vosk":
            from vosk import Model, KaldiRecognizer
            self.engine = VoskSTTEngine(model=model)
        elif engine == "sr":
            self.engine = GoogleSRSTTEngine()
        elif engine == "sphinx":
            self.engine = SphinxSTTEngine()
        else:
            raise ValueError(f"Unknown STT engine: {engine}")
    
    def add_callback(self, callback: Callable[[TranscriptionResult], None]):
        """Add transcription result callback"""
        self.callbacks.append(callback)
    
    def remove_callback(self, callback: Callable[[TranscriptionResult], None]):
        """Remove transcription result callback"""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
    
    def _notify_callbacks(self, result: TranscriptionResult):
        """Notify all callbacks with transcription result"""
        for callback in self.callbacks:
            try:
                callback(result)
            except Exception as e:
                logger.error(f"Error in transcription callback: {e}")
    
    def start_realtime(self):
        """Start real-time transcription"""
        if self.running:
            return
        
        self.running = True
        self.audio_processor.start_recording()
        
        def transcription_loop():
            while self.running:
                chunk = self.audio_processor.get_audio_chunk(timeout=0.1)
                if chunk is not None:
                    result = self.engine.transcribe_stream(chunk)
                    if result:
                        self._notify_callbacks(result)
        
        self.transcription_thread = threading.Thread(target=transcription_loop)
        self.transcription_thread.start()
        logger.info("Started real-time transcription")

    def stop_realtime(self):
        """Stop real-time transcription"""
        self.running = False
        if self.transcription_thread:
            self.transcription_thread.join()
            self.transcription_thread = None
        self.audio_processor.stop_recording()
        logger.info("Stopped real-time transcription")

    def transcribe_file(self, audio_file: str) -> TranscriptionResult:
        """
        Transcribe audio file.
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            TranscriptionResult object
        """
        result = self.engine.transcribe_file(audio_file)
        self._notify_callbacks(result)
        return result

def list_input_devices():
    """List available audio input devices."""
    if not SOUNDDEVICE_AVAILABLE:
        return []
    devices = sd.query_devices()
    input_devices = []
    for i, device in enumerate(devices):
        if device.get('max_input_channels', 0) > 0:
            input_devices.append({
                'index': i,
                'name': device.get('name'),
                'input_channels': device.get('max_input_channels'),
                'default_samplerate': device.get('default_samplerate')
            })
    return input_devices

def record_test(device: int, seconds: int = 3) -> Dict[str, Any]:
    """Record a short audio clip and return analysis."""
    if not SOUNDDEVICE_AVAILABLE:
        return {"success": False, "error": "sounddevice not available"}
    try:
        sample_rate = 16000
        recording = sd.rec(int(seconds * sample_rate), samplerate=sample_rate, channels=1, device=device)
        sd.wait()
        peak = np.max(np.abs(recording))
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_f:
            with wave.open(temp_f.name, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes((recording * 32767).astype(np.int16).tobytes())
            
            return {"success": True, "peak": f"{peak:.3f}", "file": temp_f.name}

    except Exception as e:
        return {"success": False, "error": str(e)}

# Example usage
if __name__ == "__main__":
    # Initialize STT manager with Groq engine
    stt = STTManager(engine="groq", model="whisper-large-v3")
    
    # Add callback for transcription results
    def print_transcription(result: TranscriptionResult):
        print(f"Transcribed: {result.text}")
    
    stt.add_callback(print_transcription)
    
    # Start real-time transcription
    stt.start_realtime()
    
    try:
        # Run for 30 seconds
        import time
        time.sleep(30)
    finally:
        # Stop transcription
        stt.stop_realtime() 