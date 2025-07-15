#!/usr/bin/env python3


from __future__ import annotations

import os
import subprocess
import tempfile
import logging
from typing import Optional, Dict, Any
from pathlib import Path

try:
    from gtts import gTTS  # type: ignore
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False

logger = logging.getLogger(__name__)


class TTSModule:
    """Google TTS wrapper with the same public API as the old Piper module."""
    
    def __init__(self, language: str = "en", tld: str = "com", slow: bool = False, enabled: bool = False):
        """Create TTS module.

        Args:
            language: ISO language code (e.g. "en", "en-uk", "de").
            tld: Google Translate top-level-domain which changes the accent/voice
                 for some languages, e.g. "co.uk" for UK English, "com.au" for
                 Australian, "ca" for Canadian French, etc.
            slow:  If True, speak more slowly.
            enabled: initial enabled state.
        """
        self.language = language
        self.tld = tld
        self.slow = slow
        self.enabled = enabled
        self.available = GTTS_AVAILABLE
        if self.available:
            logger.info("gTTS is available (lang=%s, tld=%s)", language, tld)
        else:
            logger.warning("gTTS package not installed. Run `pip install gTTS` to enable TTS.")

    def enable(self) -> Dict[str, Any]:
        if not self.available:
            return {"success": False, "error": "gTTS not installed (pip install gTTS)"}
        self.enabled = True
        return {"success": True, "message": "TTS enabled"}
    
    def disable(self) -> Dict[str, Any]:
        self.enabled = False
        return {"success": True, "message": "TTS disabled"}
    
    def is_enabled(self) -> bool:
        return self.enabled and self.available

    def speak(self, text: str) -> Dict[str, Any]:
        if not self.is_enabled():
            return {"success": False, "error": "TTS is disabled or not available"}
        if not text.strip():
            return {"success": False, "error": "No text to speak"}
        try:
            # Generate MP3 via gTTS (accent via tld)
            tts = gTTS(text=text, lang=self.language, tld=self.tld, slow=self.slow)
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                mp3_path = tmp.name
            tts.save(mp3_path)

            # Play the MP3
            self._play_audio(mp3_path)
            
            # Remove file afterwards
            try:
                os.unlink(mp3_path)
            except OSError:
                pass
            
            preview = text[:50] + ("..." if len(text) > 50 else "")
            return {"success": True, "message": f"Spoke: {preview}"}
        except Exception as exc:
            logger.error("TTS error: %s", exc)
            return {"success": False, "error": str(exc)}

    def _play_audio(self, file_path: str):
        """Attempt to play an audio file using common CLI players."""
        players = ["mpv", "ffplay", "ffmpeg", "cvlc", "aplay", "paplay"]
        for player in players:
            try:
                # Use -nodisp -autoexit for ffplay, --no-video for mpv, etc.
                if player == "ffplay":
                    cmd = [player, "-nodisp", "-autoexit", file_path]
                elif player == "mpv":
                    cmd = [player, "--no-video", file_path]
                else:
                    cmd = [player, file_path]
                subprocess.run(cmd, capture_output=True, timeout=15)
                return
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue
        logger.warning("No suitable audio player found to play TTS output")
            
    def get_status(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "available": self.available,
            "language": self.language,
            "tld": self.tld,
            "status": "enabled" if self.is_enabled() else "disabled",
        } 