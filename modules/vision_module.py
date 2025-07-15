import os
import io
import base64
import logging
from typing import List, Dict, Any, Optional

try:
    from PIL import Image  # type: ignore
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

try:
    from groq import Groq
except ImportError:  # pragma: no cover
    raise ImportError("groq package is required. Install with `pip install groq`")

logger = logging.getLogger(__name__)

class VisionModule:
    """
    Wrapper for Groq multimodal (vision) chat completions.
    """

    def __init__(self, client: Groq, model: str):
        self.client = client
        self.model = model

    # ---------- helpers ----------

    @staticmethod
    def _encode_image(path: str) -> str:
        """Return base-64 string for a local image file.
        If the file is larger than ~3 MiB, down-scale + convert to JPEG so that
        the final request payload stays under Groq's 4 MiB limit.
        """
        try:
            file_size = os.path.getsize(path)
        except OSError:
            file_size = 0

        # Groq request hard-limit is 4 MiB; base64 adds ~33 % overhead.
        size_limit = 3 * 1024 * 1024  # 3 MiB raw ≈ 4 MiB b64

        if _PIL_AVAILABLE and file_size > size_limit:
            try:
                with Image.open(path) as im:
                    # Optional down-scaling for very large screenshots
                    max_dim = 1600
                    if max(im.size) > max_dim:
                        im.thumbnail((max_dim, max_dim))

                    # Re-encode as JPEG with quality 85 in memory
                    buffer = io.BytesIO()
                    im.convert("RGB").save(buffer, format="JPEG", quality=85)
                    buffer.seek(0)
                    encoded = base64.b64encode(buffer.read()).decode("utf-8")
                    return encoded
            except Exception as e:
                logger.warning(f"PIL optimisation failed ({e}); falling back to raw encode")

        # Fallback: raw file bytes
        with open(path, "rb") as img:
            data = img.read()
            if not data:
                raise ValueError(f"Image file is empty: {path}")
            return base64.b64encode(data).decode("utf-8")

    @staticmethod
    def _mime_from_ext(ext: str) -> str:
        return {
            ".png": "image/png",
            ".webp": "image/webp",
            ".gif": "image/gif",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
        }.get(ext.lower(), "image/jpeg")

    @staticmethod
    def _build_content(prompt: str, image_spec: Dict[str, Any]) -> List[Dict[str, Any]]:
        return [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": image_spec},
        ]

    # ---------- public ----------

    def analyze_image(
        self,
        image_source: str,
        prompt: str = "What's in this image?",
        *,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Send an image plus a prompt to Groq and return the assistant response.

        image_source can be a local file or an HTTP(S) URL.
        """
        # Local file → data URI; URL passes through untouched
        if image_source.lower().startswith(("http://", "https://")):
            image_spec = {"url": image_source}
        else:
            if not os.path.isfile(image_source):
                raise FileNotFoundError(f"Image not found: {image_source}")
            b64 = self._encode_image(image_source)
            mime = self._mime_from_ext(os.path.splitext(image_source)[1])
            image_spec = {"url": f"data:{mime};base64,{b64}"}

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append(
            {
                "role": "user",
                "content": self._build_content(prompt, image_spec),
            }
        )

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=1,
            stream=False,
        )

        return (completion.choices[0].message.content or "").strip()

    def process_image(self, image_path: str, prompt: str = "Analyze this screenshot and describe what you see.") -> str:
        """
        Process an image file and return the analysis.
        
        Args:
            image_path: Path to the image file
            prompt: Optional prompt to guide the analysis
            
        Returns:
            str: Analysis of the image
        """
        try:
            return self.analyze_image(
                image_source=image_path,
                prompt=prompt,
                temperature=0.7,
                max_tokens=1024
            )
        except Exception as e:
            logger.error(f"Failed to process image {image_path}: {e}")
            raise
