"""Product counting with the Gemini API.

  from productcounter import GeminiCounter
  GeminiCounter().count("shelf.jpg")          # -> number of products in the image

Live line-crossing counting lives in productcounter.linecount (used by live.py).
"""
from .gemini_counter import GeminiCounter
from .masking import render_masked_image, sort_boxes
from .batch_storage import BatchStorage, batch_storage

__all__ = ["GeminiCounter", "render_masked_image", "sort_boxes", "BatchStorage", "batch_storage"]

