"""Persistent storage and retrieval for counted product batches.

Saves batch images (original and annotated), item counts, foreign object counts,
timestamps, and metadata so users can review audit history at any time.
"""
from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = BASE_DIR / "data" / "batches"


class BatchStorage:
    def __init__(self, storage_dir: Path | str | None = None):
        self.storage_dir = Path(storage_dir) if storage_dir else DEFAULT_DATA_DIR
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.storage_dir / "batches.json"
        if not self.index_file.exists():
            self._write_index([])

    def _read_index(self) -> list[dict[str, Any]]:
        if not self.index_file.exists():
            return []
        try:
            with open(self.index_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    def _write_index(self, records: list[dict[str, Any]]) -> None:
        temp_file = self.index_file.with_suffix(".tmp")
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        temp_file.replace(self.index_file)

    def generate_batch_id(self) -> str:
        now = datetime.now()
        base_id = f"BATCH-{now.strftime('%Y%m%d-%H%M%S')}"
        existing_ids = {r.get("id") for r in self._read_index()}
        if base_id not in existing_ids:
            return base_id
        counter = 1
        while f"{base_id}-{counter:02d}" in existing_ids:
            counter += 1
        return f"{base_id}-{counter:02d}"

    def save_batch(
        self,
        original_image: Image.Image,
        masked_image: Image.Image,
        count: int,
        product: str = "General Products",
        foreign_count: int = 0,
        notes: str = "",
    ) -> dict[str, Any]:
        """Saves a batch record with images to disk and updates the index."""
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        batch_id = self.generate_batch_id()
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        orig_filename = f"{batch_id}_orig.jpg"
        masked_filename = f"{batch_id}_masked.jpg"

        orig_path = self.storage_dir / orig_filename
        masked_path = self.storage_dir / masked_filename

        # Save images as RGB JPEGs
        if original_image.mode != "RGB":
            original_image = original_image.convert("RGB")
        original_image.save(orig_path, format="JPEG", quality=92)

        if masked_image.mode != "RGB":
            masked_image = masked_image.convert("RGB")
        masked_image.save(masked_path, format="JPEG", quality=92)

        record = {
            "id": batch_id,
            "timestamp": now_str,
            "product": product or "General Products",
            "count": int(count),
            "foreign_count": int(foreign_count),
            "original_image": f"data/batches/{orig_filename}",
            "masked_image": f"data/batches/{masked_filename}",
            "notes": notes,
        }

        records = self._read_index()
        records.insert(0, record)  # Newest first
        self._write_index(records)
        return record

    def list_batches(self) -> list[dict[str, Any]]:
        """Returns all batches ordered from newest to oldest."""
        return self._read_index()

    def get_batch(self, batch_id: str) -> dict[str, Any] | None:
        """Finds a single batch record by ID."""
        for rec in self._read_index():
            if rec.get("id") == batch_id:
                return rec
        return None

    def delete_batch(self, batch_id: str) -> bool:
        """Deletes a batch and cleans up its image files."""
        records = self._read_index()
        found = False
        remaining = []
        for rec in records:
            if rec.get("id") == batch_id:
                found = True
                for key in ("original_image", "masked_image"):
                    p = Path(rec.get(key, ""))
                    if p.is_file():
                        try:
                            p.unlink()
                        except Exception:
                            pass
            else:
                remaining.append(rec)

        if found:
            self._write_index(remaining)
        return found

    def clear_all(self) -> None:
        """Removes all stored batches and images."""
        records = self._read_index()
        for rec in records:
            for key in ("original_image", "masked_image"):
                p = Path(rec.get(key, ""))
                if p.is_file():
                    try:
                        p.unlink()
                    except Exception:
                        pass
        self._write_index([])


# Shared default singleton
batch_storage = BatchStorage()
