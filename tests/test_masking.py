"""Unit tests for productcounter.masking."""
import unittest
import numpy as np
from PIL import Image

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from productcounter.masking import sort_boxes, render_masked_image, get_colors_for_items



class TestMasking(unittest.TestCase):
    def setUp(self):
        self.dummy_image = Image.new("RGB", (200, 200), color=(128, 128, 128))
        self.dummy_boxes = np.array([
            [100.0, 50.0, 150.0, 90.0],
            [10.0, 20.0, 40.0, 60.0],
            [10.0, 100.0, 40.0, 140.0],
        ], dtype=np.float32)

    def test_sort_boxes(self):
        sorted_boxes = sort_boxes(self.dummy_boxes)
        # Top-most item should be first
        self.assertLess(sorted_boxes[0, 0], sorted_boxes[-1, 0])
        # Between the two top items (y=10), the left one (x=20) should come before (x=100)
        self.assertLess(sorted_boxes[0, 1], sorted_boxes[1, 1])

    def test_render_all_styles(self):
        styles = ["ellipse", "fill", "spotlight", "contour", "outline_only"]
        for s in styles:
            ann, sbs, mask, items, kpi = render_masked_image(
                self.dummy_image,
                self.dummy_boxes,
                mask_style=s,
                opacity=0.4,
                show_badges=True,
                show_borders=True,
                show_centroids=True,
            )
            self.assertIsInstance(ann, Image.Image)
            self.assertIsInstance(sbs, Image.Image)
            self.assertIsInstance(mask, Image.Image)
            self.assertEqual(len(items), 3)
            self.assertEqual(kpi["count"], 3)

    def test_empty_boxes(self):
        empty_boxes = np.zeros((0, 4), dtype=np.float32)
        ann, sbs, mask, items, kpi = render_masked_image(self.dummy_image, empty_boxes)
        self.assertEqual(len(items), 0)
        self.assertEqual(kpi["count"], 0)

    def test_foreign_objects_rendering(self):
        foreign_objs = [
            {"box": [50.0, 50.0, 120.0, 80.0], "description": "Water Bottle"}
        ]
        ann, sbs, mask, items, kpi = render_masked_image(
            self.dummy_image,
            self.dummy_boxes,
            foreign_objects=foreign_objs,
        )
        self.assertEqual(kpi["foreign_count"], 1)
        self.assertIn("Water Bottle", kpi["foreign_objects"])


if __name__ == "__main__":
    unittest.main()
