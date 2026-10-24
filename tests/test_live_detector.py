"""Unit tests for productcounter.live_detector."""
import unittest
import numpy as np
import cv2
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from productcounter.live_detector import detect_live_products


class TestLiveDetector(unittest.TestCase):
    def setUp(self):
        # Create an artificial test image with 20 white circles on dark background
        self.img = np.zeros((480, 640, 3), dtype=np.uint8)
        self.num_circles = 20
        # Draw 4 rows of 5 circles
        for r in range(4):
            for c in range(5):
                cx = 80 + c * 110
                cy = 70 + r * 100
                cv2.circle(self.img, (cx, cy), 28, (220, 220, 220), -1)

    def test_target_matched(self):
        ann, count, status, info = detect_live_products(self.img, target_count=20, product_type="circles")
        self.assertEqual(count, 20)
        self.assertEqual(status, "MATCH")
        self.assertEqual(info["diff"], 0)
        self.assertIn("TARGET MATCHED", info["status_text"])
        self.assertLess(info["latency_ms"], 100.0)

    def test_target_add_guidance(self):
        # Target 25, have 20 -> suggest ADD 5
        ann, count, status, info = detect_live_products(self.img, target_count=25, product_type="circles")
        self.assertEqual(count, 20)
        self.assertEqual(status, "ADD")
        self.assertEqual(info["diff"], -5)
        self.assertIn("ADD 5 MORE", info["status_text"])

    def test_target_remove_guidance(self):
        # Target 15, have 20 -> suggest REMOVE 5
        ann, count, status, info = detect_live_products(self.img, target_count=15, product_type="circles")
        self.assertEqual(count, 20)
        self.assertEqual(status, "REMOVE")
        self.assertEqual(info["diff"], 5)
        self.assertIn("REMOVE 5", info["status_text"])

    def test_empty_frame(self):
        empty = np.zeros((100, 100, 3), dtype=np.uint8)
        ann, count, status, info = detect_live_products(empty, target_count=20)
        self.assertEqual(count, 0)
        self.assertEqual(status, "EMPTY")


if __name__ == "__main__":
    unittest.main()
