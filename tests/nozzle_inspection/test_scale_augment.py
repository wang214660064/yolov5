import unittest

from project.nozzle_inspection.data.scale_augment import scale_yolo_box_on_canvas


class ScaleAugmentTest(unittest.TestCase):
    def test_scale_yolo_box_keeps_coordinates_valid(self):
        result = scale_yolo_box_on_canvas(
            box=(0, 0.5, 0.5, 0.2, 0.2),
            scale=1.5,
            offset_x=0.0,
            offset_y=0.0,
        )

        cls, x, y, w, h = result
        self.assertEqual(cls, 0)
        self.assertTrue(0.0 <= x <= 1.0)
        self.assertTrue(0.0 <= y <= 1.0)
        self.assertTrue(0.0 < w <= 1.0)
        self.assertTrue(0.0 < h <= 1.0)

    def test_scale_yolo_box_returns_none_when_clipped_away(self):
        result = scale_yolo_box_on_canvas(
            box=(0, 0.95, 0.95, 0.05, 0.05),
            scale=1.0,
            offset_x=0.2,
            offset_y=0.2,
        )

        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
