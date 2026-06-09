import unittest

from project.nozzle_inspection.data.label_converter import LabelConverter


class LabelConverterTest(unittest.TestCase):
    def test_convert_known_prefix_to_ng_and_ok(self):
        converter = LabelConverter()

        self.assertEqual(converter.class_id_for_stem("nozzle_tip_wrapped_0001"), 0)
        self.assertEqual(converter.class_id_for_stem("nozzle_no_extrusion_0001"), 0)
        self.assertEqual(converter.class_id_for_stem("nozzle_heavy_contamination_0001"), 0)
        self.assertEqual(converter.class_id_for_stem("nozzle_slight_contamination_0001"), 0)
        self.assertEqual(converter.class_id_for_stem("camera_occlusion_0001"), 0)
        self.assertEqual(converter.class_id_for_stem("nozzle_clean_0001"), 1)
        self.assertEqual(converter.class_id_for_stem("nozzle_extrusion_normal_0001"), 1)

    def test_convert_label_lines_rewrites_class_and_keeps_box(self):
        converter = LabelConverter()
        lines = ["3 0.5 0.5 0.2 0.1\n"]

        converted = converter.convert_lines("nozzle_tip_wrapped_0001", lines)

        self.assertEqual(converted, ["0 0.500000 0.500000 0.200000 0.100000\n"])

    def test_unknown_prefix_raises_clear_error(self):
        converter = LabelConverter()

        with self.assertRaisesRegex(ValueError, "未知样本前缀"):
            converter.class_id_for_stem("unknown_0001")

    def test_invalid_box_raises_clear_error(self):
        converter = LabelConverter()

        with self.assertRaisesRegex(ValueError, "标签坐标非法"):
            converter.convert_lines("nozzle_clean_0001", ["1 1.2 0.5 0.2 0.1\n"])


if __name__ == "__main__":
    unittest.main()
