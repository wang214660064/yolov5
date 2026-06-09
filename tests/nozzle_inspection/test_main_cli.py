import unittest

from project.nozzle_inspection.main import build_parser


class MainCliTest(unittest.TestCase):
    def test_parser_contains_chinese_description(self):
        parser = build_parser()

        self.assertIn("3D打印机喷头检测", parser.description)

    def test_parser_accepts_dry_run_train(self):
        parser = build_parser()

        args = parser.parse_args(["train", "--dry-run", "--epochs", "3"])

        self.assertEqual(args.command, "train")
        self.assertTrue(args.dry_run)
        self.assertEqual(args.epochs, 3)

    def test_parser_accepts_analyze_data(self):
        parser = build_parser()

        args = parser.parse_args(["analyze-data", "--dataset", "../dataset_2"])

        self.assertEqual(args.command, "analyze-data")
        self.assertEqual(args.dataset, "../dataset_2")


if __name__ == "__main__":
    unittest.main()
