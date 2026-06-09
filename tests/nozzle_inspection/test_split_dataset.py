import unittest

from project.nozzle_inspection.data.split_dataset import stratified_split


class SplitDatasetTest(unittest.TestCase):
    def test_stratified_split_preserves_both_classes(self):
        samples = [(f"ng_{i}", 0) for i in range(10)] + [(f"ok_{i}", 1) for i in range(10)]

        train, val = stratified_split(samples, val_ratio=0.2, seed=7)

        self.assertEqual(len(train), 16)
        self.assertEqual(len(val), 4)
        self.assertEqual({label for _, label in val}, {0, 1})

    def test_split_is_deterministic(self):
        samples = [(f"ng_{i}", 0) for i in range(8)] + [(f"ok_{i}", 1) for i in range(8)]

        first = stratified_split(samples, val_ratio=0.25, seed=3)
        second = stratified_split(samples, val_ratio=0.25, seed=3)

        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
