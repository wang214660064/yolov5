from pathlib import Path
import unittest

from project.nozzle_inspection.utils.path_utils import ProjectPaths


class ProjectPathsTest(unittest.TestCase):
    def test_project_paths_resolve_outer_dataset_and_outputs(self):
        with self.subTest("解析外层数据和输出路径"):
            import tempfile

            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                repo_root = temp_path / "yolov5"
                repo_root.mkdir()
                paths = ProjectPaths(repo_root=repo_root)

                self.assertEqual(paths.repo_root, repo_root)
                self.assertEqual(paths.outer_root, temp_path)
                self.assertEqual(paths.dataset_root, temp_path / "dataset_2")
                self.assertEqual(paths.outputs_root, temp_path / "outputs")
                self.assertEqual(paths.project_data_dir, repo_root / "project" / "nozzle_inspection" / "data")

    def test_ensure_dir_creates_directory(self):
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            paths = ProjectPaths(repo_root=temp_path / "yolov5")
            target = temp_path / "new_dir"

            created = paths.ensure_dir(target)

            self.assertEqual(created, target)
            self.assertTrue(target.exists())


if __name__ == "__main__":
    unittest.main()
