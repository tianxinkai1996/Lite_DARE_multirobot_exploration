"""Architecture tests for the three sibling coordination packages.

三个并列协调包的目录结构与旧文件清理测试。
"""
from __future__ import annotations

import inspect
import unittest
from pathlib import Path

from collision.joint_resolver import JointCollisionResolver
from deadlock.state_tracker import DeadlockStateTracker
from mergingmap.motion_coordinator import MergingMapMotionCoordinator


HERE = Path(__file__).resolve().parent
MERGINGMAP_DIR = HERE.parent
PROJECT_ROOT = MERGINGMAP_DIR.parent


class ParallelModuleLayoutTests(unittest.TestCase):
    """Verify sibling layout, direct imports, and obsolete-file removal.

    中文目的：防止后续修改重新建立旧兼容层或把碰撞、死锁代码嵌回 MergingMap。
    English implementation: checks directory parents, removed paths, public
    implementation locations, and an upper bound on source-file length.
    """

    def test_three_packages_are_parallel(self):
        """All three coordination packages must share the project root.

        中文目的：确认 mergingmap、collision 和 deadlock 为并列目录。
        中文实现：比较三个目录的父路径，并确认每个目录均为 Python 包。
        English purpose: verify the requested sibling-package structure.
        English implementation: compares parent paths and checks package markers.
        """

        package_dirs = [
            PROJECT_ROOT / "mergingmap",
            PROJECT_ROOT / "collision",
            PROJECT_ROOT / "deadlock",
        ]
        self.assertTrue(all(path.parent == PROJECT_ROOT for path in package_dirs))
        self.assertTrue(all((path / "__init__.py").is_file() for path in package_dirs))

    def test_obsolete_compatibility_files_are_removed(self):
        """Old forwarding modules must not exist.

        中文目的：明确旧运动管理、预留管理和动态区域转发文件已经无用。
        中文实现：逐一检查旧路径不存在，使残留旧导入在测试阶段立即暴露。
        English purpose: ensure obsolete compatibility facades stay removed.
        English implementation: asserts every retired path is absent.
        """

        obsolete_paths = [
            PROJECT_ROOT / "classes/multi_robot/motion_manager.py",
            PROJECT_ROOT / "classes/multi_robot/reservation_manager.py",
            PROJECT_ROOT / "mergingmap/dynamic_region_manager.py",
            PROJECT_ROOT / "test_deadlock_motion_manager.py",
        ]
        self.assertTrue(all(not path.exists() for path in obsolete_paths))

    def test_public_implementations_live_in_expected_sibling_packages(self):
        """Public classes must resolve to their real modular source files.

        中文目的：保证统一协调器、碰撞消解器和死锁跟踪器不经过隐藏转发层。
        中文实现：读取类定义源码路径，并与三个并列目录逐一匹配。
        English purpose: prove that imports point directly at real implementations.
        English implementation: compares inspected source paths with package roots.
        """

        expected = {
            MergingMapMotionCoordinator: PROJECT_ROOT / "mergingmap",
            JointCollisionResolver: PROJECT_ROOT / "collision",
            DeadlockStateTracker: PROJECT_ROOT / "deadlock",
        }
        for implementation, package_dir in expected.items():
            source = Path(inspect.getsourcefile(implementation) or "").resolve()
            self.assertEqual(source.parent, package_dir.resolve())

    def test_implementation_files_remain_below_line_limit(self):
        """Refactored source files must not grow back into monoliths.

        中文目的：控制三个目录中单个实现文件的规模，降低后续阅读和维护成本。
        中文实现：递归统计非测试 Python 文件行数，并要求每个文件不超过 500 行。
        English purpose: prevent the modular code from becoming monolithic again.
        English implementation: counts lines in non-test Python files and enforces
        a 500-line ceiling.
        """

        oversized: list[tuple[str, int]] = []
        for package_name in ("mergingmap", "collision", "deadlock"):
            package_dir = PROJECT_ROOT / package_name
            for source in package_dir.rglob("*.py"):
                if "tests" in source.parts or "test_outputs" in source.parts:
                    continue
                line_count = len(source.read_text(encoding="utf-8").splitlines())
                if line_count > 500:
                    oversized.append((str(source.relative_to(PROJECT_ROOT)), line_count))
        self.assertEqual(oversized, [])


if __name__ == "__main__":
    unittest.main()

