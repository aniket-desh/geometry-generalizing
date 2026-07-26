from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from render_breadth_replication import run_self_test


class BreadthReplicationRendererTest(unittest.TestCase):
    def test_desktop_and_mobile_renderers(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="breadth-replication-test-"
        ) as temporary:
            run_self_test(Path(temporary))


if __name__ == "__main__":
    unittest.main()
