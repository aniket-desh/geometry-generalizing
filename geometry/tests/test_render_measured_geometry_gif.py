from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from render_measured_geometry_gif import self_test


class MeasuredGeometryGifTest(unittest.TestCase):
    def test_measured_checkpoint_renderer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self_test(Path(directory))


if __name__ == "__main__":
    unittest.main()
