from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from render_model_spectrum import self_test


class ModelSpectrumRendererTest(unittest.TestCase):
    def test_strict_five_model_spectrum(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="model-spectrum-test-"
        ) as temporary:
            self_test(Path(temporary))


if __name__ == "__main__":
    unittest.main()
