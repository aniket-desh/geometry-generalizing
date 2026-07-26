from __future__ import annotations

import unittest

import numpy as np

from geogen.tasks import make_task


class CorruptionProvenanceTest(unittest.TestCase):
    def test_broken_cycle_records_nominal_and_actual_fraction(self) -> None:
        clean = make_task("cycle12")
        broken = make_task("broken12")
        changed = int(np.count_nonzero(broken.table != clean.table))

        self.assertEqual(changed, 24)
        self.assertAlmostEqual(broken.corruption_fraction, 0.15)
        self.assertAlmostEqual(
            broken.actual_corruption_fraction,
            changed / broken.table.size,
        )
        self.assertAlmostEqual(broken.actual_corruption_fraction, 1 / 6)

    def test_small_requested_fraction_preserves_rounding_distinction(self) -> None:
        task = make_task("cycle7", corruption=0.05)

        self.assertAlmostEqual(task.corruption_fraction, 0.05)
        self.assertAlmostEqual(task.actual_corruption_fraction, 2 / 7)


if __name__ == "__main__":
    unittest.main()
