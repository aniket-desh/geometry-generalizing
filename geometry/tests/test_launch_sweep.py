from __future__ import annotations

import unittest
from pathlib import Path

from launch_sweep import command_for, matrix, select_shard


class BreadthReplicationTest(unittest.TestCase):
    def test_exact_unique_coverage_across_four_shards(self) -> None:
        runs = matrix("breadth-replicate")
        expected = {
            (task, seed)
            for task in ("torus5", "cycle31", "dihedral12", "random31")
            for seed in (1, 2)
        }

        self.assertEqual({(run.task, run.seed) for run in runs}, expected)
        self.assertEqual(len(runs), len(expected))

        shards = [select_shard(runs, index, 4) for index in range(4)]
        flattened = [run for shard in shards for run in shard]
        self.assertTrue(all(len(shard) == 2 for shard in shards))
        self.assertEqual(set(flattened), set(runs))
        self.assertEqual(len(flattened), len(set(flattened)))

    def test_protocol_and_explicit_seed_identity(self) -> None:
        for run in matrix("breadth-replicate"):
            self.assertEqual(run.preset, "micro")
            self.assertEqual(run.steps, 60_000)
            self.assertEqual(run.batch_size, 4_096)
            self.assertEqual(run.train_fraction, 0.4)
            self.assertEqual(run.eval_every, 500)
            self.assertEqual(run.snapshot_every, 1_000)
            self.assertEqual(run.dense_checkpoint_every, 1_000)
            command = command_for(
                run,
                output_root=Path("/tmp/breadth-replication-test"),
                compile_model=True,
                device="cuda",
            )
            for option in (
                "--seed",
                "--split-seed",
                "--task-seed",
                "--token-seed",
            ):
                self.assertEqual(command.count(option), 1)
                self.assertEqual(
                    command[command.index(option) + 1],
                    str(run.seed),
                )


if __name__ == "__main__":
    unittest.main()
