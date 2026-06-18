import unittest

from bitweaver.allocators import dp_allocate, greedy_allocate, uniform_allocate


OPTIONS = [
    {
        2: {"size": 2, "loss": 0.12},
        4: {"size": 4, "loss": 0.03},
        8: {"size": 8, "loss": 0.00},
    },
    {
        2: {"size": 2, "loss": 0.25},
        4: {"size": 4, "loss": 0.10},
        8: {"size": 8, "loss": 0.00},
    },
    {
        2: {"size": 2, "loss": 0.08},
        4: {"size": 4, "loss": 0.02},
        8: {"size": 8, "loss": 0.00},
    },
    {
        2: {"size": 2, "loss": 0.15},
        4: {"size": 4, "loss": 0.06},
        8: {"size": 8, "loss": 0.00},
    },
    {
        2: {"size": 2, "loss": 0.30},
        4: {"size": 4, "loss": 0.18},
        8: {"size": 8, "loss": 0.00},
    },
    {
        2: {"size": 2, "loss": 0.06},
        4: {"size": 4, "loss": 0.01},
        8: {"size": 8, "loss": 0.00},
    },
]


class AllocatorTests(unittest.TestCase):
    def test_dp_allocate_finds_global_minimum_under_budget(self):
        result = dp_allocate(OPTIONS, budget=30)

        self.assertEqual(result.config, [4, 8, 4, 4, 8, 2])
        self.assertEqual(result.total_size, 30)
        self.assertAlmostEqual(result.total_loss, 0.17)

    def test_uniform_allocate_rejects_bit_width_that_exceeds_budget(self):
        result = uniform_allocate(OPTIONS, bit_width=8, budget=30)

        self.assertFalse(result.feasible)
        self.assertEqual(result.config, [8, 8, 8, 8, 8, 8])
        self.assertEqual(result.total_size, 48)

    def test_greedy_allocate_stays_within_budget(self):
        result = greedy_allocate(OPTIONS, budget=30)

        self.assertTrue(result.feasible)
        self.assertLessEqual(result.total_size, 30)
        self.assertEqual(len(result.config), len(OPTIONS))


if __name__ == "__main__":
    unittest.main()
