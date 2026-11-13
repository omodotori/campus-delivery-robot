"""Correctness and integration checks, using only the standard library."""
import unittest
from collections import deque
from main import astar, bayes, mdp_decision, START, GOAL, WALLS, TRANSITIONS


def bfs_distance(start, goal, walls, size):
    queue, seen = deque([(start, 0)]), {start}
    while queue:
        p, d = queue.popleft()
        if p == goal:
            return d
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = p[0] + dr, p[1] + dc
            if 0 <= n[0] < size and 0 <= n[1] < size and n not in walls and n not in seen:
                seen.add(n)
                queue.append((n, d + 1))
    return None


class ProjectTests(unittest.TestCase):
    def test_astar_optimal_and_valid(self):
        for walls in (WALLS, {(0, 1), (1, 1), (2, 1), (3, 1)}):
            path, _ = astar(START, GOAL, walls)
            self.assertEqual(len(path)-1, bfs_distance(START, GOAL, walls, 5))
            self.assertEqual((path[0], path[-1]), (START, GOAL))
            for a, b in zip(path, path[1:]):
                self.assertEqual(abs(a[0]-b[0])+abs(a[1]-b[1]), 1)
                self.assertNotIn(b, walls)

    def test_unreachable(self):
        self.assertEqual(astar(START, GOAL, {(0, 1), (1, 0)})[0], [])

    def test_same_start_goal(self):
        self.assertEqual(astar(START, START, WALLS)[0], [START])

    def test_bayes_and_action_integration(self):
        for alert, expected_p, expected_action in ((True, 17/24, 'Wait'), (False, 9/128, 'Move')):
            p = bayes(.3, alert)[0]
            self.assertAlmostEqual(p, expected_p)
            action, scores, _ = mdp_decision(p)
            self.assertEqual(action, expected_action)
            self.assertAlmostEqual(scores['Move'], 10-30*p)
            self.assertAlmostEqual(scores['Wait'], -2+p)

    def test_probabilities_and_bellman_transition_term(self):
        for action in TRANSITIONS.values():
            for distribution in action.values():
                self.assertAlmostEqual(sum(distribution.values()), 1)
                self.assertTrue(all(0 <= p <= 1 for p in distribution.values()))
        _, _, q = mdp_decision(.3, future_values={'clear': 4., 'blocked': -3.})
        self.assertAlmostEqual(q['Move']['clear'], 10+.9*(.9*4+.1*(-3)))
        self.assertAlmostEqual(q['Wait']['blocked'], -1+.9*(.6*4+.4*(-3)))

    def test_invalid_probability(self):
        with self.assertRaises(ValueError):
            bayes(1.1, True)
        with self.assertRaises(ValueError):
            bayes(0, True, false_positive=0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
