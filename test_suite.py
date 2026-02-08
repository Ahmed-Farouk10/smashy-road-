# test_suite.py
import unittest
import numpy as np
import os
import tempfile
import shutil
from env import SmashyRoadEnv
from q_learning import q_learning
from value_iteration import value_iteration
from policy_gradient import policy_gradient


def setup_professional_environment():
    """Create required directory structure"""
    dirs = ["results", "results/plots", "results/models", "results/logs", "results/reports"]
    try:
        for d in dirs:
            os.makedirs(d, exist_ok=True)
        print(" Professional environment setup complete!")
        return True
    except Exception as e:
        print(f" Setup failed: {e}")
        return False


class TestPoliceCatcherEnv(unittest.TestCase):
    def setUp(self):
        self.env = SmashyRoadEnv(grid_size=10, max_steps=100)

    def test_environment_initialization(self):
        self.assertEqual(self.env.grid_size, 10)
        self.assertEqual(self.env.max_steps, 100)
        self.assertEqual(self.env.action_space_size, 4)
        self.assertIsInstance(self.env.agent_pos, list)
        self.assertIsInstance(self.env.police_pos, list)
        self.assertEqual(len(self.env.agent_pos), 2)
        self.assertEqual(len(self.env._get_state()), 4)

    def test_reset_functionality(self):
        initial_state = self.env.reset()
        self.assertIsInstance(initial_state, tuple)
        self.assertEqual(len(initial_state), 4)
        self.assertEqual(self.env.steps, 0)
        self.assertEqual(self.env.score, 0)
        self.assertFalse(self.env.game_over)
        self.assertFalse(self.env.won)
        self.assertFalse(self.env.fuel_collected)

    def test_valid_actions(self):
        self.env.reset()
        for action in range(4):
            state, reward, done = self.env.step(action)
            self.assertIsInstance(state, tuple)
            self.assertIsInstance(reward, (int, float))
            self.assertIsInstance(done, bool)

    def test_position_bounds(self):
        self.env.reset()
        for _ in range(50):
            action = self.env.action_space_sample()
            self.env.step(action)
            for pos in [self.env.agent_pos, self.env.police_pos]:
                self.assertGreaterEqual(pos[0], 0)
                self.assertLess(pos[0], self.env.grid_size)
                self.assertGreaterEqual(pos[1], 0)
                self.assertLess(pos[1], self.env.grid_size)

    def test_goal_reaching(self):
      self.env.reset()
      # Move police far away
      self.env.police_pos = [0, 0]
      goal_pos = [self.env.grid_size - 1, self.env.grid_size - 1]
      self.env.agent_pos = [goal_pos[0], goal_pos[1] - 1]  # (9,8)
      state, reward, done = self.env.step(3)  # Move right
      self.assertTrue(self.env.won)
      self.assertTrue(done)
      self.assertEqual(reward, 100)

    def test_police_collision(self):
        self.env.reset()
        self.env.agent_pos = self.env.police_pos.copy()
        state, reward, done = self.env.step(0)
        self.assertTrue(self.env.game_over)
        self.assertTrue(done)
        self.assertLess(reward, -50)  # Should be -100

    def test_fuel_collection(self):
        self.env.reset()
        fuel_pos = self.env.fuel_pos
        self.env.agent_pos = [fuel_pos[0], fuel_pos[1] - 1]
        state, reward, done = self.env.step(3)  # Move right
        if self.env.agent_pos == fuel_pos and not self.env.fuel_collected:
            self.assertGreaterEqual(reward, 5)
            self.assertTrue(self.env.fuel_collected)

    def test_max_steps_termination(self):
        self.env.reset()
        for step in range(self.env.max_steps + 5):
            state, reward, done = self.env.step(0)
            if step >= self.env.max_steps - 1:
                self.assertTrue(done)
                break

    def test_save_load_game_state(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as f:
            temp_file = f.name
        try:
            self.env.reset()
            for _ in range(5):
                self.env.step(self.env.action_space_sample())
            original_pos = self.env.agent_pos.copy()
            original_score = self.env.score
            success = self.env.save_game_state(temp_file)
            self.assertTrue(success)

            # Modify state and reload
            self.env.agent_pos = [0, 0]
            self.env.score = 0
            success = self.env.load_game_state(temp_file)
            self.assertTrue(success)
            self.assertEqual(self.env.agent_pos, original_pos)
            self.assertEqual(self.env.score, original_score)
        finally:
            os.unlink(temp_file)


class TestQLearningAgent(unittest.TestCase):
    def setUp(self):
        self.env = SmashyRoadEnv(grid_size=5)

    def test_q_learning_runs(self):
        try:
            Q, rewards, wins = q_learning(self.env, episodes=5)
            self.assertEqual(len(rewards), 5)
            self.assertEqual(len(wins), 5)
            self.assertEqual(Q.shape, (5, 5, 5, 5, 4))
        except Exception as e:
            self.fail(f"Q-Learning failed: {e}")


class TestValueIterationAgent(unittest.TestCase):
    def setUp(self):
        self.env = SmashyRoadEnv(grid_size=5)

    def test_value_iteration_runs(self):
        try:
            policy = value_iteration(self.env)
            self.assertIsInstance(policy, dict)
            self.assertGreater(len(policy), 0)
            sample_state = (2, 2, 3, 3)
            if sample_state in policy:
                self.assertIn(policy[sample_state], [0, 1, 2, 3])
        except Exception as e:
            self.fail(f"Value Iteration failed: {e}")


class TestPolicyGradientAgent(unittest.TestCase):
    def setUp(self):
        self.env = SmashyRoadEnv(grid_size=5)

    def test_policy_gradient_runs(self):
        try:
            weights, bias, rewards, wins = policy_gradient(self.env, episodes=5)
            self.assertEqual(len(rewards), 5)
            self.assertEqual(len(wins), 5)
            self.assertEqual(weights.shape, (4, 4))
            self.assertEqual(bias.shape, (4,))
        except Exception as e:
            self.fail(f"Policy Gradient failed: {e}")


class TestUtils(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.test_dir)

    def tearDown(self):
        os.chdir(self.original_cwd)
        shutil.rmtree(self.test_dir)

    def test_environment_setup(self):
        success = setup_professional_environment()
        self.assertTrue(success)
        self.assertTrue(os.path.exists("results"))
        self.assertTrue(os.path.exists("results/plots"))
        self.assertTrue(os.path.exists("results/models"))
        self.assertTrue(os.path.exists("results/logs"))
        self.assertTrue(os.path.exists("results/reports"))


def run_all_tests():
    print("Running Professional Police Catcher Game Test Suite")
    print("=" * 60)
    suite = unittest.TestSuite()
    for cls in [
        TestPoliceCatcherEnv,
        TestQLearningAgent,
        TestValueIterationAgent,
        TestPolicyGradientAgent,
        TestUtils
    ]:
        suite.addTests(unittest.TestLoader().loadTestsFromTestCase(cls))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    print(f"\nOverall result: {'PASSED' if result.wasSuccessful() else 'FAILED'}")
    return result.wasSuccessful()


if __name__ == "__main__":
    run_all_tests()