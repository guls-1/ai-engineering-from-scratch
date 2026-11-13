"""Tests for main.py (Phase 14, Lesson 05)."""

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import main as lesson
from main import (
    feedback_self,
    generate,
    is_actually_correct,
    make_noisy_verifier,
    run_loop,
    run_trials,
    verify_external,
)

GOOD = (
    "- Paris is the capital of France\n"
    "- Mt Everest is in Asia\n"
    "- Water boils at 100C at sea level"
)


class TestVerifier(unittest.TestCase):
    def test_accepts_correct_output(self):
        critique, ok = verify_external(GOOD)
        self.assertTrue(ok)
        self.assertEqual(critique, "verifier: ok")

    def test_rejects_each_known_wrong_fact(self):
        germany = "- Paris is the capital of Germany\n- B\n- C"
        europe = "- A\n- Mt Everest is in Europe\n- C"
        for bad, keyword in ((germany, "germany"), (europe, "everest")):
            critique, ok = verify_external(bad)
            self.assertFalse(ok)
            self.assertIn(keyword, critique.lower())

    def test_rejects_wrong_bullet_count(self):
        critique, ok = verify_external("- Paris is the capital of France\n- Water boils")
        self.assertFalse(ok)
        self.assertIn("3 bullet", critique)

    def test_bullet_length_limit_is_60_chars(self):
        at_limit = "- " + "x" * 58
        over_limit = "- " + "x" * 59
        self.assertTrue(verify_external(f"{at_limit}\n- b\n- c")[1])
        critique, ok = verify_external(f"{over_limit}\n- b\n- c")
        self.assertFalse(ok)
        self.assertIn("60", critique)

    def test_self_critic_flags_without_naming_the_fix(self):
        wrong = generate("t", [])
        critique, ok = feedback_self(wrong)
        self.assertFalse(ok)
        self.assertNotIn("germany", critique.lower())
        self.assertEqual(feedback_self(GOOD), ("no issues", True))


class TestLoops(unittest.TestCase):
    def test_critic_converges_to_correct_answer(self):
        history = run_loop("t", use_critic=True)
        self.assertTrue(history[-1].verified)
        self.assertTrue(is_actually_correct(history[-1].output))
        self.assertEqual(len(history), 3)

    def test_self_refine_stalls_on_its_own_hallucination(self):
        history = run_loop("t", use_critic=False)
        self.assertFalse(history[-1].verified)
        self.assertEqual(len(history), 4)
        self.assertEqual(len({a.output for a in history}), 1)
        self.assertFalse(is_actually_correct(history[-1].output))

    def test_max_iters_one_gives_critic_no_chance_to_fix(self):
        history = run_loop("t", use_critic=True, max_iters=1)
        self.assertEqual(len(history), 1)
        self.assertFalse(history[0].verified)

    def test_generate_applies_the_critique_it_is_given(self):
        first = generate("t", [])
        germany = lesson.Attempt(1, first, "contradicts germany", False)
        second = generate("t", [germany])
        self.assertIn("France", second)
        self.assertIn("Europe", second)
        everest = lesson.Attempt(2, second, "everest is wrong", False)
        self.assertEqual(generate("t", [germany, everest]), GOOD)


class TestNoisyVerifier(unittest.TestCase):
    def test_zero_false_positive_rate_is_transparent(self):
        noisy = make_noisy_verifier(verify_external, 0.0, random.Random(0))
        wrong = generate("t", [])
        self.assertEqual(noisy(wrong), verify_external(wrong))

    def test_full_false_positive_rate_accepts_wrong_output(self):
        noisy = make_noisy_verifier(verify_external, 1.0, random.Random(0))
        wrong = generate("t", [])
        self.assertEqual(noisy(wrong), ("verifier: ok", True))

    def test_never_rejects_a_correct_output(self):
        noisy = make_noisy_verifier(verify_external, 0.5, random.Random(1))
        for _ in range(50):
            self.assertTrue(noisy(GOOD)[1])


class TestTrials(unittest.TestCase):
    def test_clean_verifier_is_verified_and_correct(self):
        results = run_trials(0.0, n=50)
        self.assertTrue(all(verified and correct for verified, correct in results))

    def test_false_positives_produce_verified_but_wrong(self):
        results = run_trials(0.3, n=400)
        wrong = [r for r in results if r[0] and not r[1]]
        correct_rate = sum(c for _, c in results) / len(results)
        self.assertGreater(len(wrong), 0)
        self.assertLess(correct_rate, 0.7)
        self.assertGreater(correct_rate, 0.3)

    def test_trials_are_deterministic(self):
        self.assertEqual(run_trials(0.3, n=100), run_trials(0.3, n=100))

    def test_one_iteration_never_verifies_a_clean_run(self):
        results = run_trials(0.0, n=20, max_iters=1)
        self.assertFalse(any(verified for verified, _ in results))


class TestCriticPairings(unittest.TestCase):
    def test_returns_one_entry_per_pairing(self):
        out = lesson.compare_critics(n=50)
        self.assertEqual(set(out), set(lesson.CRITIC_PAIRINGS))

    def test_rates_are_probabilities(self):
        for correct, wrong in lesson.compare_critics(n=50).values():
            self.assertTrue(0.0 <= correct <= 1.0)
            self.assertTrue(0.0 <= wrong <= 1.0)

    def test_lower_miss_rate_critic_is_more_often_correct(self):
        out = lesson.compare_critics({"weak": 0.6, "strong": 0.1}, n=500)
        self.assertGreater(out["strong"][0], out["weak"][0])
        self.assertLess(out["strong"][1], out["weak"][1])

    def test_perfect_critic_is_always_correct(self):
        out = lesson.compare_critics({"perfect": 0.0}, n=50)
        self.assertEqual(out["perfect"], (1.0, 0.0))

    def test_comparison_is_deterministic(self):
        self.assertEqual(lesson.compare_critics(n=100), lesson.compare_critics(n=100))


if __name__ == "__main__":
    unittest.main()
