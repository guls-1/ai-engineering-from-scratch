"""Toy Self-Refine and CRITIC loop with a noisy-verifier experiment.

Lesson: phases/14-agent-engineering/05-self-refine-and-critic/docs/en.md
Sources: Madaan et al. 2023 (Self-Refine); Gou et al. 2023 (CRITIC).
Task: a 3-bullet summary, each bullet under 60 chars, with no known factual errors.
Run: python3 main.py   Tests: python3 -m unittest discover tests -v
"""

from __future__ import annotations
import random
from dataclasses import dataclass, field


KNOWN_WRONG_FACTS = [
    "paris is the capital of germany",
    "mt everest is in europe",
    "the sun orbits the earth",
]

rng = random.Random(42)


@dataclass
class Attempt:
    iteration: int
    output: str
    critique: str
    verified: bool


def generate(topic: str, history: list[Attempt]) -> str:
    if not history:
        return (
            "- Paris is the capital of Germany\n"
            "- Mt Everest is in Europe\n"
            "- Water boils at 100C"
        )
    last = history[-1]
    if "germany" in last.critique.lower():
        return (
            "- Paris is the capital of France\n"
            "- Mt Everest is in Europe\n"
            "- Water boils at 100C"
        )
    if "everest" in last.critique.lower():
        return (
            "- Paris is the capital of France\n"
            "- Mt Everest is in Asia\n"
            "- Water boils at 100C at sea level"
        )
    return history[-1].output


def feedback_self(output: str) -> tuple[str, bool]:
    if "Germany" in output and "Paris" in output:
        return "first bullet reads wrong, double-check capital", False
    if "Europe" in output and "Everest" in output:
        return "second bullet's continent looks off to me", False
    return "no issues", True


def verify_external(output: str) -> tuple[str, bool]:
    text = output.lower()
    for fact in KNOWN_WRONG_FACTS:
        key = fact.split(" is ")[0] if " is " in fact else fact
        if "paris" in text and "germany" in text:
            return f"verifier: 'paris is the capital of germany' contradicts reference data", False
        if "everest" in text and "europe" in text:
            return f"verifier: 'mt everest is in europe' contradicts reference data", False
    if len([l for l in output.splitlines() if l.startswith("-")]) != 3:
        return "verifier: expected 3 bullet lines", False
    if any(len(l) > 60 for l in output.splitlines()):
        return "verifier: bullet exceeds 60 chars", False
    return "verifier: ok", True


def is_actually_correct(output: str) -> bool:
    text = output.lower()
    if "paris" in text and "germany" in text:
        return False
    if "everest" in text and "europe" in text:
        return False
    return True


def make_noisy_verifier(base, fp_rate, rng):
    def verify(output):
        critique, ok = base(output)
        if not ok and rng.random() < fp_rate:
            return "verifier: ok", True   # false positive
        return critique, ok
    return verify


noisy_verify = make_noisy_verifier(verify_external, fp_rate=0.3, rng=rng)


def refine(topic: str, prev: str, critique: str, history: list[Attempt]) -> str:
    return generate(topic, history)


def run_loop(
    topic: str, use_critic: bool | None, max_iters: int = 4, verifier=None
) -> list[Attempt]:
    history: list[Attempt] = []
    output = generate(topic, history)
    verify = verifier or (verify_external if use_critic else (lambda o: feedback_self(o)) if use_critic is False else noisy_verify)
    for i in range(1, max_iters + 1):
        critique, ok = verify(output)
        history.append(Attempt(i, output, critique, ok))
        if ok:
            break
        output = refine(topic, output, critique, history)
    return history


def run_trials(
    fp_rate: float, n: int = 1000, max_iters: int = 4
) -> list[tuple[bool, bool]]:
    results: list[tuple[bool, bool]] = []
    for seed in range(n):
        rng = random.Random(seed)
        noisy = make_noisy_verifier(verify_external, fp_rate, rng)
        history = run_loop("world facts", True, max_iters, verifier=noisy)
        final = history[-1]
        results.append((final.verified, is_actually_correct(final.output)))
    return results


def print_trials_table(rates: tuple[float, ...] = (0.0, 0.3), n: int = 1000) -> None:
    print(f"\nNoisy verifier trials (n={n}, max_iters=4)")
    print("-" * 60)
    print(f"  {'fp rate':>8} {'verified':>10} {'correct':>10} {'verified+wrong':>16}")
    for fp in rates:
        results = run_trials(fp, n)
        verified = sum(v for v, _ in results) / n
        correct = sum(c for _, c in results) / n
        wrong = sum(v and not c for v, c in results) / n
        print(f"  {fp:>8.0%} {verified:>10.1%} {correct:>10.1%} {wrong:>16.1%}")


# Assumed miss rates (false-positive rate of the critic), not measurements.
# A same-model critic shares the generator's blind spots, so it misses more of
# the generator's own errors. A smaller critic from another model is weaker but
# its mistakes are independent of the generator's.
CRITIC_PAIRINGS = {
    "same model critiques itself": 0.5,
    "big generator, small critic": 0.3,
}


def compare_critics(
    pairings: dict[str, float] = CRITIC_PAIRINGS, n: int = 1000
) -> dict[str, tuple[float, float]]:
    """Return {label: (correct_rate, verified_but_wrong_rate)} per pairing."""
    out: dict[str, tuple[float, float]] = {}
    for label, fp in pairings.items():
        results = run_trials(fp, n)
        correct = sum(c for _, c in results) / n
        wrong = sum(v and not c for v, c in results) / n
        out[label] = (correct, wrong)
    return out


def print_critic_comparison(n: int = 1000) -> None:
    print(f"\nGenerator-critic pairings (n={n}, max_iters=4)")
    print("-" * 60)
    print(f"  {'pairing':<30} {'miss':>5} {'correct':>9} {'verified+wrong':>15}")
    for label, (correct, wrong) in compare_critics(n=n).items():
        print(f"  {label:<30} {CRITIC_PAIRINGS[label]:>5.0%} {correct:>9.1%} {wrong:>15.1%}")


def print_run(label: str, history: list[Attempt]) -> None:
    print(f"\n{label}")
    print("-" * 60)
    for a in history:
        tag = "OK " if a.verified else "..."
        print(f"  iter {a.iteration} {tag} critique: {a.critique}")
        for line in a.output.splitlines():
            print(f"    {line}")


def main() -> None:
    print("=" * 70)
    print("SELF-REFINE and CRITIC — Phase 14, Lesson 05")
    print("=" * 70)

    hist_self = run_loop("world facts", use_critic=False)
    print_run("Self-Refine (self-critique only)", hist_self)

    hist_critic = run_loop("world facts", use_critic=True)
    print_run("CRITIC (external verifier)", hist_critic)

    hist_noisy = run_loop("world facts", use_critic=None)
    print_run("Noisy Verifier", hist_noisy)

    def summary(hist: list[Attempt]) -> str:
        return "passed" if hist and hist[-1].verified else "did not converge"

    print()
    print(f"Self-Refine ended: {summary(hist_self)}  after {len(hist_self)} iters")
    print(f"CRITIC    ended: {summary(hist_critic)}  after {len(hist_critic)} iters")
    print(f"Noisy Verifier ended: {summary(hist_noisy)}  after {len(hist_noisy)} iters")
    print_trials_table()
    print_critic_comparison()

    print()
    print("Observation: CRITIC's verifier is grounded against reference data; a")
    print("self-critic can fail to flag its own confident-sounding hallucination.")


if __name__ == "__main__":
    main()
