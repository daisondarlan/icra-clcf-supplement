"""
Unit tests for the temporal gate, run against synthetic label sequences
(not real flight data -- this validates the gate's logic, not the
classifier's accuracy on real terrain).

Run from the repository root: python tests/test_temporal_gate.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from temporal_gate import TemporalGate


def run(seq, n_streak=2):
    gate = TemporalGate(n_streak=n_streak)
    return [gate.update(lbl) for lbl in seq]


def check(name, seq, expected, n_streak=2):
    got = run(seq, n_streak)
    status = "PASS" if got == expected else "FAIL"
    print(f"[{status}] {name}\n    input:    {seq}\n    expected: {expected}\n    got:      {got}")
    return status == "PASS"


def main():
    results = []

    # A single SAFE frame is not enough to commit (N=2)
    results.append(check(
        "single SAFE frame does not commit",
        ['SAFE'],
        ['NOT SAFE'],
    ))

    # Two consecutive SAFE frames commit to SAFE
    results.append(check(
        "two consecutive SAFE frames commit",
        ['SAFE', 'SAFE'],
        ['NOT SAFE', 'SAFE'],
    ))

    # A single spurious NOT SAFE frame resets progress, delaying commitment
    results.append(check(
        "spurious NOT SAFE resets the streak",
        ['SAFE', 'NOT SAFE', 'SAFE', 'SAFE'],
        ['NOT SAFE', 'NOT SAFE', 'NOT SAFE', 'SAFE'],
    ))

    # An invalid/dropout frame (None) is treated the same as NOT SAFE -- fail closed
    results.append(check(
        "invalid frame (None) resets the streak like NOT SAFE",
        ['SAFE', 'SAFE', None, 'SAFE'],
        ['NOT SAFE', 'SAFE', 'NOT SAFE', 'NOT SAFE'],
    ))

    # Once safe, staying safe stays safe
    results.append(check(
        "sustained SAFE run stays committed",
        ['SAFE', 'SAFE', 'SAFE', 'SAFE'],
        ['NOT SAFE', 'SAFE', 'SAFE', 'SAFE'],
    ))

    n_pass = sum(results)
    print(f"\n{n_pass}/{len(results)} checks passed")
    if n_pass != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
