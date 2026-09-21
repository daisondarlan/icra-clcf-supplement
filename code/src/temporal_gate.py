"""
Deterministic multi-frame consistency gate for the landing classifier.

Per-frame classification (GNB/SVM/likelihood-ratio) can flip on a single
noisy or borderline frame. This gate only commits to SAFE once N consecutive
frames have independently agreed; any NOT SAFE or invalid frame resets the
streak. It is intentionally conservative (fails toward NOT SAFE), matching
the fail-closed behavior already applied to the per-frame classifiers.
"""


class TemporalGate:
    def __init__(self, n_streak=2):
        if n_streak < 1:
            raise ValueError("n_streak must be >= 1")
        self.n_streak = n_streak
        self.streak_count = 0

    def reset(self):
        self.streak_count = 0

    def update(self, per_frame_label):
        """per_frame_label: 'SAFE' or 'NOT SAFE' (or None for an invalid frame).
        Returns the gated decision: 'SAFE' or 'NOT SAFE'."""
        if per_frame_label == 'SAFE':
            self.streak_count += 1
        else:
            self.streak_count = 0

        return 'SAFE' if self.streak_count >= self.n_streak else 'NOT SAFE'
