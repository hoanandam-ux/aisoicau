"""
patterns.py – Pattern Recognition Logic
Detects Long Streaks (AAAA), Alternating (ABAB), and Symmetry (AABB / AAABBB)
in a binary H/L sequence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


PatternType = Literal["streak", "alternating", "symmetry", "mixed", "none"]


@dataclass
class PatternResult:
    pattern_type: PatternType
    description: str
    streak_info: dict = field(default_factory=dict)
    alternating_score: float = 0.0
    symmetry_info: dict = field(default_factory=dict)
    dominant_label: str | None = None
    raw_flags: dict = field(default_factory=dict)


class PatternDetector:
    """
    Stateless detector. Call `detect(labels)` with a list of 'H'/'L' strings.
    Works on arbitrary window sizes; caller typically passes the last 20.
    """

    # Minimum run length to qualify as a "streak"
    STREAK_MIN: int = 3
    # Fraction of consecutive alternations to qualify
    ALTERNATING_THRESHOLD: float = 0.75

    def detect(self, labels: list[str]) -> PatternResult:
        if not labels:
            return PatternResult(pattern_type="none", description="No data.")

        flags: dict[str, bool] = {
            "streak": self._has_streak(labels),
            "alternating": self._has_alternating(labels),
            "symmetry": self._has_symmetry(labels),
        }
        alt_score = self._alternating_score(labels)
        streak_info = self._streak_detail(labels)
        sym_info = self._symmetry_detail(labels)
        dominant = self._dominant(labels)

        active = [k for k, v in flags.items() if v]

        if len(active) >= 2:
            ptype: PatternType = "mixed"
            desc = f"Mixed pattern detected: {', '.join(active)}."
        elif flags["streak"]:
            ptype = "streak"
            desc = (
                f"Long streak of '{streak_info['current_symbol']}' "
                f"({streak_info['current_run']} consecutive)."
            )
        elif flags["alternating"]:
            ptype = "alternating"
            desc = f"Alternating ABAB pattern (score {alt_score:.0%})."
        elif flags["symmetry"]:
            ptype = "symmetry"
            desc = f"Symmetry block detected (AABB / AAABBB): {sym_info}."
        else:
            ptype = "none"
            desc = "No dominant pattern detected."

        return PatternResult(
            pattern_type=ptype,
            description=desc,
            streak_info=streak_info,
            alternating_score=alt_score,
            symmetry_info=sym_info,
            dominant_label=dominant,
            raw_flags=flags,
        )

    # ── detectors ────────────────────────────────────────────────────────────

    def _has_streak(self, labels: list[str]) -> bool:
        return self._streak_detail(labels)["current_run"] >= self.STREAK_MIN

    def _streak_detail(self, labels: list[str]) -> dict:
        if not labels:
            return {"current_run": 0, "current_symbol": None, "max_run": 0}
        cur_sym = labels[-1]
        cur_run = 1
        for s in reversed(labels[:-1]):
            if s == cur_sym:
                cur_run += 1
            else:
                break

        max_run = 1
        run = 1
        for i in range(1, len(labels)):
            if labels[i] == labels[i - 1]:
                run += 1
                max_run = max(max_run, run)
            else:
                run = 1
        return {"current_run": cur_run, "current_symbol": cur_sym, "max_run": max_run}

    def _alternating_score(self, labels: list[str]) -> float:
        if len(labels) < 2:
            return 0.0
        alternations = sum(
            1 for i in range(1, len(labels)) if labels[i] != labels[i - 1]
        )
        return alternations / (len(labels) - 1)

    def _has_alternating(self, labels: list[str]) -> bool:
        return self._alternating_score(labels) >= self.ALTERNATING_THRESHOLD

    def _symmetry_detail(self, labels: list[str]) -> dict:
        """
        Look for AABB or AAABBB blocks in the last portion of the sequence.
        Returns info about the best match found.
        """
        best: dict = {"found": False, "block_size": 0, "position": -1}
        n = len(labels)
        # Try block sizes 2, 3, 4
        for block in (2, 3, 4):
            pattern_len = block * 2
            if n < pattern_len:
                continue
            window = labels[n - pattern_len:]
            first_half = window[:block]
            second_half = window[block:]
            if (
                len(set(first_half)) == 1
                and len(set(second_half)) == 1
                and first_half[0] != second_half[0]
            ):
                best = {"found": True, "block_size": block, "position": n - pattern_len}
                break
        return best

    def _has_symmetry(self, labels: list[str]) -> bool:
        return self._symmetry_detail(labels).get("found", False)

    def _dominant(self, labels: list[str]) -> str | None:
        if not labels:
            return None
        h = labels.count("H")
        l = labels.count("L")
        if h == l:
            return None
        return "H" if h > l else "L"
