"""
fsrs.py - FSRS v5 Spaced Repetition Algorithm
Free Spaced Repetition Scheduler
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import IntEnum
from typing import Optional


# ══════════════════════════════════════════════════════════════════
# ENUMS
# ══════════════════════════════════════════════════════════════════

class Rating(IntEnum):
    Again = 1
    Hard  = 2
    Good  = 3
    Easy  = 4


class State(IntEnum):
    New      = 0
    Learning = 1
    Review   = 2
    Relearn  = 3


# ══════════════════════════════════════════════════════════════════
# CARD
# ═════════════════════════���════════════════════════════════════════

@dataclass
class Card:
    question_id : str       = ""
    state       : State     = State.New
    due         : datetime  = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    stability   : float     = 0.0
    difficulty  : float     = 0.0
    elapsed_days: int       = 0
    scheduled_days: int     = 0
    reps        : int       = 0
    lapses      : int       = 0
    last_review : Optional[datetime] = None

    def to_dict(self) -> dict:
        return {
            "question_id"   : self.question_id,
            "state"         : int(self.state),
            "due"           : self.due.isoformat(),
            "stability"     : self.stability,
            "difficulty"    : self.difficulty,
            "elapsed_days"  : self.elapsed_days,
            "scheduled_days": self.scheduled_days,
            "reps"          : self.reps,
            "lapses"        : self.lapses,
            "last_review"   : self.last_review.isoformat()
                              if self.last_review else None,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Card":
        c = cls()
        c.question_id    = d.get("question_id",    "")
        c.state          = State(d.get("state",    0))
        c.stability      = d.get("stability",      0.0)
        c.difficulty     = d.get("difficulty",     0.0)
        c.elapsed_days   = d.get("elapsed_days",   0)
        c.scheduled_days = d.get("scheduled_days", 0)
        c.reps           = d.get("reps",           0)
        c.lapses         = d.get("lapses",         0)
        due_raw = d.get("due")
        if due_raw:
            try:
                c.due = datetime.fromisoformat(due_raw)
                if c.due.tzinfo is None:
                    c.due = c.due.replace(tzinfo=timezone.utc)
            except Exception:
                c.due = datetime.now(timezone.utc)
        else:
            c.due = datetime.now(timezone.utc)
        lr = d.get("last_review")
        if lr:
            try:
                c.last_review = datetime.fromisoformat(lr)
                if c.last_review.tzinfo is None:
                    c.last_review = c.last_review.replace(tzinfo=timezone.utc)
            except Exception:
                c.last_review = None
        return c

    @property
    def retention(self) -> float:
        if self.stability <= 0 or self.elapsed_days <= 0:
            return 1.0
        return math.exp(
            math.log(0.9) * self.elapsed_days / self.stability
        )


# ══════════════════════════════════════════════════════════════════
# REVIEW LOG
# ══════════════════════════════════════════════════════════════════

@dataclass
class ReviewLog:
    question_id    : str
    rating         : Rating
    state          : State
    due            : datetime
    stability      : float
    difficulty     : float
    elapsed_days   : int
    last_elapsed   : int
    scheduled_days : int
    review         : datetime


# ══════════════════════════════════════════════════════════════════
# FSRS PARAMETERS
# ══════════════════════════════════════════════════════════════════

DEFAULT_W = [
    0.4072, 1.1829, 3.1262, 15.4722,
    7.2102, 0.5316, 1.0651, 0.0589,
    1.5330, 0.1544, 1.0071, 1.9395,
    0.1100, 0.2900, 2.2700, 0.2500,
    2.9898, 0.5100, 0.4300,
]


# ══════════════════════════════════════════════════════════════════
# FSRS SCHEDULER
# ══════════════════════════════════════════════════════════════════

class FSRS:

    def __init__(
        self,
        w             : list  = None,
        request_retention: float = 0.9,
        maximum_interval : int   = 36500,
    ):
        self.w                  = w or DEFAULT_W
        self.request_retention  = request_retention
        self.maximum_interval   = maximum_interval

    # ── Core formulas ─────────────────────────────────────────────

    def _init_stability(self, r: Rating) -> float:
        return max(self.w[r - 1], 0.1)

    def _init_difficulty(self, r: Rating) -> float:
        return min(max(
            self.w[4] - math.exp(self.w[5] * (r - 1)) + 1,
            1.0
        ), 10.0)

    def _next_difficulty(self, d: float, r: Rating) -> float:
        next_d = d - self.w[6] * (r - 3)
        return min(max(
            self._mean_reversion(self.w[4], next_d),
            1.0
        ), 10.0)

    def _mean_reversion(self, init: float, current: float) -> float:
        return self.w[7] * init + (1 - self.w[7]) * current

    def _short_term_stability(
        self, s: float, r: Rating
    ) -> float:
        return s * math.exp(self.w[17] * (r - 3 + self.w[18]))

    def _next_recall_stability(
        self, d: float, s: float, r_val: float, rating: Rating
    ) -> float:
        hard_penalty = self.w[15] if rating == Rating.Hard  else 1.0
        easy_bonus   = self.w[16] if rating == Rating.Easy  else 1.0
        return s * (
            math.exp(self.w[8])
            * (11 - d)
            * math.pow(s, -self.w[9])
            * (math.exp((1 - r_val) * self.w[10]) - 1)
            * hard_penalty
            * easy_bonus
            + 1
        )

    def _next_forget_stability(
        self, d: float, s: float, r_val: float
    ) -> float:
        return (
            self.w[11]
            * math.pow(d, -self.w[12])
            * (math.pow(s + 1, self.w[13]) - 1)
            * math.exp((1 - r_val) * self.w[14])
        )

    def _next_interval(self, s: float) -> int:
        ivl = s / math.log(self.request_retention) * math.log(0.9)
        return min(max(round(ivl), 1), self.maximum_interval)

    # ── Public API ────────────────────────────────────────────────

    def repeat(
        self,
        card       : Card,
        now        : datetime = None,
        rating     : Rating   = Rating.Good,
    ) -> tuple[Card, ReviewLog]:
        if now is None:
            now = datetime.now(timezone.utc)

        # Ensure timezone-aware
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        if card.due.tzinfo is None:
            card.due = card.due.replace(tzinfo=timezone.utc)

        elapsed = max(0, (now - card.due).days) \
                  if card.last_review else 0

        card.elapsed_days = elapsed
        card.last_review  = now

        if card.state == State.New:
            card = self._review_new(card, rating, now)

        elif card.state == State.Learning:
            card = self._review_learning(card, rating, now)

        elif card.state == State.Review:
            card = self._review_review(card, rating, now)

        elif card.state == State.Relearn:
            card = self._review_relearn(card, rating, now)

        log = ReviewLog(
            question_id    = card.question_id,
            rating         = rating,
            state          = card.state,
            due            = card.due,
            stability      = card.stability,
            difficulty     = card.difficulty,
            elapsed_days   = card.elapsed_days,
            last_elapsed   = elapsed,
            scheduled_days = card.scheduled_days,
            review         = now,
        )
        return card, log

    # ── State handlers ────────────────────────────────────────────

    def _review_new(
        self, card: Card, rating: Rating, now: datetime
    ) -> Card:
        card.stability  = self._init_stability(rating)
        card.difficulty = self._init_difficulty(rating)
        card.reps      += 1

        if rating == Rating.Again:
            card.state          = State.Learning
            card.scheduled_days = 0
            card.due            = now + timedelta(minutes=1)
        elif rating == Rating.Hard:
            card.state          = State.Learning
            card.scheduled_days = 0
            card.due            = now + timedelta(minutes=5)
        elif rating == Rating.Good:
            card.state          = State.Learning
            card.scheduled_days = 0
            card.due            = now + timedelta(minutes=10)
        else:  # Easy
            card.state          = State.Review
            ivl                 = self._next_interval(card.stability)
            card.scheduled_days = ivl
            card.due            = now + timedelta(days=ivl)
        return card

    def _review_learning(
        self, card: Card, rating: Rating, now: datetime
    ) -> Card:
        card.stability  = self._short_term_stability(
            card.stability, rating
        )
        card.difficulty = self._next_difficulty(card.difficulty, rating)
        card.reps      += 1

        if rating == Rating.Again:
            card.state          = State.Learning
            card.scheduled_days = 0
            card.due            = now + timedelta(minutes=1)
        elif rating in (Rating.Hard, Rating.Good):
            card.state          = State.Learning
            card.scheduled_days = 0
            card.due            = now + timedelta(minutes=10)
        else:  # Easy
            card.state          = State.Review
            ivl                 = self._next_interval(card.stability)
            card.scheduled_days = ivl
            card.due            = now + timedelta(days=ivl)
        return card

    def _review_review(
        self, card: Card, rating: Rating, now: datetime
    ) -> Card:
        r_val           = card.retention
        card.difficulty = self._next_difficulty(card.difficulty, rating)
        card.reps      += 1

        if rating == Rating.Again:
            card.lapses        += 1
            card.stability      = self._next_forget_stability(
                card.difficulty, card.stability, r_val
            )
            card.state          = State.Relearn
            card.scheduled_days = 0
            card.due            = now + timedelta(minutes=10)
        else:
            card.stability      = self._next_recall_stability(
                card.difficulty, card.stability, r_val, rating
            )
            ivl                 = self._next_interval(card.stability)
            card.state          = State.Review
            card.scheduled_days = ivl
            card.due            = now + timedelta(days=ivl)
        return card

    def _review_relearn(
        self, card: Card, rating: Rating, now: datetime
    ) -> Card:
        card.stability  = self._short_term_stability(
            card.stability, rating
        )
        card.difficulty = self._next_difficulty(card.difficulty, rating)
        card.reps      += 1

        if rating == Rating.Again:
            card.state          = State.Relearn
            card.scheduled_days = 0
            card.due            = now + timedelta(minutes=10)
        else:
            card.state          = State.Review
            ivl                 = self._next_interval(card.stability)
            card.scheduled_days = ivl
            card.due            = now + timedelta(days=ivl)
        return card

    # ── Utility ───────────────────────────────────────────────────

    def get_retrievability(self, card: Card, now: datetime = None) -> float:
        if now is None:
            now = datetime.now(timezone.utc)
        if card.stability <= 0:
            return 0.0
        elapsed = max(0, (now - card.due).days)
        return math.exp(math.log(0.9) * elapsed / card.stability)

    def due_cards(
        self, cards: list[Card], now: datetime = None
    ) -> list[Card]:
        if now is None:
            now = datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        result = []
        for c in cards:
            due = c.due
            if due.tzinfo is None:
                due = due.replace(tzinfo=timezone.utc)
            if due <= now:
                result.append(c)
        return result