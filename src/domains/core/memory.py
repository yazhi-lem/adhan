"""Memory Manager — ``UserContext`` and ``PersonalizationProfile`` persistence.

The production target is **Postgres, the same instance family as the legal
pipeline's pgvector setup** — deliberately not a new service, so Core Domain adds
no new infra surface and no new DPDP audit surface (§6). Phase 1 ships the
protocol plus an in-memory implementation, which is enough to exercise the
orchestrator; the Postgres store lands with that instance rather than ahead of it.

**Consent is a required keyword argument, not a config default.** §4.3 makes
retention of ``previous_interactions`` and reflection outcomes a hard sequencing
dependency on the DPDP consent flow — no training on production logs until that
mechanism is live and audited. Putting ``consented`` in the signature forces each
call site to decide, instead of inheriting a default that silently happens to be
permissive. Both write paths return whether the record was actually retained.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from domains.core.schemas import (
    Interaction,
    PersonalizationProfile,
    PreferencePair,
    UserContext,
)


class MemoryStore(Protocol):
    """The persistence contract the kernel depends on."""

    def load(self, user_id: str) -> UserContext | None:
        ...

    def save_profile(self, profile: PersonalizationProfile) -> None:
        ...

    def append_interaction(
        self, user_id: str, interaction: Interaction, *, consented: bool
    ) -> bool:
        ...

    def record_preference_pair(self, pair: PreferencePair, *, consented: bool) -> bool:
        ...


@dataclass
class InMemoryStore:
    """Process-local store. Fine for tests and single-turn CLI runs, nothing else.

    Deliberately not a cache in front of anything — when the Postgres store
    exists, this stays as the test double rather than becoming a layer.
    """

    max_interactions: int = 50
    _profiles: dict[str, PersonalizationProfile] = field(default_factory=dict)
    _interactions: dict[str, list[Interaction]] = field(default_factory=dict)
    _pairs: list[PreferencePair] = field(default_factory=list)
    #: Writes refused for lack of consent, kept as a counter so the gap is
    #: observable in dev rather than invisible.
    declined_writes: int = 0

    def load(self, user_id: str) -> UserContext | None:
        profile = self._profiles.get(user_id)
        if profile is None:
            return None
        return UserContext(
            user_id=user_id,
            profile=profile,
            previous_interactions=list(self._interactions.get(user_id, [])),
        )

    def save_profile(self, profile: PersonalizationProfile) -> None:
        self._profiles[profile.user_id] = profile
        self._interactions.setdefault(profile.user_id, [])

    def append_interaction(
        self, user_id: str, interaction: Interaction, *, consented: bool
    ) -> bool:
        if not consented:
            self.declined_writes += 1
            return False
        turns = self._interactions.setdefault(user_id, [])
        turns.append(interaction)
        if len(turns) > self.max_interactions:
            del turns[: len(turns) - self.max_interactions]
        return True

    def record_preference_pair(self, pair: PreferencePair, *, consented: bool) -> bool:
        """Retain a reflection-derived (chosen, rejected) pair for the DPO loop.

        This is the flywheel's collection point (§4.1) and therefore exactly the
        path §4.3 gates: a pair harvested from an unconsented conversation is
        still production data, and is dropped.
        """
        if not consented:
            self.declined_writes += 1
            return False
        self._pairs.append(pair)
        return True

    @property
    def preference_pairs(self) -> tuple[PreferencePair, ...]:
        return tuple(self._pairs)
