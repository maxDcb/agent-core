from __future__ import annotations

from agent_core.logging_utils import get_logger
from agent_core.specialists.types import SpecialistProfile

logger = get_logger(__name__)


class SpecialistRegistry:
    def __init__(self) -> None:
        self._profiles: dict[str, SpecialistProfile] = {}
        logger.debug("Initialized empty specialist registry")

    def register(self, profile: SpecialistProfile) -> None:
        self._profiles[profile.profile_id] = profile
        logger.info("Registered specialist profile", extra={"profile_id": profile.profile_id})

    def get(self, profile_id: str) -> SpecialistProfile | None:
        return self._profiles.get(profile_id)

    def require(self, profile_id: str) -> SpecialistProfile:
        profile = self.get(profile_id)
        if profile is None:
            raise KeyError(f"Unknown specialist profile: {profile_id}")
        return profile

    def list_profile_ids(self) -> list[str]:
        return sorted(self._profiles.keys())
