from __future__ import annotations

from dataclasses import dataclass

from fluxway.common.io_lane import IOLane
from fluxway.control_plane.checkpointer.repository import CheckpointRepository


@dataclass(frozen=True, slots=True)
class CheckpointRepositorySession:
    repository: CheckpointRepository
    generation: int
    lane: IOLane

    @classmethod
    async def open(cls, directory: str) -> CheckpointRepositorySession:
        lane = IOLane("checkpoint-repository", capacity=1)
        try:
            repository, generation = await lane.call(cls._open_repository, directory)
            return cls(repository, generation, lane)
        except BaseException:
            await lane.close()
            raise

    @staticmethod
    def _open_repository(directory: str) -> tuple[CheckpointRepository, int]:
        repository = CheckpointRepository(directory)
        return repository, repository.claim_coordinator_generation()

    async def close(self) -> None:
        await self.lane.close()
