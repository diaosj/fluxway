from __future__ import annotations

from fluxway.common.ingestion import IngestionLayoutReceiver


class Receiver:
    async def activate_ingestion_layout(self, activation: object) -> None:
        return None


class InvalidReceiver:
    pass


def test_ingestion_receiver_is_structural_and_typed() -> None:
    assert isinstance(Receiver(), IngestionLayoutReceiver)
    assert not isinstance(InvalidReceiver(), IngestionLayoutReceiver)


def test_receiver_contract_exposes_only_typed_hook() -> None:
    assert IngestionLayoutReceiver.__name__ == "IngestionLayoutReceiver"
    assert callable(IngestionLayoutReceiver.activate_ingestion_layout)
