import fluxway
from fluxway.common import operator_policy
from fluxway.common import types as common_types
from fluxway.common.types import configs


POLICY_NAMES = (
    "ArgumentType",
    "AsyncPolicy",
    "BarrierConfig",
    "BarrierType",
    "PeriodicBarrierConfig",
    "SchedulePolicy",
    "StandardBarrierConfig",
    "StatePolicy",
)


def test_operator_policies_have_one_import_path():
    for name in POLICY_NAMES:
        policy_type = getattr(operator_policy, name)
        assert policy_type.__module__ == "fluxway.common.operator_policy"
        assert not hasattr(configs, name)
        assert not hasattr(common_types, name)
        assert not hasattr(fluxway, name)
