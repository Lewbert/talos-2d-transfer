"""The simulated drivers must accept exactly what the real ones accept.

This has bitten twice, the same way both times: a caller passes a
parameter the sim lacks, the job raises TypeError on the DEVICE thread,
and the symptom is a control that quietly does nothing. Once for
``move_abs_um(speed_pps=…)`` (which made the whole scan path raise), and
once for ``move_rel_um`` (which made *go to sample* do nothing at all,
2026-09-18) — neither caught, because every test's stub manager swallowed
whatever it was handed.

A signature is a contract, and the sim exists to stand in for the
hardware. So it is checked, not remembered.
"""

from __future__ import annotations

import inspect

import pytest

from talos.hal.devices.zolix import ZolixXYRStage
from talos.hal.sim.sim_zolix import SimZolixXYRStage

#: The methods the application actually calls through the manager. A
#: method the sim lacks entirely is a larger failure than a mismatch, so
#: both are checked.
STAGE_API = ("move_abs_um", "move_rel_um", "move_continuous", "stop",
             "stop_axis", "get_position", "wait_idle")


def _signature(cls, name: str) -> inspect.Signature:
    """The method's signature WITHOUT ``self`` (read off the class, these
    are plain functions), so a positional count is the count a caller
    writes."""
    assert hasattr(cls, name), f"{cls.__name__} has no {name}()"
    parameters = list(inspect.signature(getattr(cls, name)).parameters.values())
    if parameters and parameters[0].name == "self":
        parameters = parameters[1:]
    return inspect.Signature(parameters)


@pytest.mark.parametrize("name", STAGE_API)
def test_the_sim_stage_accepts_what_the_real_stage_accepts(name):
    """Every parameter the real driver takes must exist on the sim, under
    the same name and of the same kind.

    Defaults are deliberately NOT compared: the sim may legitimately poll
    faster than hardware (``wait_idle``'s ``poll_s`` does), and a
    different default cannot break a caller. A missing parameter can, and
    does — see the module docstring.
    """
    real = _signature(ZolixXYRStage, name)
    sim = _signature(SimZolixXYRStage, name)
    for parameter in real.parameters.values():
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            continue
        assert parameter.name in sim.parameters, (
            f"SimZolixXYRStage.{name}() does not accept "
            f"{parameter.name!r}, which the real driver does — a caller "
            f"using it gets a TypeError on the device thread, and the "
            f"control it drives looks dead")
        assert sim.parameters[parameter.name].kind is parameter.kind, (
            f"{name}({parameter.name}) is "
            f"{sim.parameters[parameter.name].kind} on the sim and "
            f"{parameter.kind} on the real driver")


def test_every_call_the_app_can_make_binds_on_both_drivers():
    """The manager carries arguments as a TUPLE, so the calls to check are
    the positional ones. Every prefix that is valid on the real driver —
    from the shortest legal call to the longest — must also be valid on
    the sim, because that is exactly the set of calls callers make."""
    for name in STAGE_API:
        real = _signature(ZolixXYRStage, name)
        sim = _signature(SimZolixXYRStage, name)
        positional = [p for p in real.parameters.values()
                      if p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                                    inspect.Parameter.POSITIONAL_OR_KEYWORD)]
        required = [p for p in positional
                    if p.default is inspect.Parameter.empty]
        for count in range(len(required), len(positional) + 1):
            args = list(range(count))
            real.bind(*args)                # a legal call, by construction
            sim.bind(*args)                 # ...and it must stay legal here
