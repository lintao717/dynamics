"""
Parameter specification and nested ModelParams replacement.

Stage 1 (CHECKED broadcast): 4 parameters
  propagation.beta_M  [0, 1]
  activation.alpha_0  [-6, 2]
  decay.gamma_0       [-6, 6]
  viral.beta_V        [0, 1]
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from dynamics_simulation.config import ModelParams


@dataclass(frozen=True)
class ParameterSpec:
    """Specification of one calibratable parameter."""

    path: str  # dotted path, e.g. "propagation.beta_M"
    low: float
    high: float

    def __post_init__(self):
        if self.low >= self.high:
            raise ValueError(
                f"low={self.low} must be < high={self.high} for {self.path}"
            )


class Stage1ParameterSet:
    """Restricted 4-parameter candidate set for CHECKED broadcast calibration.

    These are CANDIDATE calibration parameters — NOT proven to be identifiable
    from CHECKED alone. Currently fitted against active_count A(t) only; the
    other loss targets (cumulative_users, interaction_count, peak_time, final_size)
    are zero-weighted. True identifiability requires multi-target recovery
    experiments with known θ* and checkpointed parameter recovery tests.
    """

    @staticmethod
    def to_specs() -> tuple[ParameterSpec, ...]:
        return (
            ParameterSpec("propagation.beta_M", 0.0, 1.0),
            ParameterSpec("activation.alpha_0", -6.0, 2.0),
            ParameterSpec("decay.gamma_0", -6.0, 6.0),
            ParameterSpec("viral.beta_V", 0.0, 1.0),
        )

    @staticmethod
    def bounds() -> list[tuple[float, float]]:
        specs = Stage1ParameterSet.to_specs()
        return [(s.low, s.high) for s in specs]


def apply_parameter_vector(
    base: ModelParams,
    specs: tuple[ParameterSpec, ...],
    values: list[float] | tuple[float, ...],
) -> ModelParams:
    """Apply a parameter vector to *base*, returning a new ModelParams.

    Args:
        base: Base ModelParams to clone and modify.
        specs: Ordered ParameterSpecs defining which fields to set.
        values: Parameter values in the same order as specs.

    Returns:
        A new ModelParams with the specified fields replaced.

    Raises:
        ValueError: If length mismatch, out-of-bounds, or non-finite values.
        AttributeError: If a path does not exist on ModelParams.
    """
    if len(values) != len(specs):
        raise ValueError(
            f"values length {len(values)} != specs length {len(specs)}"
        )

    for spec, val in zip(specs, values):
        if not math.isfinite(val):
            raise ValueError(
                f"Non-finite value {val} for {spec.path}"
            )
        if not (spec.low <= val <= spec.high):
            raise ValueError(
                f"Value {val} for {spec.path} outside bounds "
                f"[{spec.low}, {spec.high}]"
            )

    p = base
    for spec, val in zip(specs, values):
        parts = spec.path.split(".")
        if len(parts) != 2:
            raise ValueError(f"Expected 'section.field' path, got {spec.path}")

        section, field = parts
        section_obj = getattr(p, section)
        new_section = replace(section_obj, **{field: float(val)})
        p = replace(p, **{section: new_section})

    return p
