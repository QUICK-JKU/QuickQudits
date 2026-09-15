"""Gate-dependent noise configuration.

This module deliberately keeps gate-noise policy separate from circuit execution.
A :class:`GateNoiseConfig` materializes its rules as the ordinary
``NoiseChannel`` operations already understood by the simulators.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union

import numpy as np

if TYPE_CHECKING:
    from .circuit import QuantumCircuit


NoiseModel = Union[float, np.ndarray]


def _copy_model(model: Optional[NoiseModel]) -> Optional[NoiseModel]:
    """Copy array models so callers cannot mutate stored configuration."""
    if isinstance(model, np.ndarray):
        return model.copy()
    return model


@dataclass(frozen=True)
class _GateNoiseRule:
    before: Optional[NoiseModel]
    after: Optional[NoiseModel]
    color: str


class GateNoiseConfig:
    """Configure local noise immediately before and/or after named gates.

    Rules match gates by their exact, case-sensitive ``Gate.name``. Applying a
    configuration creates a circuit containing regular ``NoiseChannel``
    operations, so execution backends do not need special knowledge of this
    class.

    Notes
    -----
    One- and two-qudit gates are supported. For a two-qudit gate, the configured
    one-qudit noise model is applied independently to each operand. Noise models
    use the same format as :meth:`QuantumCircuit.add_noise_channel`: either a
    scalar total Weyl-error probability or a ``(d, d)`` probability matrix.
    """

    def __init__(self) -> None:
        self._rules: dict[str, _GateNoiseRule] = {}

    def __repr__(self) -> str:
        return f"GateNoiseConfig(gates={list(self._rules)!r})"

    def __len__(self) -> int:
        return len(self._rules)

    def set(
        self,
        gate_name: str,
        *,
        before: Optional[NoiseModel] = None,
        after: Optional[NoiseModel] = None,
        color: str = "#FF0000",
    ) -> None:
        """Set (or replace) the noise rule for a gate name.

        Parameters
        ----------
        gate_name : str
            Exact gate name to match, for example ``"H"`` or ``"X"``.
        before, after : float or np.ndarray, optional
            One-qudit Weyl noise to insert on every operand before and/or after
            each matching gate. At least one must be supplied. For a two-qudit
            gate the same model is applied independently to both operands.
        color : str, default="#FF0000"
            Base visualization color for both inserted channels.
        """
        if not isinstance(gate_name, str) or not gate_name:
            raise ValueError("gate_name must be a non-empty string")
        if before is None and after is None:
            raise ValueError("At least one of `before` or `after` must be provided")
        if not isinstance(color, str):
            raise TypeError("color must be a string")

        self._rules[gate_name] = _GateNoiseRule(
            before=_copy_model(before),
            after=_copy_model(after),
            color=color,
        )

    def remove(self, gate_name: str) -> None:
        """Remove the rule for ``gate_name`` if one exists."""
        self._rules.pop(gate_name, None)

    def clear(self) -> None:
        """Remove all configured gate-noise rules."""
        self._rules.clear()

    def apply(
        self, circuit: QuantumCircuit, *, in_place: bool = False
    ) -> QuantumCircuit:
        """Materialize configured noise channels into ``circuit``.

        By default a noisy copy is returned and the input circuit is unchanged.
        With ``in_place=True``, the input circuit is updated atomically and also
        returned. Gates added after this method is called are not affected.

        Raises
        ------
        ValueError
            If a rule matches a gate that does not act on one or two qudits, or
            if an underlying noise model is invalid for the circuit dimension.
        TypeError
            If ``circuit`` is not a ``QuantumCircuit``.
        """
        from .circuit import QuantumCircuit

        if not isinstance(circuit, QuantumCircuit):
            raise TypeError("circuit must be a QuantumCircuit")

        target = circuit.copy()
        matches: list[tuple[int, tuple[int, ...], _GateNoiseRule]] = []

        for index, (gate, qudits, _) in enumerate(circuit.ops):
            rule = self._rules.get(gate.name)
            if rule is None:
                continue
            if gate.n_qudits not in (1, 2) or len(qudits) != gate.n_qudits:
                raise ValueError(
                    f"Gate-noise rules support only one- and two-qudit gates; "
                    f"{gate.name!r} acts on {gate.n_qudits} qudits"
                )
            matches.append((index, qudits, rule))

        for index, qudits, rule in reversed(matches):
            # Insert at a fixed program boundary. Reversing the operands keeps
            # the resulting channels in the gate's operand order.
            if rule.after is not None:
                for qudit in reversed(qudits):
                    target.insert_noise_channel(
                        index + 1,
                        qudit,
                        _copy_model(rule.after),
                        color=rule.color,
                    )
            if rule.before is not None:
                for qudit in reversed(qudits):
                    target.insert_noise_channel(
                        index,
                        qudit,
                        _copy_model(rule.before),
                        color=rule.color,
                    )

        if not in_place:
            return target

        circuit._ops = target._ops
        circuit.noise_registry = target.noise_registry
        circuit._noise_model_cache = target._noise_model_cache
        circuit._next_noise_model_id = target._next_noise_model_id
        circuit._kraus_operator_cache.clear()
        return circuit


__all__ = ["GateNoiseConfig", "NoiseModel"]
