from __future__ import annotations

import sys
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path


def _normalize_macro_name(name: str) -> str:
    return name.replace("\\[", "[").replace("\\]", "]")


@dataclass(frozen=True)
class _PinSpec:
    macro_name: str | None
    fixed_position: tuple[float, float] | None
    offset: tuple[float, float]


@dataclass(frozen=True)
class _NetSpec:
    weight: float
    pins: tuple[_PinSpec, ...]


class MacroPlacementWirelengthEvaluator:
    def __init__(self, *, netlist_path: str | Path) -> None:
        self.netlist_path = Path(netlist_path).resolve()
        placement_util_os, plc_client_os = _load_macroplacement_modules()
        plc = placement_util_os.create_placement_cost(
            plc_client=plc_client_os,
            netlist_file=self.netlist_path.as_posix(),
        )
        self._plc = plc
        self._movable_macro_names = frozenset(
            _normalize_macro_name(plc.indices_to_mod_name[index]) for index in plc.hard_macro_indices
        )
        self._nets = self._build_nets()

    @cached_property
    def movable_macro_names(self) -> frozenset[str]:
        return self._movable_macro_names

    def evaluate(self, placement: dict[str, tuple[float, float]]) -> float:
        normalized_placement = {_normalize_macro_name(name): coords for name, coords in placement.items()}
        total_hpwl = 0.0
        for net in self._nets:
            x_coords = []
            y_coords = []
            for pin in net.pins:
                if pin.fixed_position is not None:
                    x, y = pin.fixed_position
                else:
                    assert pin.macro_name is not None
                    x0, y0 = normalized_placement[pin.macro_name]
                    dx, dy = pin.offset
                    x = x0 + dx
                    y = y0 + dy
                x_coords.append(x)
                y_coords.append(y)
            total_hpwl += net.weight * ((max(x_coords) - min(x_coords)) + (max(y_coords) - min(y_coords)))
        return float(total_hpwl)

    def _build_nets(self) -> tuple[_NetSpec, ...]:
        nets = []
        for driver_pin_name, sink_pin_names in self._plc.nets.items():
            driver_spec = self._pin_spec(driver_pin_name)
            if driver_spec is None:
                continue
            pin_specs = [driver_spec]
            for sink_pin_name in sink_pin_names:
                sink_spec = self._pin_spec(sink_pin_name)
                if sink_spec is not None:
                    pin_specs.append(sink_spec)
            if len(pin_specs) < 2:
                continue
            if not any(pin.macro_name in self._movable_macro_names for pin in pin_specs if pin.macro_name is not None):
                continue
            driver_idx = self._plc.mod_name_to_indices[driver_pin_name]
            weight = float(self._plc.modules_w_pins[driver_idx].get_weight())
            nets.append(_NetSpec(weight=weight, pins=tuple(pin_specs)))
        return tuple(nets)

    def _pin_spec(self, pin_name: str) -> _PinSpec | None:
        if pin_name not in self._plc.mod_name_to_indices:
            return None
        pin_idx = self._plc.mod_name_to_indices[pin_name]
        pin = self._plc.modules_w_pins[pin_idx]
        pin_type = pin.get_type()
        if pin_type == "PORT":
            return _PinSpec(macro_name=None, fixed_position=pin.get_pos(), offset=(0.0, 0.0))
        if pin_type != "MACRO_PIN":
            return None
        ref_idx = self._plc.get_ref_node_id(pin_idx)
        if ref_idx == -1:
            return None
        ref_node = self._plc.modules_w_pins[ref_idx]
        ref_name = _normalize_macro_name(ref_node.get_name())
        x_offset, y_offset = pin.get_offset()
        if ref_name in self._movable_macro_names:
            return _PinSpec(macro_name=ref_name, fixed_position=None, offset=(x_offset, y_offset))
        ref_x, ref_y = ref_node.get_pos()
        return _PinSpec(
            macro_name=None,
            fixed_position=(ref_x + x_offset, ref_y + y_offset),
            offset=(0.0, 0.0),
        )


def _load_macroplacement_modules():
    codeelements_root = _macroplacement_codeelements_root()
    if str(codeelements_root) not in sys.path:
        sys.path.insert(0, str(codeelements_root))
    from Plc_client import placement_util_os  # type: ignore
    from Plc_client import plc_client_os  # type: ignore

    return placement_util_os, plc_client_os


def _macroplacement_codeelements_root() -> Path:
    return Path(__file__).resolve().parents[1] / "tmp" / "MacroPlacement" / "CodeElements"
