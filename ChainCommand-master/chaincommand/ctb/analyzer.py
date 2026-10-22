"""Clear-to-Build analyzer — determines if all components are available to build."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..bom.models import BOMTree
from ..utils.logging_config import get_logger

log = get_logger(__name__)


@dataclass
class PartShortage:
    """A single component shortage."""
    part_id: str
    name: str
    required_qty: float
    available_qty: float
    shortage_qty: float
    lead_time_days: int
    estimated_availability_date: Optional[str] = None  # ISO date
    supplier_count: int = 0
    is_critical: bool = False  # True if on critical path


@dataclass
class CTBReport:
    """Clear-to-Build report for an assembly."""
    assembly_id: str
    build_quantity: float
    is_clear: bool  # True if all components available
    clear_percentage: float  # 0-100, how much of build qty is achievable
    shortages: List[PartShortage] = field(default_factory=list)
    total_parts: int = 0
    available_parts: int = 0
    longest_wait_days: int = 0
    estimated_build_date: Optional[str] = None
    total_material_cost: float = 0.0


class CTBAnalyzer:
    """Analyze build readiness given BOM trees and inventory levels."""

    def __init__(self) -> None:
        pass

    def analyze(
        self,
        bom_tree: BOMTree,
        assembly_id: str,
        build_qty: float,
        inventory: Dict[str, float],
        on_order: Optional[Dict[str, float]] = None,
    ) -> CTBReport:
        """Determine if assembly can be built with current inventory.

        Args:
            bom_tree: BOM tree containing the assembly
            assembly_id: Root part ID to build
            build_qty: Number of units to build
            inventory: Dict of part_id -> available quantity
            on_order: Dict of part_id -> quantity on order (optional)
        """
        on_order = on_order or {}

        # Explode BOM to get all required components
        explosion = bom_tree.explode(assembly_id, parent_qty=build_qty)

        if not explosion:
            return CTBReport(
                assembly_id=assembly_id,
                build_quantity=build_qty,
                is_clear=True,
                clear_percentage=100.0,
            )

        shortages: List[PartShortage] = []
        total_parts = 0
        available_parts = 0
        longest_wait = 0
        total_cost = 0.0

        # Aggregate requirements by part_id — only check "buy" items
        # ("make" items are assembled in-house, not sourced from inventory)
        requirements: Dict[str, float] = {}
        part_info: Dict[str, dict] = {}
        for row in explosion:
            if row.make_or_buy == "make":
                continue  # skip sub-assemblies built in-house
            if row.part_id in requirements:
                requirements[row.part_id] += row.extended_quantity
            else:
                requirements[row.part_id] = row.extended_quantity
                part_info[row.part_id] = {
                    "name": row.name,
                    "lead_time_days": row.lead_time_days,
                    "unit_cost": row.unit_cost,
                    "supplier_count": row.supplier_count,
                    "make_or_buy": row.make_or_buy,
                }

        # Check each component
        critical_parts = self._find_critical_parts(bom_tree, assembly_id)

        for part_id, required_qty in requirements.items():
            info = part_info[part_id]
            # NOTE: on_order is treated as immediately available. In production,
            # this should be discounted by expected arrival date vs. build date.
            available = inventory.get(part_id, 0.0) + on_order.get(part_id, 0.0)
            total_parts += 1
            total_cost += required_qty * info["unit_cost"]

            if available >= required_qty:
                available_parts += 1
            else:
                shortage_qty = required_qty - available
                is_critical = part_id in critical_parts

                shortages.append(PartShortage(
                    part_id=part_id,
                    name=info["name"],
                    required_qty=round(required_qty, 2),
                    available_qty=round(available, 2),
                    shortage_qty=round(shortage_qty, 2),
                    lead_time_days=info["lead_time_days"],
                    supplier_count=info["supplier_count"],
                    is_critical=is_critical,
                ))

                longest_wait = max(longest_wait, info["lead_time_days"])

        # Calculate clear percentage (what fraction of build qty is achievable)
        if not requirements:
            clear_pct = 100.0
        else:
            # Min ratio across all components
            ratios = []
            for part_id, required_qty in requirements.items():
                available = inventory.get(part_id, 0.0) + on_order.get(part_id, 0.0)
                ratio = min(1.0, available / required_qty) if required_qty > 0 else 1.0
                ratios.append(ratio)
            clear_pct = min(ratios) * 100.0 if ratios else 100.0

        # Sort shortages by criticality then lead time
        shortages.sort(key=lambda s: (-s.is_critical, -s.lead_time_days))

        report = CTBReport(
            assembly_id=assembly_id,
            build_quantity=build_qty,
            is_clear=len(shortages) == 0,
            clear_percentage=round(clear_pct, 1),
            shortages=shortages,
            total_parts=total_parts,
            available_parts=available_parts,
            longest_wait_days=longest_wait,
            total_material_cost=round(total_cost, 2),
        )

        log.info(
            "ctb_analysis_complete",
            assembly_id=assembly_id,
            is_clear=report.is_clear,
            clear_pct=report.clear_percentage,
            shortages=len(shortages),
        )
        return report

    def analyze_multi(
        self,
        bom_trees: Dict[str, BOMTree],
        build_plan: Dict[str, float],
        inventory: Dict[str, float],
        on_order: Optional[Dict[str, float]] = None,
    ) -> List[CTBReport]:
        """Analyze CTB for multiple assemblies with shared inventory tracking.

        Each assembly's analysis uses a running copy of the inventory pool.
        After each assembly is analyzed, the parts it would consume are
        subtracted so that subsequent assemblies see the reduced availability.
        """
        # Work on a mutable copy so the caller's dict is not modified
        remaining_inventory = dict(inventory)
        remaining_on_order = dict(on_order) if on_order else {}
        reports = []

        for assembly_id, qty in build_plan.items():
            tree = bom_trees.get(assembly_id)
            if tree:
                report = self.analyze(
                    tree, assembly_id, qty, remaining_inventory, remaining_on_order,
                )
                reports.append(report)

                # Only subtract consumed parts from the shared pool when the
                # assembly is fully buildable; non-buildable assemblies should
                # not deplete inventory meant for other assemblies.
                if report.is_clear:
                    explosion = tree.explode(assembly_id, parent_qty=qty)
                    for row in explosion:
                        if row.make_or_buy == "make":
                            continue
                        needed = row.extended_quantity
                        # Consume from on_order first, then inventory
                        oo = remaining_on_order.get(row.part_id, 0.0)
                        from_oo = min(oo, needed)
                        remaining_on_order[row.part_id] = oo - from_oo
                        needed -= from_oo

                        inv = remaining_inventory.get(row.part_id, 0.0)
                        from_inv = min(inv, needed)
                        remaining_inventory[row.part_id] = inv - from_inv

        return reports

    def _find_critical_parts(self, bom_tree: BOMTree, root_id: str) -> set:
        """Find parts on the critical path (longest lead-time chain)."""
        critical = set()
        self._trace_critical_path(bom_tree, root_id, critical, set())
        return critical

    def _trace_critical_path(self, tree: BOMTree, part_id: str, critical: set, visited: set) -> int:
        if part_id in visited:
            return 0  # cycle detected — break recursion
        visited.add(part_id)

        children = tree.get_children(part_id)
        if not children:
            item = tree.items.get(part_id)
            visited.discard(part_id)
            return item.lead_time_days if item else 0

        max_path = 0
        max_child = None
        for child in children:
            path = self._trace_critical_path(tree, child.part_id, critical, visited)
            if path >= max_path:
                max_path = path
                max_child = child.part_id

        if max_child:
            critical.add(max_child)

        item = tree.items.get(part_id)
        visited.discard(part_id)
        return max_path + (item.lead_time_days if item else 0)
