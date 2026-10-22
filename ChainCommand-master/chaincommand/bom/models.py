"""BOM data models and tree operations."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from ..utils.logging_config import get_logger

log = get_logger(__name__)


@dataclass
class BOMItem:
    """A single item in the Bill of Materials."""
    part_id: str
    name: str
    parent_id: Optional[str] = None  # None = top-level assembly
    quantity_per: float = 1.0  # qty needed per parent unit
    unit_cost: float = 0.0
    lead_time_days: int = 7
    level: int = 0  # BOM level (0=finished good, 1=sub-assembly, 2+=component)
    suppliers: List[str] = field(default_factory=list)
    is_phantom: bool = False  # phantom = pass-through sub-assembly
    scrap_rate: float = 0.0  # expected scrap percentage (0 <= scrap_rate < 1)
    make_or_buy: str = "buy"  # "make" or "buy"

    def __post_init__(self) -> None:
        if self.scrap_rate < 0:
            raise ValueError(
                f"BOMItem '{self.part_id}': scrap_rate must be >= 0, got {self.scrap_rate}"
            )
        if self.scrap_rate >= 1.0:
            raise ValueError(
                f"BOMItem '{self.part_id}': scrap_rate must be < 1.0 "
                f"(got {self.scrap_rate}); a 100% scrap rate means "
                f"no usable output and causes division by zero in cost/quantity calculations"
            )


@dataclass
class BOMExplosionRow:
    """A single row in a BOM explosion report."""
    level: int
    part_id: str
    name: str
    quantity_per_parent: float
    extended_quantity: float  # cumulative qty needed for 1 top-level unit
    unit_cost: float
    extended_cost: float  # extended_quantity * unit_cost
    lead_time_days: int
    supplier_count: int
    make_or_buy: str


class BOMTree:
    """Multi-tier BOM tree with explosion, where-used, and cost rollup."""

    def __init__(self, items: Optional[List[BOMItem]] = None) -> None:
        self._items: Dict[str, BOMItem] = {}
        self._children: Dict[str, List[str]] = {}  # parent_id -> [child_ids]
        self._parents: Dict[str, str] = {}  # child_id -> parent_id
        if items:
            for item in items:
                self.add_item(item)

    @property
    def items(self) -> Dict[str, BOMItem]:
        return dict(self._items)

    @property
    def root_items(self) -> List[BOMItem]:
        """Top-level assemblies (no parent)."""
        return [item for item in self._items.values() if item.parent_id is None]

    def add_item(self, item: BOMItem) -> None:
        """Add an item to the BOM tree."""
        # Clean up old parent's _children if this part already exists under a different parent
        old_parent = self._parents.get(item.part_id)
        if old_parent is not None and old_parent != item.parent_id:
            if old_parent in self._children:
                self._children[old_parent] = [
                    c for c in self._children[old_parent] if c != item.part_id
                ]
        self._items[item.part_id] = item
        if item.parent_id is not None:
            children_list = self._children.setdefault(item.parent_id, [])
            if item.part_id not in children_list:
                children_list.append(item.part_id)
            self._parents[item.part_id] = item.parent_id
        elif item.part_id in self._parents:
            # Item moved to root — remove old parent mapping
            del self._parents[item.part_id]
        log.debug("bom_item_added", part_id=item.part_id, parent=item.parent_id)

    def remove_item(self, part_id: str) -> bool:
        """Remove an item and its children from the BOM tree."""
        if part_id not in self._items:
            return False
        # Remove children recursively
        for child_id in list(self._children.get(part_id, [])):
            self.remove_item(child_id)
        # Clean parent references
        parent = self._parents.pop(part_id, None)
        if parent and parent in self._children:
            self._children[parent] = [c for c in self._children[parent] if c != part_id]
        self._children.pop(part_id, None)
        del self._items[part_id]
        return True

    def get_children(self, part_id: str) -> List[BOMItem]:
        """Get direct children of a part."""
        child_ids = self._children.get(part_id, [])
        return [self._items[cid] for cid in child_ids if cid in self._items]

    def explode(self, part_id: str, parent_qty: float = 1.0) -> List[BOMExplosionRow]:
        """Multi-level BOM explosion — recursively expand all sub-assemblies.

        Returns a flat list of all components needed to build `parent_qty` units
        of the given part, with extended quantities and costs.
        """
        item = self._items.get(part_id)
        if item is None:
            return []

        result: List[BOMExplosionRow] = []
        self._explode_recursive(part_id, parent_qty, 0, result)
        return result

    def _explode_recursive(
        self, part_id: str, parent_qty: float, level: int,
        result: List[BOMExplosionRow], visited: Optional[Set[str]] = None,
    ) -> None:
        if visited is None:
            visited = set()
        if part_id in visited:
            log.warning("bom_cycle_detected", part_id=part_id)
            return
        visited.add(part_id)

        children = self.get_children(part_id)
        for child in children:
            # Account for scrap rate — always divide; when scrap_rate=0, divisor=1
            effective_qty = child.quantity_per * parent_qty / (1 - child.scrap_rate)

            # Phantom items pass through: recurse into their children but
            # do not emit a row for the phantom itself.
            if not child.is_phantom:
                row = BOMExplosionRow(
                    level=level + 1,
                    part_id=child.part_id,
                    name=child.name,
                    quantity_per_parent=child.quantity_per,
                    extended_quantity=round(effective_qty, 4),
                    unit_cost=child.unit_cost,
                    extended_cost=round(effective_qty * child.unit_cost, 2),
                    lead_time_days=child.lead_time_days,
                    supplier_count=len(child.suppliers),
                    make_or_buy=child.make_or_buy,
                )
                result.append(row)

            # Recurse into sub-assemblies (phantom children inherit parent level)
            if child.part_id in self._children:
                next_level = level if child.is_phantom else level + 1
                self._explode_recursive(child.part_id, effective_qty, next_level, result, visited)

        # Backtrack: allow this node to be visited under different parents
        visited.discard(part_id)

    def where_used(self, part_id: str) -> List[BOMItem]:
        """Find all assemblies that use a given part (reverse BOM lookup)."""
        result: List[BOMItem] = []
        self._where_used_recursive(part_id, result, set())
        return result

    def _where_used_recursive(self, part_id: str, result: List[BOMItem], visited: Set[str]) -> None:
        parent_id = self._parents.get(part_id)
        if parent_id and parent_id not in visited:
            visited.add(parent_id)
            parent = self._items.get(parent_id)
            if parent:
                result.append(parent)
                self._where_used_recursive(parent_id, result, visited)

    def cost_rollup(self, part_id: str, _visited: Optional[Set[str]] = None) -> float:
        """Calculate total cost of a part including all sub-components."""
        if _visited is None:
            _visited = set()
        if part_id in _visited:
            log.warning("cost_rollup_cycle_detected", part_id=part_id)
            return 0.0
        _visited.add(part_id)

        item = self._items.get(part_id)
        if item is None:
            _visited.discard(part_id)
            return 0.0

        children = self.get_children(part_id)
        if not children:
            _visited.discard(part_id)
            return item.unit_cost

        child_cost = sum(
            child.quantity_per * self.cost_rollup(child.part_id, _visited)
            / (1 - child.scrap_rate)
            for child in children
        )
        _visited.discard(part_id)
        return child_cost + item.unit_cost

    def critical_path(self, part_id: str, _visited: Optional[Set[str]] = None) -> int:
        """Calculate the longest lead-time path through the BOM tree."""
        if _visited is None:
            _visited = set()
        if part_id in _visited:
            log.warning("critical_path_cycle_detected", part_id=part_id)
            return 0
        _visited.add(part_id)

        item = self._items.get(part_id)
        if item is None:
            _visited.discard(part_id)
            return 0

        children = self.get_children(part_id)
        if not children:
            _visited.discard(part_id)
            return item.lead_time_days

        max_child_path = max(self.critical_path(c.part_id, _visited) for c in children)
        _visited.discard(part_id)
        return item.lead_time_days + max_child_path

    def depth(self, part_id: str, _visited: Optional[Set[str]] = None) -> int:
        """Get the max depth of the BOM tree from this part."""
        if _visited is None:
            _visited = set()
        if part_id in _visited:
            log.warning("depth_cycle_detected", part_id=part_id)
            return 0
        _visited.add(part_id)

        children = self.get_children(part_id)
        if not children:
            _visited.discard(part_id)
            return 0
        result = 1 + max(self.depth(c.part_id, _visited) for c in children)
        _visited.discard(part_id)
        return result

    def validate(self) -> List[str]:
        """Validate BOM tree integrity. Returns list of error messages."""
        errors: List[str] = []
        for part_id, item in self._items.items():
            # Check parent exists
            if item.parent_id is not None and item.parent_id not in self._items:
                errors.append(f"Part {part_id} references non-existent parent {item.parent_id}")
            # Check no circular references
            if self._has_cycle(part_id):
                errors.append(f"Circular reference detected at part {part_id}")
            # Check quantity > 0
            if item.quantity_per <= 0:
                errors.append(f"Part {part_id} has non-positive quantity_per: {item.quantity_per}")
        return errors

    def _has_cycle(self, start_id: str) -> bool:
        """Detect cycles using DFS — checks both parent chain and child subtree."""
        # Check parent chain (upward cycle)
        visited: Set[str] = set()
        current = start_id
        while current in self._parents:
            if current in visited:
                return True
            visited.add(current)
            current = self._parents[current]

        # Check child subtree (downward cycle) via DFS
        visiting: Set[str] = set()  # nodes on the current DFS path

        def _dfs(node_id: str) -> bool:
            if node_id in visiting:
                return True
            visiting.add(node_id)
            for child_id in self._children.get(node_id, []):
                if _dfs(child_id):
                    return True
            visiting.discard(node_id)
            return False

        return _dfs(start_id)
