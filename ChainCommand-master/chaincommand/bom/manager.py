"""BOM Manager — high-level BOM operations and synthetic data generation."""
from __future__ import annotations

import random
from typing import Dict, List, Optional

from ..utils.logging_config import get_logger
from .models import BOMItem, BOMTree

log = get_logger(__name__)


# Realistic BOM templates for supply chain products
_ASSEMBLY_TEMPLATES = {
    "PCB Assembly": {
        "sub_assemblies": ["Main PCB", "Power Module"],
        "components": {
            "Main PCB": ["MCU Chip", "Memory IC", "Capacitor 100nF", "Resistor 10K", "Connector USB-C"],
            "Power Module": ["Voltage Regulator", "Inductor 10uH", "Capacitor 470uF", "Diode Schottky"],
        },
    },
    "Sensor Module": {
        "sub_assemblies": ["Sensor Board", "Housing"],
        "components": {
            "Sensor Board": ["Temperature Sensor", "ADC Chip", "Capacitor 10nF", "Resistor 4.7K"],
            "Housing": ["Plastic Case", "Gasket", "Mounting Screws 4pk"],
        },
    },
    "Motor Assembly": {
        "sub_assemblies": ["Motor Unit", "Controller Board"],
        "components": {
            "Motor Unit": ["Stator Core", "Rotor", "Bearing 6201", "Shaft 8mm"],
            "Controller Board": ["Motor Driver IC", "MOSFET N-Ch", "Heatsink", "Capacitor 1000uF"],
        },
    },
}


class BOMManager:
    """High-level BOM management with synthetic data generation."""

    def __init__(self) -> None:
        self._trees: Dict[str, BOMTree] = {}

    @property
    def assemblies(self) -> Dict[str, BOMTree]:
        return dict(self._trees)

    def create_tree(self, assembly_id: str, items: Optional[List[BOMItem]] = None) -> BOMTree:
        """Create a new BOM tree for an assembly."""
        tree = BOMTree(items)
        self._trees[assembly_id] = tree
        log.info("bom_tree_created", assembly_id=assembly_id, items=len(tree.items))
        return tree

    def get_tree(self, assembly_id: str) -> Optional[BOMTree]:
        return self._trees.get(assembly_id)

    def generate_synthetic_boms(self, n_assemblies: int = 5, seed: int = 42) -> List[BOMTree]:
        """Generate realistic synthetic BOM trees."""
        rng = random.Random(seed)
        templates = list(_ASSEMBLY_TEMPLATES.items())
        trees: List[BOMTree] = []

        for i in range(n_assemblies):
            tmpl_name, tmpl = templates[i % len(templates)]
            assembly_id = f"ASM-{i+1:04d}"

            items: List[BOMItem] = []

            # Top-level assembly
            items.append(BOMItem(
                part_id=assembly_id,
                name=f"{tmpl_name} v{i+1}",
                parent_id=None,
                quantity_per=1.0,
                unit_cost=rng.uniform(5, 20),  # assembly labor cost
                lead_time_days=rng.randint(1, 3),
                level=0,
                make_or_buy="make",
            ))

            # Sub-assemblies
            for j, sa_name in enumerate(tmpl["sub_assemblies"]):
                sa_id = f"{assembly_id}-SA{j+1:02d}"
                items.append(BOMItem(
                    part_id=sa_id,
                    name=sa_name,
                    parent_id=assembly_id,
                    quantity_per=1.0,
                    unit_cost=rng.uniform(2, 8),
                    lead_time_days=rng.randint(2, 7),
                    level=1,
                    make_or_buy="make",
                ))

                # Components for this sub-assembly
                component_names = tmpl["components"].get(sa_name, [])
                for k, comp_name in enumerate(component_names):
                    comp_id = f"{sa_id}-C{k+1:02d}"
                    n_suppliers = rng.randint(1, 3)
                    supplier_ids = [f"SUP-{sid:04d}" for sid in rng.sample(range(1, 21), n_suppliers)]
                    items.append(BOMItem(
                        part_id=comp_id,
                        name=comp_name,
                        parent_id=sa_id,
                        quantity_per=float(rng.choice([1, 2, 4, 10])),
                        unit_cost=rng.uniform(0.01, 15.0),
                        lead_time_days=rng.randint(3, 21),
                        level=2,
                        suppliers=supplier_ids,
                        scrap_rate=rng.uniform(0, 0.05),
                        make_or_buy="buy",
                    ))

            tree = self.create_tree(assembly_id, items)
            trees.append(tree)

        log.info("synthetic_boms_generated", count=n_assemblies)
        return trees

    def find_single_source_risks(self) -> List[Dict]:
        """Find components with only one supplier (single-source risk).

        Distinguishes between zero suppliers (data quality issue) and
        exactly one supplier (true single-source risk).
        """
        risks = []
        for assembly_id, tree in self._trees.items():
            for part_id, item in tree.items.items():
                if item.make_or_buy != "buy":
                    continue
                n = len(item.suppliers)
                if n == 0:
                    risks.append({
                        "assembly_id": assembly_id,
                        "part_id": part_id,
                        "name": item.name,
                        "supplier_count": 0,
                        "unit_cost": item.unit_cost,
                        "lead_time_days": item.lead_time_days,
                        "risk_type": "no_supplier_data",
                    })
                    log.warning(
                        "no_supplier_data",
                        assembly_id=assembly_id,
                        part_id=part_id,
                    )
                elif n == 1:
                    risks.append({
                        "assembly_id": assembly_id,
                        "part_id": part_id,
                        "name": item.name,
                        "supplier_count": 1,
                        "unit_cost": item.unit_cost,
                        "lead_time_days": item.lead_time_days,
                        "risk_type": "single_source",
                    })
        return risks

    def find_long_lead_items(self, threshold_days: int = 14) -> List[Dict]:
        """Find components with lead time exceeding threshold."""
        results = []
        for assembly_id, tree in self._trees.items():
            for part_id, item in tree.items.items():
                if item.lead_time_days >= threshold_days:
                    results.append({
                        "assembly_id": assembly_id,
                        "part_id": part_id,
                        "name": item.name,
                        "lead_time_days": item.lead_time_days,
                        "level": item.level,
                    })
        return results

    def get_summary(self) -> Dict:
        """Get summary statistics across all BOM trees."""
        total_items = 0
        total_cost = 0.0
        max_depth = 0
        max_lead = 0
        counted_parts: set = set()
        costed_roots: set = set()

        for _assembly_id, tree in self._trees.items():
            roots = tree.root_items
            for root in roots:
                # Count each item only once across all roots in this tree
                for part_id in tree.items:
                    if part_id not in counted_parts:
                        counted_parts.add(part_id)
                        total_items += 1
                # Only roll up cost once per root to avoid double-counting
                # shared components in multi-root trees
                if root.part_id not in costed_roots:
                    costed_roots.add(root.part_id)
                    total_cost += tree.cost_rollup(root.part_id)
                max_depth = max(max_depth, tree.depth(root.part_id))
                max_lead = max(max_lead, tree.critical_path(root.part_id))

        return {
            "assembly_count": len(self._trees),
            "total_items": total_items,
            "total_rollup_cost": round(total_cost, 2),
            "max_bom_depth": max_depth,
            "max_critical_path_days": max_lead,
            "single_source_risks": len(self.find_single_source_risks()),
        }
