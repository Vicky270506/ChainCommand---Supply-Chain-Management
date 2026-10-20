"""Tests for BOM management module."""
import pytest

from chaincommand.bom.manager import BOMManager
from chaincommand.bom.models import BOMItem, BOMTree


@pytest.fixture
def sample_bom():
    """Create a sample 3-level BOM tree."""
    items = [
        BOMItem(part_id="ASM-001", name="Widget Assembly", parent_id=None, quantity_per=1.0,
                unit_cost=5.0, lead_time_days=2, level=0, make_or_buy="make"),
        BOMItem(part_id="SA-001", name="Sub-Assembly A", parent_id="ASM-001", quantity_per=1.0,
                unit_cost=3.0, lead_time_days=3, level=1, make_or_buy="make"),
        BOMItem(part_id="SA-002", name="Sub-Assembly B", parent_id="ASM-001", quantity_per=2.0,
                unit_cost=2.0, lead_time_days=4, level=1, make_or_buy="make"),
        BOMItem(part_id="C-001", name="Resistor", parent_id="SA-001", quantity_per=10.0,
                unit_cost=0.01, lead_time_days=7, level=2, suppliers=["SUP-001", "SUP-002"],
                make_or_buy="buy"),
        BOMItem(part_id="C-002", name="Capacitor", parent_id="SA-001", quantity_per=5.0,
                unit_cost=0.05, lead_time_days=10, level=2, suppliers=["SUP-001"],
                make_or_buy="buy"),
        BOMItem(part_id="C-003", name="IC Chip", parent_id="SA-002", quantity_per=1.0,
                unit_cost=2.50, lead_time_days=14, level=2, suppliers=["SUP-003"],
                make_or_buy="buy", scrap_rate=0.02),
    ]
    return BOMTree(items)


class TestBOMTree:
    def test_add_and_get_items(self, sample_bom):
        assert len(sample_bom.items) == 6
        assert sample_bom.items["ASM-001"].name == "Widget Assembly"

    def test_root_items(self, sample_bom):
        roots = sample_bom.root_items
        assert len(roots) == 1
        assert roots[0].part_id == "ASM-001"

    def test_get_children(self, sample_bom):
        children = sample_bom.get_children("ASM-001")
        assert len(children) == 2
        child_ids = {c.part_id for c in children}
        assert child_ids == {"SA-001", "SA-002"}

    def test_explode(self, sample_bom):
        explosion = sample_bom.explode("ASM-001")
        assert len(explosion) >= 5  # 2 sub-assemblies + 3 components
        # Check extended quantities
        resistor = next(r for r in explosion if r.part_id == "C-001")
        assert resistor.extended_quantity == 10.0  # 1 * 10
        ic = next(r for r in explosion if r.part_id == "C-003")
        # IC has scrap_rate=0.02, qty_per=1.0, parent_qty=2.0
        assert ic.extended_quantity > 2.0  # adjusted for scrap

    def test_where_used(self, sample_bom):
        parents = sample_bom.where_used("C-001")
        parent_ids = [p.part_id for p in parents]
        assert "SA-001" in parent_ids
        assert "ASM-001" in parent_ids

    def test_cost_rollup(self, sample_bom):
        cost = sample_bom.cost_rollup("ASM-001")
        assert cost > 5.0  # must be more than just assembly cost
        # Should include sub-assembly + component costs
        assert cost > 10.0

    def test_critical_path(self, sample_bom):
        path = sample_bom.critical_path("ASM-001")
        # ASM(2) + SA-002(4) + C-003(14) = 20, or ASM(2) + SA-001(3) + C-002(10) = 15
        assert path >= 15

    def test_depth(self, sample_bom):
        assert sample_bom.depth("ASM-001") == 2  # 3 levels = depth 2
        assert sample_bom.depth("C-001") == 0  # leaf

    def test_remove_item(self, sample_bom):
        # Remove sub-assembly A and its children
        assert sample_bom.remove_item("SA-001")
        assert "SA-001" not in sample_bom.items
        assert "C-001" not in sample_bom.items  # children removed too
        assert "C-002" not in sample_bom.items

    def test_validate_valid_bom(self, sample_bom):
        errors = sample_bom.validate()
        assert len(errors) == 0

    def test_validate_missing_parent(self):
        tree = BOMTree([
            BOMItem(part_id="C-999", name="Orphan", parent_id="NONEXISTENT", quantity_per=1.0)
        ])
        errors = tree.validate()
        assert any("non-existent parent" in e for e in errors)


class TestBOMManager:
    def test_generate_synthetic_boms(self):
        mgr = BOMManager()
        trees = mgr.generate_synthetic_boms(n_assemblies=3)
        assert len(trees) == 3
        assert len(mgr.assemblies) == 3

    def test_synthetic_bom_structure(self):
        mgr = BOMManager()
        mgr.generate_synthetic_boms(n_assemblies=1)
        tree = list(mgr.assemblies.values())[0]
        roots = tree.root_items
        assert len(roots) == 1
        # Should have sub-assemblies and components
        assert len(tree.items) > 5

    def test_find_single_source_risks(self):
        mgr = BOMManager()
        mgr.generate_synthetic_boms(n_assemblies=3)
        risks = mgr.find_single_source_risks()
        # Some components should have single source
        assert isinstance(risks, list)

    def test_find_long_lead_items(self):
        mgr = BOMManager()
        mgr.generate_synthetic_boms(n_assemblies=3)
        items = mgr.find_long_lead_items(threshold_days=14)
        assert isinstance(items, list)

    def test_get_summary(self):
        mgr = BOMManager()
        mgr.generate_synthetic_boms(n_assemblies=3)
        summary = mgr.get_summary()
        assert summary["assembly_count"] == 3
        assert summary["total_items"] > 0
        assert summary["max_bom_depth"] >= 2

    def test_deterministic_generation(self):
        mgr1 = BOMManager()
        mgr1.generate_synthetic_boms(n_assemblies=2, seed=42)
        mgr2 = BOMManager()
        mgr2.generate_synthetic_boms(n_assemblies=2, seed=42)
        s1 = mgr1.get_summary()
        s2 = mgr2.get_summary()
        assert s1["total_rollup_cost"] == s2["total_rollup_cost"]
