"""Tests for CTB (Clear-to-Build) analyzer."""
import pytest

from chaincommand.bom.models import BOMItem, BOMTree
from chaincommand.ctb.analyzer import CTBAnalyzer


@pytest.fixture
def sample_bom():
    items = [
        BOMItem(part_id="ASM-001", name="Assembly", parent_id=None, quantity_per=1.0,
                unit_cost=5.0, lead_time_days=2, level=0, make_or_buy="make"),
        BOMItem(part_id="SA-001", name="Sub-A", parent_id="ASM-001", quantity_per=1.0,
                unit_cost=2.0, lead_time_days=3, level=1, make_or_buy="make"),
        BOMItem(part_id="C-001", name="Resistor", parent_id="SA-001", quantity_per=10.0,
                unit_cost=0.01, lead_time_days=5, level=2, suppliers=["S1", "S2"],
                make_or_buy="buy"),
        BOMItem(part_id="C-002", name="Capacitor", parent_id="SA-001", quantity_per=5.0,
                unit_cost=0.05, lead_time_days=7, level=2, suppliers=["S1"],
                make_or_buy="buy"),
    ]
    return BOMTree(items)


class TestCTBAnalyzer:
    def test_all_parts_available(self, sample_bom):
        analyzer = CTBAnalyzer()
        inventory = {"C-001": 100.0, "C-002": 50.0}
        report = analyzer.analyze(sample_bom, "ASM-001", 1.0, inventory)
        assert report.is_clear
        assert report.clear_percentage == 100.0
        assert len(report.shortages) == 0

    def test_shortage_detected(self, sample_bom):
        analyzer = CTBAnalyzer()
        inventory = {"C-001": 5.0, "C-002": 50.0}  # only 5 resistors, need 10
        report = analyzer.analyze(sample_bom, "ASM-001", 1.0, inventory)
        assert not report.is_clear
        assert report.clear_percentage < 100.0
        assert len(report.shortages) >= 1
        shortage = next(s for s in report.shortages if s.part_id == "C-001")
        assert shortage.shortage_qty > 0

    def test_no_inventory(self, sample_bom):
        analyzer = CTBAnalyzer()
        report = analyzer.analyze(sample_bom, "ASM-001", 1.0, {})
        assert not report.is_clear
        assert report.clear_percentage == 0.0

    def test_build_multiple_units(self, sample_bom):
        analyzer = CTBAnalyzer()
        inventory = {"C-001": 100.0, "C-002": 50.0}
        report = analyzer.analyze(sample_bom, "ASM-001", 10.0, inventory)
        # Need 100 resistors, have 100 -> OK
        # Need 50 capacitors, have 50 -> OK
        assert report.is_clear
        assert report.build_quantity == 10.0

    def test_on_order_counted(self, sample_bom):
        analyzer = CTBAnalyzer()
        inventory = {"C-001": 5.0, "C-002": 50.0}
        on_order = {"C-001": 10.0}  # 5 + 10 = 15 >= 10 needed
        report = analyzer.analyze(sample_bom, "ASM-001", 1.0, inventory, on_order)
        assert report.is_clear

    def test_longest_wait(self, sample_bom):
        analyzer = CTBAnalyzer()
        report = analyzer.analyze(sample_bom, "ASM-001", 1.0, {})
        assert report.longest_wait_days > 0

    def test_total_material_cost(self, sample_bom):
        analyzer = CTBAnalyzer()
        report = analyzer.analyze(sample_bom, "ASM-001", 1.0, {})
        assert report.total_material_cost > 0

    def test_empty_bom(self):
        analyzer = CTBAnalyzer()
        tree = BOMTree([BOMItem(part_id="EMPTY", name="Empty", parent_id=None)])
        report = analyzer.analyze(tree, "EMPTY", 1.0, {})
        assert report.is_clear
