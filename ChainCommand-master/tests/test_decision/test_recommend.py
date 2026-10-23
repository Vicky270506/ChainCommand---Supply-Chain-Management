from chaincommand.decision.recommend import constrain


def test_constraints():
    assert constrain(-5, 10, 5, 100) == (0, "no_order")
    assert constrain(3, 10, 5, 100) == (10, "order")                  # minimum order quantity
    assert constrain(12, 1, 5, 100) == (15, "order")                 # pack-size rounding
    assert constrain(50, 10, 5, 22) == (20, "order_capped_storage")  # storage limit
    assert constrain(50, 30, 5, 22) == (0, "blocked_storage")