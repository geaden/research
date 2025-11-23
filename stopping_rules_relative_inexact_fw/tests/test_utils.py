from utils import significant_figures


def test_significant_numbers():
    assert significant_figures(0.000132456, 3) == 0.000132
