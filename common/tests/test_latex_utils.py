from common.latex_utils import latex_table


def test_latex_table():
    table = latex_table(
        caption="foo", values=[("a", 0.2), ("b", 0.3)], headers=["A", "B"]
    )

    print(table)
    assert "||c c||" in table
    assert "a & 0.2 \\" in table
    assert "\end{table}" in table
