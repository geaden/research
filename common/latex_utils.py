"""Module contains utility functions for LaTeX representation."""


def latex_table(
    caption: str, values: list[tuple[object, object]], headers: list[str]
) -> str:
    """Prints LaTeX table."""

    columns = " ".join(["c" for _ in headers])
    heads = " & ".join(headers)
    table = [r"\begin{table}[!h]"]
    table += [rf"    \caption{{{caption}}}"]
    table += [r"    \begin{center}"]
    table += [rf"        \begin{{tabular}}{{||{columns}||}}"]
    table += [r"            \hline"]
    table += [rf"            {heads} \\ [0.5ex]"]
    table += [r"            \hline\hline"]
    for items in values:
        c = " & ".join([str(item) for item in items])
        table += [rf"            {c} \\"]
        table += [r"            \hline"]
    table += [r"        \end{tabular}"]
    table += [r"    \end{center}"]
    table += [r"\end{table}"]
    return "\n".join(table)
