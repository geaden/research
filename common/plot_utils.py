import matplotlib
import matplotlib.pyplot as plt

SOLID_LINE: tuple[str] = ("-", "b")


def preamble():
    matplotlib.use("pgf")
    params = {
        "text.latex.preamble": (
            r"\usepackage{amsmath}"
            r"\usepackage[T1, T2A]{fontenc}"
            r"\usepackage[utf8]{inputenc}"
            r"\usepackage[main=russian,english]{babel}"
            r"\usepackage{tikz}"
        ),
        "text.usetex": True,
        "font.size": 11,
        "font.family": "lmodern",
    }
    plt.rcParams.update(params)


def do_show_plot(filename: str, show_plot: bool, interactive: bool) -> None:
    if not show_plot:
        return

    plt.show()
    print(f"Dumping to {filename}: {not interactive}...")
    if not interactive:
        plt.savefig("images/" + filename)
