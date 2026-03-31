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


def do_show_plot(filename: str, show_plot: bool, interactive: bool, should_draw: bool = False) -> None:
    if not show_plot:
        return

    if should_draw:
        plt.draw()
    else:
        plt.show()

    print(f"Check if dump to {filename} is needed: {not interactive}")
    if not interactive:
        print(f"Dumping to {filename}: {not interactive}...")
        plt.savefig("images/" + filename)
