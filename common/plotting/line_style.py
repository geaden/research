class LineStyle:

    _STYLES = ["-", "--", "-."]
    _counter = 0

    def next(self) -> dict[str, str]:
        style = self._STYLES[self._counter % len(self._STYLES)]
        self._counter += 1
        return {"linestyle": style, "lw": 2}

    def reset(self) -> None:
        self._counter = 0
