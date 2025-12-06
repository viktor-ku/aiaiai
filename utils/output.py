from pathlib import Path


def output(name: str, section: str | None = None) -> str:
    base = Path("output")

    if section:
        base /= section

    base.mkdir(parents=True, exist_ok=True)

    return str(base / name)


def png(filename: str) -> str:
    return f"{filename}.png"
