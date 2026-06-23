from pathlib import Path


def set_up() -> Path:
    output_dir = Path.cwd() / Path("output")
    if not output_dir.exists():
        output_dir.mkdir()
    return output_dir
