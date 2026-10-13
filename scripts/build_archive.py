import argparse
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EXCLUDE = {
    "results",
    "references",
    "figures",
    "generated_tables",
    "dist",
    "__pycache__",
    ".pytest_cache",
    ".git",
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=ROOT / "dist" / "code_submission.zip")
    a = p.parse_args()
    files = [
        f
        for f in ROOT.rglob("*")
        if f.is_file()
        and not (set(f.relative_to(ROOT).parts) & EXCLUDE)
        and f.suffix not in {".pyc", ".pyo", ".pth", ".npz", ".log"}
        and f.name != ".DS_Store"
    ]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if a.out.resolve() in [f.resolve() for f in files]:
        raise ValueError(
            "Choose an archive destination outside the included source files"
        )
    with zipfile.ZipFile(a.out, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for f in sorted(files):
            info = zipfile.ZipInfo(
                "code_submission/" + f.relative_to(ROOT).as_posix(),
                date_time=(1980, 1, 1, 0, 0, 0),
            )
            info.external_attr = 0o100644 << 16
            z.writestr(info, f.read_bytes(), compress_type=zipfile.ZIP_DEFLATED)
    print(f"{a.out} ({len(files)} files)")


if __name__ == "__main__":
    main()
