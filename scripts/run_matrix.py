import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [20260915, 20260916, 20260917]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", choices=["ball", "lshape"], required=True)
    p.add_argument(
        "--run", action="store_true", help="Actually execute the printed commands"
    )
    p.add_argument("--no-compile", action="store_true")
    a = p.parse_args()
    jobs = []
    if a.suite == "ball":
        for domain in ["interval", "disk"]:
            for s in [0.3, 0.5, 0.7]:
                for seed in SEEDS:
                    jobs.append(
                        [
                            sys.executable,
                            str(ROOT / "examples" / "constant_ball.py"),
                            "--domain",
                            domain,
                            "--s",
                            str(s),
                            "--seed",
                            str(seed),
                        ]
                    )
    else:
        for s in [0.3, 0.5, 0.7]:
            for N in [128, 256, 512]:
                seeds = SEEDS + [20261006, 20261007] if N == 128 else SEEDS
                for seed in seeds:
                    jobs.append(
                        [
                            sys.executable,
                            str(ROOT / "examples" / "geometry_domains.py"),
                            "--domain",
                            "lshape",
                            "--s",
                            str(s),
                            "--N",
                            str(N),
                            "--seed",
                            str(seed),
                        ]
                    )
    for command in jobs:
        if a.no_compile:
            command += ["--no-compile"]
        print(" ".join(command), flush=True)
        if a.run:
            subprocess.run(command, check=True, cwd=ROOT)
    print(f"{len(jobs)} paired groups; " + ("executed" if a.run else "dry run only"))


if __name__ == "__main__":
    main()
