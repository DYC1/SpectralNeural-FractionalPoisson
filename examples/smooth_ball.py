from pathlib import Path
import runpy
import sys

runner = Path(__file__).with_name("constant_ball.py")
sys.argv = [str(runner), "--problem", "smooth", "--betas", "s", *sys.argv[1:]]
runpy.run_path(str(runner), run_name="__main__")
