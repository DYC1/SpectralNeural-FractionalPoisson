import argparse
import json
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import torch
from core import geometry, spectral, interpolate_matrix, test_grid

p = argparse.ArgumentParser()
p.add_argument("--device", default="cuda:0")
p.add_argument(
    "--domain",
    nargs="+",
    choices=["lshape", "square_hole", "annulus"],
    default=["lshape"],
)
p.add_argument("--s", type=float, nargs="+", default=[0.3, 0.5, 0.7])
p.add_argument("--N", type=int, nargs="+", default=[2048])
p.add_argument("--evaluation-grid", type=int, default=401)
p.add_argument("--out", default=str(ROOT / "references"))
a = p.parse_args()
if any(N < 2 for N in a.N) or any(not 0 < s < 1 for s in a.s) or a.evaluation_grid < 3:
    p.error("Require N>=2, 0<s<1, and evaluation-grid>=3")
torch.set_num_threads(4)
torch.set_default_dtype(torch.float64)
if not a.device.startswith("cuda") or not torch.cuda.is_available():
    raise RuntimeError(
        "Reference computation requires CUDA; CPU fallback is prohibited."
    )
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
for N in a.N:
    for s in a.s:
        points, w, A, spec, scale, y, x = spectral(N, 2, s, a.device)
        rootw = w.sqrt()
        grid = np.linspace(-1, 1, a.evaluation_grid)
        B = torch.tensor(interpolate_matrix(y.cpu().numpy(), grid), device=a.device)
        test = test_grid(2, a.device, a.evaluation_grid)
        for domain in a.domain:
            target = out / f"{domain}-s{s:g}-N{N}.npz"
            if target.exists():
                previous = np.load(target)
                if (
                    "execution_platform" in previous
                    and str(previous["execution_platform"]) == "cuda"
                    and previous["solution"].shape
                    == (a.evaluation_grid, a.evaluation_grid)
                ):
                    print("SKIP " + str(target), flush=True)
                    continue
                raise RuntimeError(
                    "Refuse to reuse or overwrite an existing non-CUDA reference."
                )
            start = time.time()
            inside = (geometry(points, domain) > 0).reshape(N + 1, N + 1)

            def H(z):
                coeff = A @ (rootw * z * inside) @ A.T
                return 2 * scale * rootw * (A.T @ (spec * coeff) @ A) * inside

            b = rootw * inside
            z = torch.zeros_like(b)
            r = b.clone()
            direction = r.clone()
            rr = (r * r).sum()
            norm = float(rr.sqrt())
            for iteration in range(3000):
                hd = H(direction)
                alpha = rr / (direction * hd).sum()
                z += alpha * direction
                r -= alpha * hd
                new = (r * r).sum()
                if float(new.sqrt()) / norm < 1e-10:
                    break
                direction = r + (new / rr) * direction
                rr = new
            residual = float(torch.linalg.vector_norm(H(z) - b)) / norm
            assert residual < 2e-10, (domain, s, N, residual)
            u = z / rootw
            field = B @ u @ B.T
            field *= (geometry(test, domain) > 1e-14).reshape(
                a.evaluation_grid, a.evaluation_grid
            )
            meta = dict(
                execution_platform="cuda",
                device=a.device,
                gpu_name=torch.cuda.get_device_name(torch.device(a.device)),
                domain=domain,
                s=s,
                N=N,
                evaluation_grid=a.evaluation_grid,
                relative_residual=residual,
                iterations=iteration + 1,
                seconds=time.time() - start,
                reference="unrestricted nodal Ritz",
            )
            np.savez_compressed(target, solution=field.cpu().numpy(), **meta)
            target.with_suffix(".json").write_text(json.dumps(meta, indent=2))
            print(json.dumps(meta), flush=True)
