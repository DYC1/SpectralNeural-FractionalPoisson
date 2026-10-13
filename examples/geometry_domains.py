import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import torch
from core import geometry, boundary_distance, spectral, Net, test_grid

p = argparse.ArgumentParser()
p.add_argument("--domain", choices=["lshape", "square_hole", "annulus"], required=True)
p.add_argument("--s", type=float, required=True)
p.add_argument("--seed", type=int, default=20260915)
p.add_argument("--device", default="cuda:0")
p.add_argument("--epochs", type=int, default=20000)
p.add_argument("--out", default=str(ROOT / "results" / "geometry_domains"))
p.add_argument("--smoke", action="store_true")
p.add_argument("--N", type=int, default=128)
p.add_argument("--betas", nargs="+", default=["0.01", "s", "1", "2"])
p.add_argument("--reference-dir", default=str(ROOT / "references"))
p.add_argument("--reference-N", type=int, default=2048)
p.add_argument("--evaluation-grid", type=int, default=401)
p.add_argument("--no-compile", action="store_true")
a = p.parse_args()
torch.set_num_threads(2)
torch.set_default_dtype(torch.float64)
if not 0 < a.s < 1 or a.N < 2 or a.epochs < 1 or a.evaluation_grid < 3:
    p.error("Require 0<s<1, N>=2, epochs>=1, and evaluation-grid>=3")
if not a.device.startswith("cuda") or not torch.cuda.is_available():
    raise RuntimeError("Formal experiments require CUDA; CPU fallback is prohibited.")
d = 2
N = a.N
device = a.device
betas = [a.s if str(b) == "s" else float(b) for b in a.betas]
K = len(betas)
if any(not np.isfinite(b) or b <= 0 for b in betas):
    p.error("All boundary exponents must be positive")
ref_path = Path(a.reference_dir) / f"{a.domain}-s{a.s:g}-N{a.reference_N}.npz"
reference_data = None
if not a.smoke:
    if not ref_path.exists():
        raise FileNotFoundError("First run make_reference.py: " + str(ref_path))
    reference_data = np.load(ref_path)
    if str(reference_data["execution_platform"]) != "cuda":
        raise RuntimeError("CUDA reference required")
    if reference_data["solution"].shape != (a.evaluation_grid, a.evaluation_grid):
        raise ValueError("Reference and evaluation grids differ")
    if (
        float(reference_data["s"]) != a.s
        or str(reference_data["domain"]) != a.domain
        or int(reference_data["N"]) != a.reference_N
    ):
        raise ValueError("Reference configuration mismatch")
path = Path(a.out)
path.mkdir(parents=True, exist_ok=True)
stem = f"{a.domain}-s{a.s:g}-N{N}-seed{a.seed}"
target = path / (stem + ".json")
if target.exists() and not a.smoke:
    old = json.loads(target.read_text())
    if all(
        old.get(k) == v
        for k, v in dict(
            epochs=a.epochs,
            betas=betas,
            N=N,
            execution_platform="cuda",
            reference_N=a.reference_N,
            evaluation_grid=a.evaluation_grid,
        ).items()
    ):
        print("SKIP " + stem, flush=True)
        raise SystemExit(0)
    raise RuntimeError(
        "Output exists with another configuration; choose a different --out"
    )
xy, w, A, spec, scale, _, _ = spectral(N, d, a.s, device)
g = geometry(xy, a.domain)
inside = g > 0
factors = torch.stack([g**b for b in betas])
active_indices = inside.nonzero().flatten()
active_points = xy[inside]
active_factors = factors[:, inside]
reaction = False
rhs = torch.ones_like(g)
torch.manual_seed(a.seed)
np.random.seed(a.seed)
model = Net(d).to(device)
models = [Net(d).to(device) for _ in betas]
for m in models:
    m.load_state_dict(model.state_dict())
params, buffers = torch.func.stack_module_state(models)
template = Net(d).to("meta")


def one(par, buf, x):
    return torch.func.functional_call(template, (par, buf), (x,))


forward = torch.vmap(one, in_dims=(0, 0, None))


def energy_per_beta(u):
    U = u.reshape(K, N + 1, N + 1)
    coeff = A @ (U * w) @ A.T
    return scale * (coeff.square() * spec).sum((1, 2)) - (
        U * rhs.reshape(N + 1, N + 1) * w
    ).sum((1, 2))


def objective():
    values = forward(params, buffers, active_points)
    U = torch.zeros_like(factors).index_copy(1, active_indices, values * active_factors)
    return energy_per_beta(U)


if not a.no_compile:
    objective = torch.compile(objective)
opt = torch.optim.AdamW(list(params.values()), lr=1e-3, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=400, gamma=0.9)
start = time.time()
history = []
for epoch in range(a.epochs):
    opt.zero_grad()
    energies = objective()
    energies.sum().backward()
    opt.step()
    scheduler.step()
    if epoch % 1000 == 0 or epoch == a.epochs - 1:
        entry = {
            "epoch": epoch + 1,
            "energies": energies.detach().cpu().tolist(),
            "seconds": time.time() - start,
        }
        history.append(entry)
        print(json.dumps({"group": stem, **entry}), flush=True)
if a.smoke:
    print(
        json.dumps(
            {
                "smoke_seconds": time.time() - start,
                "epochs": a.epochs,
                "domain": a.domain,
            }
        ),
        flush=True,
    )
    raise SystemExit(0)
test = test_grid(d, device, a.evaluation_grid)
tg = geometry(test, a.domain)
ti = tg > 1e-14
with torch.no_grad():
    pred = forward(params, buffers, test) * torch.stack([tg**b for b in betas])
    truth = torch.tensor(reference_data["solution"].ravel(), device=device)
    reference = str(ref_path)
    selections = {
        "global": ti,
        "boundary": ti & (boundary_distance(test, a.domain) <= 0.1),
    }
    if a.domain == "lshape":
        selections["corner"] = ti & (test.square().sum(-1).sqrt() < 0.1)
    errors = {
        key: (
            torch.linalg.vector_norm((pred - truth)[:, sel], dim=1)
            / torch.linalg.vector_norm(truth[sel])
        )
        .cpu()
        .tolist()
        for key, sel in selections.items()
    }
    assert all(np.isfinite(v).all() for v in errors.values()), errors
result = dict(
    version=2,
    problem="pure_fractional_poisson_f1",
    execution_platform="cuda",
    device=device,
    gpu_name=torch.cuda.get_device_name(torch.device(device)),
    visible_gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
    domain=a.domain,
    dimension=d,
    s=a.s,
    betas=betas,
    seed=a.seed,
    N=N,
    epochs=a.epochs,
    initialization="Xavier-normal weights; default Linear biases",
    optimizer="AdamW",
    lr=1e-3,
    weight_decay=0.01,
    scheduler="50 decays, gamma=.9",
    dtype="float64",
    h=1 / 3,
    sinc_bounds=[-100, 100],
    reaction=reaction,
    reference=reference,
    reference_N=a.reference_N,
    evaluation_grid=a.evaluation_grid,
    depth=2,
    width=100,
    errors=errors,
    seconds=time.time() - start,
    history=history,
)
temporary = target.with_suffix(".json.tmp")
temporary.write_text(json.dumps(result, indent=2))
temporary.replace(target)
torch.save(
    {
        "params": {k: v.detach().cpu() for k, v in params.items()},
        "buffers": buffers,
        "configuration": result,
    },
    path / (stem + ".pth"),
)
print("COMPLETE " + stem, flush=True)
