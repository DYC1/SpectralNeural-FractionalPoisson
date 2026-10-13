import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import torch
from core import Net, geometry, exact, boundary_distance, spectral

p = argparse.ArgumentParser()
p.add_argument("--domain", choices=["interval", "disk"], required=True)
p.add_argument("--s", type=float, required=True)
p.add_argument("--seed", type=int, default=20260915)
p.add_argument("--epochs", type=int, default=20000)
p.add_argument("--out")
p.add_argument("--preflight", action="store_true")
p.add_argument("--N", type=int, default=128)
p.add_argument("--betas", nargs="+", default=["0.01", "s", "1", "2"])
p.add_argument("--no-compile", action="store_true")
p.add_argument("--evaluation-grid", type=int, default=401)
p.add_argument("--problem", choices=["poisson", "smooth"], default="poisson")
a = p.parse_args()
torch.set_default_dtype(torch.float64)
torch.set_num_threads(2)
if not 0 < a.s < 1 or a.N < 2 or a.epochs < 1 or a.evaluation_grid < 3:
    p.error("Require 0<s<1, N>=2, epochs>=1, and evaluation-grid>=3")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA required; no CPU fallback")
device = "cuda:0"
N = a.N
d = 1 if a.domain == "interval" else 2
betas = [a.s if str(b) == "s" else float(b) for b in a.betas]
if any(not np.isfinite(b) or b <= 0 for b in betas):
    p.error("All boundary exponents must be positive")
problem = (
    "pure_fractional_poisson_f1"
    if a.problem == "poisson"
    else "smooth_fractional_poisson"
)
out = (
    Path(a.out)
    if a.out
    else ROOT / "results" / ("constant_ball" if a.problem == "poisson" else "smooth_ball")
)
out.mkdir(parents=True, exist_ok=True)
stem = f"{a.domain}-s{a.s:g}-N{N}-seed{a.seed}"
target = out / (stem + ".json")
if target.exists() and not a.preflight:
    old = json.loads(target.read_text())
    if all(
        old.get(k) == v
        for k, v in dict(
            problem=problem,
            epochs=a.epochs,
            betas=betas,
            N=N,
            evaluation_grid=a.evaluation_grid,
        ).items()
    ):
        print("SKIP " + stem, flush=True)
        sys.exit()
    raise RuntimeError(
        "Output exists with a different configuration; choose another --out directory"
    )
xy, w, A, spec, scale, _, _ = spectral(N, d, a.s, device)
g = geometry(xy, a.domain)
indices = (g > 0).nonzero().flatten()
xinside = xy[g > 0]
K = len(betas)
factors = torch.stack([g**b for b in betas])
active_factors = factors[:, g > 0]
rhs = torch.ones_like(g)
if a.problem == "smooth":
    from scipy.special import gamma, hyp2f1

    r2 = xinside.square().sum(-1).cpu().numpy()
    load = (
        2 ** (2 * a.s)
        * gamma(d / 2 + a.s)
        / (gamma(d / 2) * gamma(2 - a.s))
        * hyp2f1(d / 2 + a.s, a.s - 1, d / 2, r2)
    )
    rhs = torch.zeros_like(g)
    rhs[g > 0] = torch.as_tensor(load, device=device)
    if not torch.isfinite(rhs).all():
        raise RuntimeError("Non-finite smooth-problem load")
torch.manual_seed(a.seed)
np.random.seed(a.seed)
initial = Net(d).to(device)
models = [Net(d).to(device) for _ in betas]
for model in models:
    model.load_state_dict(initial.state_dict())
params, buffers = torch.func.stack_module_state(models)
template = Net(d).to("meta")


def one(par, buf, x):
    return torch.func.functional_call(template, (par, buf), (x,))


forward = torch.vmap(one, in_dims=(0, 0, None))


def energy(values):
    if d == 1:
        coeff = (values * w) @ A.T
        return scale * (coeff.square() * spec).sum(-1) - (values * rhs * w).sum(-1)
    U = values.reshape(K, N + 1, N + 1)
    coeff = A @ (U * w) @ A.T
    return scale * (coeff.square() * spec).sum((1, 2)) - (
        U * rhs.reshape(N + 1, N + 1) * w
    ).sum((1, 2))


def objective():
    values = forward(params, buffers, xinside) * active_factors
    U = torch.zeros_like(factors).index_copy(1, indices, values)
    return energy(U)


if a.preflight:
    prediction = forward(params, buffers, xinside)
    direct = torch.stack([m(xinside) for m in models])
    assert torch.allclose(prediction, direct, atol=1e-12, rtol=1e-12)
    objective().sum().backward()
    for i, m in enumerate(models):
        values = m(xinside) * active_factors[i]
        U = torch.zeros_like(factors).index_copy(1, indices, torch.stack([values] * K))
        energy(U)[i].backward()
        for name, par in m.named_parameters():
            assert torch.allclose(
                params[name].grad[i], par.grad, atol=1e-10, rtol=1e-10
            ), name
    assert (factors[:, g == 0] == 0).all()
    trial = torch.stack([exact(xy, a.s)] * K)
    zero = torch.zeros_like(trial)
    assert torch.allclose(energy(trial) - energy(zero), energy(trial), atol=1e-12)
    load = (trial * rhs * w.reshape(-1)).sum(-1)
    assert torch.allclose(
        energy(2 * trial) - 4 * energy(trial), 2 * load, atol=1e-10, rtol=1e-10
    )
    print(
        json.dumps(
            dict(
                preflight="passed",
                dimension=d,
                betas=betas,
                execution_platform="cuda",
                problem=problem,
                checks=[
                    "paired predictions",
                    "independent gradients",
                    "exterior zeros",
                    "quadratic/load homogeneity",
                ],
            )
        ),
        flush=True,
    )
    sys.exit()

if not a.no_compile:
    objective = torch.compile(objective)
opt = torch.optim.AdamW(list(params.values()), lr=1e-3, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=400, gamma=0.9)
start = time.time()
history = []
for epoch in range(a.epochs):
    opt.zero_grad()
    E = objective()
    E.sum().backward()
    opt.step()
    scheduler.step()
    if epoch % 1000 == 0 or epoch == a.epochs - 1:
        entry = dict(
            epoch=epoch + 1,
            energies=E.detach().cpu().tolist(),
            seconds=time.time() - start,
        )
        history.append(entry)
        print(json.dumps(dict(group=stem, **entry)), flush=True)


def evaluate(count):
    grid = torch.linspace(-1, 1, count, device=device)
    if d == 1:
        points = grid[:, None]
    else:
        X, Y = torch.meshgrid(grid, grid, indexing="ij")
        points = torch.stack([X.flatten(), Y.flatten()], 1)
    tg = geometry(points, a.domain)
    inside = tg > 1e-14
    pred = forward(params, buffers, points) * torch.stack([tg**b for b in betas])
    truth = exact(points, a.s) if a.problem == "poisson" else tg
    masks = {
        "global": inside,
        "boundary": inside & (boundary_distance(points, a.domain) <= 0.1),
    }
    return {
        name: (
            torch.linalg.vector_norm((pred - truth)[:, mask], dim=1)
            / torch.linalg.vector_norm(truth[mask])
        )
        .cpu()
        .tolist()
        for name, mask in masks.items()
    }


with torch.no_grad():
    errors = {f"grid{n}": evaluate(n) for n in sorted(set([100, a.evaluation_grid]))}
assert all(np.isfinite(v).all() for grid in errors.values() for v in grid.values())
result = dict(
    version=1,
    problem=problem,
    reaction=False,
    rhs="1" if a.problem == "poisson" else "hypergeometric load",
    reference="analytical",
    execution_platform="cuda",
    device=device,
    gpu_name=torch.cuda.get_device_name(0),
    visible_gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
    domain=a.domain,
    dimension=d,
    s=a.s,
    betas=betas,
    seed=a.seed,
    N=N,
    epochs=a.epochs,
    optimizer="AdamW",
    lr=1e-3,
    weight_decay=0.01,
    scheduler="StepLR every400 gamma=.9",
    initialization="Xavier-normal weights; default Linear biases",
    depth=2,
    width=100,
    dtype="float64",
    h=1 / 3,
    sinc_bounds=[-100, 100],
    boundary_layer_width=0.1,
    evaluation_grid=a.evaluation_grid,
    error_denominator="reference norm on the same selected region",
    errors=errors,
    history=history,
    seconds=time.time() - start,
)
tmp = target.with_suffix(".json.tmp")
tmp.write_text(json.dumps(result, indent=2))
tmp.replace(target)
torch.save(
    dict(
        params={k: v.detach().cpu() for k, v in params.items()},
        buffers=buffers,
        configuration=result,
    ),
    out / (stem + ".pth"),
)
print("COMPLETE " + stem, flush=True)
