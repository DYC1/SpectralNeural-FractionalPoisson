import argparse
from pathlib import Path
import sys
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from core import Net, geometry, exact, test_grid


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--reference", type=Path, help="Reference NPZ for the selected domain")
    p.add_argument("--out", type=Path, default=ROOT / "figures")
    a = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for model evaluation")
    torch.set_default_dtype(torch.float64)
    saved = torch.load(a.checkpoint, map_location="cuda:0", weights_only=True)
    c = saved["configuration"]
    d = c["dimension"]
    count = c["evaluation_grid"]
    points = test_grid(d, "cuda:0", count)
    g = geometry(points, c["domain"])
    if c["reference"] == "analytical":
        truth = (
            exact(points, c["s"]) if c["problem"] == "pure_fractional_poisson_f1" else g
        )
    else:
        ref = a.reference if a.reference else Path(c["reference"])
        if not ref.exists():
            raise FileNotFoundError("Specify the reference NPZ with --reference")
        data = np.load(ref)
        if data["solution"].shape != (count, count):
            raise ValueError("Reference grid mismatch")
        if str(data["domain"]) != c["domain"] or float(data["s"]) != c["s"]:
            raise ValueError("Reference domain or fractional order mismatch")
        truth = torch.as_tensor(data["solution"].ravel(), device="cuda:0")
    a.out.mkdir(parents=True, exist_ok=True)
    for i, beta in enumerate(c["betas"]):
        model = Net(d).to("cuda:0")
        model.load_state_dict({k: v[i] for k, v in saved["params"].items()})
        model.eval()
        with torch.no_grad():
            pred = model(points) * g**beta
        x = points.detach().cpu().numpy()
        t = truth.detach().cpu().numpy()
        u = pred.detach().cpu().numpy()
        fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
        if d == 1:
            axes[0].plot(x[:, 0], t)
            axes[1].plot(x[:, 0], u)
            axes[2].plot(x[:, 0], np.abs(u - t))
        else:
            for ax, field in zip(axes[:3], [t, u, np.abs(u - t)]):
                im = ax.imshow(
                    field.reshape(count, count).T, origin="lower", extent=(-1, 1, -1, 1)
                )
                fig.colorbar(im, ax=ax)
        for ax, title in zip(
            axes[:3], ["Exact / reference", "Numerical solution", "Pointwise error"]
        ):
            ax.set_title(title)
        axes[3].plot(
            [v["epoch"] for v in c["history"]], [v["energies"][i] for v in c["history"]]
        )
        axes[3].set_title("Ritz energy")
        axes[3].set_xlabel("Training step")
        fig.suptitle(f"{c['domain']}, s={c['s']}, beta={beta}, N={c['N']}")
        fig.tight_layout()
        target = a.out / f"{a.checkpoint.stem}-beta{beta:g}.png"
        fig.savefig(target, dpi=180)
        plt.close(fig)
        print(target)


if __name__ == "__main__":
    main()
