"""
Rejection sampling — Beta(alpha, beta) via uniform proposal
Exact translation of rejection_sampling.R
Target: f(x) ∝ x^(α-1)(1-x)^(β-1),  Proposal: Uniform[0,1]
"""
import numpy as np
import plotly.graph_objects as go
from scipy import stats


def run(alpha: float = 3.0, beta: float = 4.0,
        n_target: int = 10000, seed: int = 5) -> dict:
    rng = np.random.default_rng(seed)

    # Unnormalised target (same as R: f_target <- function(x) x^2 * (1-x)^3)
    def f_target(x):
        return (x ** (alpha - 1)) * ((1 - x) ** (beta - 1))

    # M = max of f_target over [0,1] at the mode
    mode = (alpha - 1) / (alpha + beta - 2) if (alpha > 1 and beta > 1) else 0.5
    mode = np.clip(mode, 0.001, 0.999)
    M = f_target(mode) * 1.05   # slight overestimate for safety

    accepted = []
    n_proposed = 0

    while len(accepted) < n_target:
        batch = 1000
        x_prop = rng.uniform(0, 1, batch)
        u = rng.uniform(0, 1, batch)
        keep = u <= f_target(x_prop) / M
        accepted.extend(x_prop[keep].tolist())
        n_proposed += batch

    accepted = np.array(accepted[:n_target])
    accept_rate = n_target / n_proposed
    theoretical_mean = alpha / (alpha + beta)

    return dict(
        accepted=accepted,
        accept_rate=accept_rate * 100,
        n_proposed=n_proposed,
        sample_mean=accepted.mean(),
        theoretical_mean=theoretical_mean,
        alpha=alpha, beta_param=beta,
    )


def plot(res: dict) -> go.Figure:
    acc = res["accepted"]
    counts, edges = np.histogram(acc, bins=60, density=True)
    centers = (edges[:-1] + edges[1:]) / 2

    x_seq = np.linspace(0.001, 0.999, 400)
    theory = stats.beta.pdf(x_seq, res["alpha"], res["beta_param"])

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=centers, y=counts,
        marker_color="rgba(212,83,126,0.4)",
        marker_line_color="#D4537E", marker_line_width=0.4,
        name=f"Accepted samples (n={len(acc):,})",
    ))
    fig.add_trace(go.Scatter(
        x=x_seq, y=theory,
        mode="lines", line=dict(color="#E24B4A", width=2.5),
        name=f"Beta({res['alpha']}, {res['beta_param']}) PDF",
    ))

    fig.update_layout(
        title=f"Rejection sampling → Beta({res['alpha']}, {res['beta_param']})",
        xaxis_title="x", yaxis_title="Density",
        height=420, paper_bgcolor="#141720", plot_bgcolor="#0d0f12",
        font=dict(color="#8b91a8", size=11), margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)"), bargap=0,
    )
    fig.update_xaxes(gridcolor="#2a2f3e")
    fig.update_yaxes(gridcolor="#2a2f3e")
    return fig
