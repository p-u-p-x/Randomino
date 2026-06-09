"""
Monte Carlo simulation — estimates π by throwing random points
Exact translation of monte_carlo.R
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def run(n: int, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, n)
    y = rng.uniform(-1, 1, n)
    inside = (x ** 2 + y ** 2) <= 1

    pi_est = 4 * np.mean(inside)
    error = abs(pi_est - np.pi)

    # Convergence series — 100 checkpoints
    checkpoints = np.linspace(100, n, min(200, n), dtype=int)
    cumpi = [4 * inside[:k].mean() for k in checkpoints]

    # Scatter — cap at 3000 points so browser stays fast
    cap = min(n, 3000)
    scatter_inside = np.column_stack([x[:cap][inside[:cap]], y[:cap][inside[:cap]]])
    scatter_outside = np.column_stack([x[:cap][~inside[:cap]], y[:cap][~inside[:cap]]])

    return dict(
        pi_est=pi_est,
        error=error,
        pct_inside=np.mean(inside) * 100,
        checkpoints=checkpoints.tolist(),
        cumpi=cumpi,
        scatter_inside=scatter_inside,
        scatter_outside=scatter_outside,
    )


def plot(res: dict) -> go.Figure:
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Random point scatter", "π convergence"),
    )

    ins = res["scatter_inside"]
    out = res["scatter_outside"]

    fig.add_trace(go.Scatter(
        x=ins[:, 0], y=ins[:, 1], mode="markers",
        marker=dict(size=2, color="rgba(93,202,165,0.4)"),
        name="Inside", showlegend=True,
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=out[:, 0], y=out[:, 1], mode="markers",
        marker=dict(size=2, color="rgba(216,90,48,0.4)"),
        name="Outside", showlegend=True,
    ), row=1, col=1)

    # Quarter-circle boundary
    theta = np.linspace(0, np.pi / 2, 200)
    fig.add_trace(go.Scatter(
        x=np.cos(theta), y=np.sin(theta),
        mode="lines", line=dict(color="white", width=1, dash="dot"),
        name="Circle", showlegend=False,
    ), row=1, col=1)

    # Convergence line
    fig.add_trace(go.Scatter(
        x=res["checkpoints"], y=res["cumpi"],
        mode="lines", line=dict(color="#5DCAA5", width=1.5),
        name="π estimate",
    ), row=1, col=2)

    fig.add_hline(
        y=np.pi, line_dash="dash", line_color="#E24B4A",
        annotation_text="True π", row=1, col=2,
    )

    fig.update_layout(
        height=380, paper_bgcolor="#141720", plot_bgcolor="#0d0f12",
        font=dict(color="#8b91a8", size=11), margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor="#2a2f3e", zerolinecolor="#2a2f3e")
    fig.update_yaxes(gridcolor="#2a2f3e", zerolinecolor="#2a2f3e")
    return fig
