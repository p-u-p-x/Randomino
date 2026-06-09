"""
Inverse CDF sampling — Triangular distribution on [0, 1]
Exact translation of inverse_cdf.R
"""
import numpy as np
import plotly.graph_objects as go


def rtriangular(n: int, a: float = 0.0, b: float = 1.0,
                c: float = 0.3, seed: int = 99) -> np.ndarray:
    """Exact Python translation of R's rtriangular function"""
    rng = np.random.default_rng(seed)
    u = rng.uniform(0, 1, n)
    fc = (c - a) / (b - a)
    samples = np.where(
        u < fc,
        a + np.sqrt(u * (b - a) * (c - a)),
        b - np.sqrt((1 - u) * (b - a) * (b - c)),
    )
    return samples


def dtriangular(x: np.ndarray, a: float = 0.0,
                b: float = 1.0, c: float = 0.3) -> np.ndarray:
    """Triangular PDF — exact translation of R's dtriangular"""
    return np.where(
        x < c,
        2 * (x - a) / ((b - a) * (c - a)),
        2 * (b - x) / ((b - a) * (b - c)),
    )


def run(n: int = 100000, c: float = 0.3, seed: int = 99) -> dict:
    a, b = 0.0, 1.0
    samples = rtriangular(n, a, b, c, seed)
    theoretical_mean = (a + b + c) / 3
    return dict(
        samples=samples,
        sample_mean=samples.mean(),
        theoretical_mean=theoretical_mean,
        mode=c,
        a=a, b=b, c=c,
    )


def plot(res: dict) -> go.Figure:
    samples = res["samples"]
    counts, edges = np.histogram(samples, bins=80, density=True)
    centers = (edges[:-1] + edges[1:]) / 2

    x_seq = np.linspace(0, 1, 500)
    theory = dtriangular(x_seq, res["a"], res["b"], res["c"])

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=centers, y=counts,
        marker_color="rgba(55,138,221,0.35)",
        marker_line_color="#378ADD", marker_line_width=0.4,
        name="Samples",
    ))  # <-- REMOVED invalid 'bargap=0' argument

    fig.add_trace(go.Scatter(
        x=x_seq, y=theory,
        mode="lines", line=dict(color="#E24B4A", width=2.5),
        name="Theoretical PDF",
    ))

    fig.update_layout(
        title="Triangular distribution via inverse CDF",
        xaxis_title="x", yaxis_title="Density",
        height=420, paper_bgcolor="#141720", plot_bgcolor="#0d0f12",
        font=dict(color="#8b91a8", size=11), margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)"), bargap=0,
    )
    fig.update_xaxes(gridcolor="#2a2f3e")
    fig.update_yaxes(gridcolor="#2a2f3e")
    return fig