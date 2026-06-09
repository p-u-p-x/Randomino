"""
Bootstrap — 95% CI for the median of an Exponential sample
Exact translation of bootstrap.R
"""
import numpy as np
import plotly.graph_objects as go
from scipy import stats


def run(n: int = 50, B: int = 10000, rate: float = 0.5, seed: int = 123) -> dict:
    rng = np.random.default_rng(seed)

    # data <- rexp(n, rate=rate)
    data = rng.exponential(scale=1 / rate, size=n)
    sample_median = np.median(data)

    # boot_medians <- replicate(B, median(sample(data, replace=TRUE)))
    boot_medians = np.array([
        np.median(rng.choice(data, size=n, replace=True))
        for _ in range(B)
    ])

    ci = np.quantile(boot_medians, [0.025, 0.975])
    theoretical_median = np.log(2) / rate   # true median of Exp(rate)

    return dict(
        sample_median=sample_median,
        ci_lo=ci[0],
        ci_hi=ci[1],
        ci_width=ci[1] - ci[0],
        theoretical_median=theoretical_median,
        boot_medians=boot_medians,
    )


def plot(res: dict) -> go.Figure:
    bm = res["boot_medians"]
    counts, bin_edges = np.histogram(bm, bins=50, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=bin_centers, y=counts,
        marker_color="rgba(127,119,221,0.45)",
        marker_line_color="#7F77DD", marker_line_width=0.5,
        name="Bootstrap dist",
    ))

    max_y = counts.max() * 1.15
    for val, label, color in [
        (res["ci_lo"], "CI 2.5%", "#E24B4A"),
        (res["ci_hi"], "CI 97.5%", "#E24B4A"),
        (res["sample_median"], "Sample median", "#5DCAA5"),
    ]:
        fig.add_trace(go.Scatter(
            x=[val, val], y=[0, max_y],
            mode="lines",
            line=dict(color=color, width=2, dash="dash"),
            name=label,
        ))

    fig.update_layout(
        title="Bootstrap distribution of median",
        xaxis_title="Median", yaxis_title="Density",
        height=400, paper_bgcolor="#141720", plot_bgcolor="#0d0f12",
        font=dict(color="#8b91a8", size=11), margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        bargap=0,
    )
    fig.update_xaxes(gridcolor="#2a2f3e")
    fig.update_yaxes(gridcolor="#2a2f3e")
    return fig
