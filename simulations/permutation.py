"""
Permutation test — two-sample difference in means
Exact translation of permutation.R
"""
import numpy as np
import plotly.graph_objects as go


DEFAULT_A = [12.1, 11.8, 13.4, 10.9, 12.7, 11.5]
DEFAULT_B = [14.2, 13.6, 15.1, 14.8, 13.9]


def run(group_a: list = None, group_b: list = None,
        n_perm: int = 10000, seed: int = 21) -> dict:
    if group_a is None:
        group_a = DEFAULT_A
    if group_b is None:
        group_b = DEFAULT_B

    rng = np.random.default_rng(seed)
    a, b = np.array(group_a), np.array(group_b)
    obs_stat = b.mean() - a.mean()   # matches R: mean(group_B) - mean(group_A)

    combined = np.concatenate([a, b])
    n_a = len(a)

    null_dist = np.zeros(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(combined)
        null_dist[i] = perm[n_a:].mean() - perm[:n_a].mean()

    p_val = np.mean(np.abs(null_dist) >= np.abs(obs_stat))

    return dict(
        obs_stat=obs_stat,
        p_val=p_val,
        significant=p_val < 0.05,
        null_dist=null_dist,
        mean_a=float(a.mean()),
        mean_b=float(b.mean()),
    )


def plot(res: dict) -> go.Figure:
    nd = res["null_dist"]
    counts, edges = np.histogram(nd, bins=60, density=True)
    centers = (edges[:-1] + edges[1:]) / 2
    max_y = counts.max() * 1.15
    obs = res["obs_stat"]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=centers, y=counts,
        marker_color="rgba(159,225,203,0.35)",
        marker_line_color="#1D9E75", marker_line_width=0.4,
        name="Null distribution",
    ))

    # Observed stat and its mirror (two-tailed)
    for val, dash, label in [
        (obs, "solid", f"Observed Δ = {obs:.3f}"),
        (-obs, "dash", f"−Observed Δ"),
    ]:
        fig.add_trace(go.Scatter(
            x=[val, val], y=[0, max_y],
            mode="lines", line=dict(color="#E24B4A", width=2, dash=dash),
            name=label,
        ))

    fig.update_layout(
        title=f"Permutation null distribution  (p = {res['p_val']:.4f})",
        xaxis_title="Difference in means", yaxis_title="Density",
        height=420, paper_bgcolor="#141720", plot_bgcolor="#0d0f12",
        font=dict(color="#8b91a8", size=11), margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)"), bargap=0,
    )
    fig.update_xaxes(gridcolor="#2a2f3e")
    fig.update_yaxes(gridcolor="#2a2f3e")
    return fig
