"""
IRLS step-by-step convergence visualiser
Shows every iteration: η, p, W, z, β update, log-likelihood
Works for both Binomial and Poisson GLM.
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def run_all_iters(y: np.ndarray, x: np.ndarray,
                  family: str = "binomial", tol: float = 1e-8, maxit: int = 50):
    """
    Returns full per-iteration detail for visualisation.
    family: 'binomial' or 'poisson'
    """
    n = len(y)
    X = np.column_stack([np.ones(n), x])
    m = np.ones(n)
    beta = np.zeros(2)

    all_iters = []

    for it in range(maxit):
        eta = X @ beta

        if family == "binomial":
            p = sigmoid(eta)
            mu = m * p
            W = m * p * (1 - p)
            W = np.maximum(W, 1e-10)
            z = eta + (y - mu) / W
            ll = float(np.sum(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12)))
        else:  # poisson
            mu = np.exp(np.clip(eta, -20, 20))
            W = np.maximum(mu, 1e-10)
            z = eta + (y - mu) / W
            ll = float(np.sum(y * np.log(mu + 1e-12) - mu))

        XtWX = X.T @ (W[:, None] * X)
        XtWz = X.T @ (W * z)
        try:
            beta_new = np.linalg.solve(XtWX, XtWz)
        except np.linalg.LinAlgError:
            break

        delta = float(np.max(np.abs(beta_new - beta)))

        all_iters.append({
            "iter": it,
            "beta": beta.copy(),
            "beta_new": beta_new.copy(),
            "eta": eta.copy(),
            "mu": mu.copy(),
            "W": W.copy(),
            "z": z.copy(),
            "ll": ll,
            "delta": delta,
        })

        beta = beta_new
        if delta < tol:
            break

    return all_iters, X


def run(y_input: list = None, x_input: list = None,
        family: str = "binomial") -> dict:
    if y_input is None:
        # Default: 5 observations as in study guide examples
        x_input = [1, 2, 3, 4, 5]
        y_input = [0, 0, 1, 1, 1]

    x = np.array(x_input, dtype=float)
    y = np.array(y_input, dtype=float)

    all_iters, X = run_all_iters(y, x, family)
    return dict(all_iters=all_iters, x=x, y=y, X=X, family=family)


def plot(res: dict, selected_iter: int = None) -> go.Figure:
    iters = res["all_iters"]
    if not iters:
        fig = go.Figure()
        fig.update_layout(paper_bgcolor="#141720")
        return fig

    if selected_iter is None:
        selected_iter = len(iters) - 1
    selected_iter = min(selected_iter, len(iters) - 1)

    iter_nums = [d["iter"] for d in iters]
    b0s = [d["beta"][0] for d in iters]
    b1s = [d["beta"][1] for d in iters]
    lls = [d["ll"] for d in iters]
    deltas = [d["delta"] for d in iters]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "β₀ and β₁ convergence",
            "Log-likelihood per iteration",
            "Weights W at selected iter",
            "Working response z at selected iter",
        ),
    )

    # β trace
    fig.add_trace(go.Scatter(
        x=iter_nums, y=b0s, mode="lines+markers",
        line=dict(color="#5DCAA5", width=2), marker=dict(size=5),
        name="β₀",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=iter_nums, y=b1s, mode="lines+markers",
        line=dict(color="#EF9F27", width=2), marker=dict(size=5),
        name="β₁",
    ), row=1, col=1)
    # Highlight selected
    fig.add_vline(x=selected_iter, line_dash="dash",
                  line_color="rgba(255,255,255,0.3)", row=1, col=1)

    # Log-likelihood
    fig.add_trace(go.Scatter(
        x=iter_nums, y=lls, mode="lines+markers",
        line=dict(color="#7F77DD", width=2), marker=dict(size=5),
        name="ℓ(β)",
    ), row=1, col=2)
    fig.add_vline(x=selected_iter, line_dash="dash",
                  line_color="rgba(255,255,255,0.3)", row=1, col=2)

    # W at selected iter
    cur = iters[selected_iter]
    obs_idx = list(range(1, len(cur["W"]) + 1))
    fig.add_trace(go.Bar(
        x=obs_idx, y=cur["W"],
        marker_color="rgba(93,202,165,0.5)",
        marker_line_color="#5DCAA5", marker_line_width=0.8,
        name=f"W (iter {selected_iter})",
    ), row=2, col=1)

    # z at selected iter
    fig.add_trace(go.Bar(
        x=obs_idx, y=cur["z"],
        marker_color="rgba(239,159,39,0.5)",
        marker_line_color="#EF9F27", marker_line_width=0.8,
        name=f"z (iter {selected_iter})",
    ), row=2, col=2)
    fig.add_hline(y=0, line_color="#E24B4A", line_dash="dash", row=2, col=2)

    # Annotation: current β values
    cb = cur["beta_new"]
    fig.add_annotation(
        text=f"β = [{cb[0]:.4f}, {cb[1]:.4f}]  |  Δβ = {cur['delta']:.2e}",
        xref="paper", yref="paper", x=0.5, y=1.07,
        showarrow=False, font=dict(color="#5DCAA5", size=12),
    )

    fig.update_layout(
        height=580, paper_bgcolor="#141720", plot_bgcolor="#0d0f12",
        font=dict(color="#8b91a8", size=11), margin=dict(l=10, r=10, t=70, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor="#2a2f3e", zerolinecolor="#2a2f3e")
    fig.update_yaxes(gridcolor="#2a2f3e", zerolinecolor="#2a2f3e")
    return fig
