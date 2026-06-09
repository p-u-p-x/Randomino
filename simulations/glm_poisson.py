"""
Poisson GLM — IRLS from scratch
Matches aaaa.pdf (Poisson GLM residuals lecture) formulas exactly.
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def irls_poisson(X: np.ndarray, y: np.ndarray,
                 tol: float = 1e-8, maxit: int = 100):
    """
    IRLS for Poisson GLM with log link.
    Step 1: η = Xβ
    Step 2: μ = exp(η)
    Step 3: W = diag(μ)
    Step 4: z = η + (y - μ)/μ
    Step 5: β_new = (X'WX)^-1 X'Wz
    """
    n, p = X.shape
    beta = np.zeros(p)
    history = [{"beta": beta.copy(), "ll": np.nan, "delta": np.nan}]

    for _ in range(maxit):
        eta = X @ beta
        mu = np.exp(np.clip(eta, -20, 20))
        W = np.maximum(mu, 1e-10)
        z = eta + (y - mu) / W

        XtWX = X.T @ (W[:, None] * X)
        XtWz = X.T @ (W * z)
        try:
            beta_new = np.linalg.solve(XtWX, XtWz)
        except np.linalg.LinAlgError:
            break

        delta = np.max(np.abs(beta_new - beta))
        mu_new = np.exp(np.clip(X @ beta_new, -20, 20))
        ll = float(np.sum(y * np.log(mu_new + 1e-12) - mu_new))
        history.append({"beta": beta_new.copy(), "ll": ll, "delta": delta})
        beta = beta_new
        if delta < tol:
            break

    return beta, history


def run(beta0_true: float = 1.0, beta1_true: float = 0.5,
        n: int = 80, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)

    x = rng.normal(3, 1.0, n)
    mu_true = np.exp(beta0_true + beta1_true * x)
    y = rng.poisson(mu_true).astype(float)

    X = np.column_stack([np.ones(n), x])
    beta_hat, history = irls_poisson(X, y)

    mu_hat = np.exp(X @ beta_hat)

    # Pearson residuals: (y - μ̂) / √μ̂
    pearson = (y - mu_hat) / np.sqrt(mu_hat + 1e-12)

    # Deviance residuals
    with np.errstate(divide="ignore", invalid="ignore"):
        d_i = 2 * (np.where(y > 0, y * np.log(y / (mu_hat + 1e-12)), 0) - (y - mu_hat))
    deviance_resid = np.sign(y - mu_hat) * np.sqrt(np.maximum(d_i, 0))

    total_deviance = float(np.sum(d_i))
    df = n - 2
    phi = total_deviance / df   # dispersion; ~1 = good, >>1 = overdispersion

    return dict(
        x=x, y=y, mu_hat=mu_hat,
        beta_hat=beta_hat,
        pearson=pearson,
        deviance_resid=deviance_resid,
        history=history,
        rate_ratio=float(np.exp(beta_hat[1])),
        deviance=total_deviance,
        phi=phi,
        n_iter=len(history) - 1,
    )


def plot(res: dict) -> go.Figure:
    x, y, mu_hat = res["x"], res["y"], res["mu_hat"]
    beta = res["beta_hat"]

    x_seq = np.linspace(x.min(), x.max(), 200)
    mu_seq = np.exp(beta[0] + beta[1] * x_seq)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Data + fitted Poisson curve",
            "Pearson residuals vs fitted μ̂",
            "β convergence (IRLS)",
            "Log-likelihood climbing",
        ),
    )

    fig.add_trace(go.Scatter(
        x=x, y=y, mode="markers",
        marker=dict(size=5, color="#7F77DD", opacity=0.6),
        name="Observed counts",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=x_seq, y=mu_seq, mode="lines",
        line=dict(color="#AFA9EC", width=2), name="μ̂",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=mu_hat, y=res["pearson"], mode="markers",
        marker=dict(size=4, color="#5DCAA5", opacity=0.6),
        name="Pearson resid",
    ), row=1, col=2)
    fig.add_hline(y=0, line_color="#E24B4A", line_dash="dash", row=1, col=2)

    hist = res["history"]
    iters = list(range(len(hist)))
    fig.add_trace(go.Scatter(
        x=iters, y=[h["beta"][0] for h in hist],
        mode="lines+markers", line=dict(color="#5DCAA5", width=1.8),
        marker=dict(size=4), name="β₀",
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=iters, y=[h["beta"][1] for h in hist],
        mode="lines+markers", line=dict(color="#EF9F27", width=1.8),
        marker=dict(size=4), name="β₁",
    ), row=2, col=1)

    lls = [h["ll"] for h in hist[1:]]
    fig.add_trace(go.Scatter(
        x=iters[1:], y=lls, mode="lines+markers",
        line=dict(color="#D4537E", width=1.8),
        marker=dict(size=4), name="log-lik",
    ), row=2, col=2)

    fig.update_layout(
        height=640, paper_bgcolor="#141720", plot_bgcolor="#0d0f12",
        font=dict(color="#8b91a8", size=11), margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor="#2a2f3e", zerolinecolor="#2a2f3e")
    fig.update_yaxes(gridcolor="#2a2f3e", zerolinecolor="#2a2f3e")
    return fig
