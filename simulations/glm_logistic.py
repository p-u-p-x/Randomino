"""
Logistic Regression (Binomial GLM) — IRLS from scratch
Matches binomial.R and binomial_model.pdf formulas exactly.
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def irls_logistic(X: np.ndarray, y: np.ndarray,
                  m: np.ndarray = None, tol: float = 1e-8, maxit: int = 100):
    """
    IRLS for Binomial GLM.
    X : (n, p) design matrix with intercept column
    y : (n,) success counts
    m : (n,) trial counts (all 1 for binary)
    Returns beta, history of (beta, log_lik, delta_beta) per iteration.
    """
    n, p = X.shape
    if m is None:
        m = np.ones(n)

    beta = np.zeros(p)
    history = [{"beta": beta.copy(), "ll": np.nan, "delta": np.nan}]

    for iteration in range(maxit):
        eta = X @ beta                        # Step 1: η = Xβ
        p_hat = sigmoid(eta)                  # Step 2: p = sigmoid(η)
        mu = m * p_hat                        # μ = m·p

        W = m * p_hat * (1 - p_hat)          # Step 3: weights
        W = np.maximum(W, 1e-10)

        z = eta + (y - mu) / W               # Step 4: working response

        # Step 5: β_new = (X'WX)^-1 X'Wz
        W_diag = np.diag(W)
        XtWX = X.T @ W_diag @ X
        XtWz = X.T @ (W * z)
        try:
            beta_new = np.linalg.solve(XtWX, XtWz)
        except np.linalg.LinAlgError:
            break

        delta = np.max(np.abs(beta_new - beta))
        p_new = sigmoid(X @ beta_new)
        ll = float(np.sum(y * np.log(p_new + 1e-12) + (m - y) * np.log(1 - p_new + 1e-12)))
        history.append({"beta": beta_new.copy(), "ll": ll, "delta": delta})

        beta = beta_new
        if delta < tol:
            break

    return beta, history


def run(beta0_true: float = -3.0, beta1_true: float = 1.5,
        n: int = 80, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)

    # Generate data
    x = rng.normal(3, 1.5, n)
    eta_true = beta0_true + beta1_true * x
    p_true = sigmoid(eta_true)
    y = rng.binomial(1, p_true).astype(float)

    X = np.column_stack([np.ones(n), x])
    m = np.ones(n)

    beta_hat, history = irls_logistic(X, y, m)

    p_hat = sigmoid(X @ beta_hat)

    # Pearson residuals: (y - μ) / sqrt(Var(Y))
    pearson = (y - p_hat) / np.sqrt(p_hat * (1 - p_hat) + 1e-12)

    # Deviance residuals
    with np.errstate(divide="ignore", invalid="ignore"):
        d_i = 2 * (np.where(y > 0, y * np.log(y / (p_hat + 1e-12)), 0) +
                   np.where(y < 1, (1 - y) * np.log((1 - y) / (1 - p_hat + 1e-12)), 0))
    deviance_resid = np.sign(y - p_hat) * np.sqrt(np.maximum(d_i, 0))

    return dict(
        x=x, y=y, p_hat=p_hat,
        beta_hat=beta_hat,
        pearson=pearson,
        deviance_resid=deviance_resid,
        history=history,
        odds_ratio=float(np.exp(beta_hat[1])),
        n_iter=len(history) - 1,
        beta0_true=beta0_true,
        beta1_true=beta1_true,
    )


def plot(res: dict) -> go.Figure:
    x, y, p_hat = res["x"], res["y"], res["p_hat"]
    beta = res["beta_hat"]

    x_seq = np.linspace(x.min(), x.max(), 200)
    p_seq = sigmoid(beta[0] + beta[1] * x_seq)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Data + fitted sigmoid",
            "Pearson residuals vs fitted p̂",
            "β convergence (IRLS)",
            "Log-likelihood climbing",
        ),
    )

    # Data + sigmoid curve
    for val, label, color in [(0, "y=0", "#D85A30"), (1, "y=1", "#1D9E75")]:
        mask = y == val
        fig.add_trace(go.Scatter(
            x=x[mask], y=y[mask], mode="markers",
            marker=dict(size=5, color=color, opacity=0.6),
            name=label,
        ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=x_seq, y=p_seq, mode="lines",
        line=dict(color="#5DCAA5", width=2), name="p̂",
    ), row=1, col=1)

    # Pearson residuals
    fig.add_trace(go.Scatter(
        x=p_hat, y=res["pearson"], mode="markers",
        marker=dict(size=4, color="#7F77DD", opacity=0.6),
        name="Pearson resid",
    ), row=1, col=2)
    fig.add_hline(y=0, line_color="#E24B4A", line_dash="dash", row=1, col=2)

    # β convergence
    hist = res["history"]
    iters = list(range(len(hist)))
    b0s = [h["beta"][0] for h in hist]
    b1s = [h["beta"][1] for h in hist]
    fig.add_trace(go.Scatter(x=iters, y=b0s, mode="lines+markers",
                             line=dict(color="#5DCAA5", width=1.8),
                             marker=dict(size=4), name="β₀"), row=2, col=1)
    fig.add_trace(go.Scatter(x=iters, y=b1s, mode="lines+markers",
                             line=dict(color="#EF9F27", width=1.8),
                             marker=dict(size=4), name="β₁"), row=2, col=1)

    # Log-likelihood
    lls = [h["ll"] for h in hist[1:]]
    fig.add_trace(go.Scatter(x=iters[1:], y=lls, mode="lines+markers",
                             line=dict(color="#7F77DD", width=1.8),
                             marker=dict(size=4), name="log-lik"), row=2, col=2)

    fig.update_layout(
        height=640, paper_bgcolor="#141720", plot_bgcolor="#0d0f12",
        font=dict(color="#8b91a8", size=11), margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        showlegend=True,
    )
    fig.update_xaxes(gridcolor="#2a2f3e", zerolinecolor="#2a2f3e")
    fig.update_yaxes(gridcolor="#2a2f3e", zerolinecolor="#2a2f3e")
    return fig
