"""
MCMC — Metropolis-Hastings for Normal posterior
Prior: μ ~ N(0, 10²),  Likelihood: X|μ ~ N(μ, 1)
Exact translation of MCMC.R
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats


OBS = np.array([2.1, 1.8, 2.5, 1.9, 2.3])   # fixed observed data from MCMC.R


def log_posterior(mu: float, obs: np.ndarray = OBS) -> float:
    """log prior  + log likelihood  (identical to R's log_posterior)"""
    log_prior = stats.norm.logpdf(mu, 0, 10)
    log_lik = np.sum(stats.norm.logpdf(obs, mu, 1))
    return log_prior + log_lik


def run(n_iter: int = 20000, proposal_sd: float = 0.5,
        burnin_pct: float = 25, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)

    chain = np.zeros(n_iter)
    chain[0] = 0.0
    accepted = 0

    for i in range(1, n_iter):
        proposal = chain[i - 1] + rng.normal(0, proposal_sd)
        log_alpha = log_posterior(proposal) - log_posterior(chain[i - 1])
        if np.log(rng.uniform()) < log_alpha:
            chain[i] = proposal
            accepted += 1
        else:
            chain[i] = chain[i - 1]

    burnin = int(n_iter * burnin_pct / 100)
    post = chain[burnin:]
    accept_rate = accepted / (n_iter - 1)

    return dict(
        chain=chain,
        post=post,
        posterior_mean=post.mean(),
        ci_lo=np.quantile(post, 0.025),
        ci_hi=np.quantile(post, 0.975),
        accept_rate=accept_rate * 100,
    )


def plot(res: dict) -> go.Figure:
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Trace plot", "Posterior of μ"))

    # Trace — show last 2000 as in R code
    trace_vals = res["chain"][-2000:]
    fig.add_trace(go.Scatter(
        y=trace_vals, mode="lines",
        line=dict(color="#EF9F27", width=0.8),
        name="Chain",
    ), row=1, col=1)

    # Posterior histogram
    post = res["post"]
    counts, edges = np.histogram(post, bins=60, density=True)
    centers = (edges[:-1] + edges[1:]) / 2
    fig.add_trace(go.Bar(
        x=centers, y=counts,
        marker_color="rgba(239,159,39,0.4)",
        marker_line_color="#EF9F27", marker_line_width=0.4,
        name="Posterior",
    ), row=1, col=2)

    max_y = counts.max() * 1.15
    for val, label in [(res["ci_lo"], "2.5%"), (res["ci_hi"], "97.5%"),
                       (res["posterior_mean"], "Mean")]:
        fig.add_trace(go.Scatter(
            x=[val, val], y=[0, max_y],
            mode="lines",
            line=dict(color="#E24B4A", width=1.5, dash="dash"),
            name=label, showlegend=False,
        ), row=1, col=2)

    fig.update_layout(
        height=380, paper_bgcolor="#141720", plot_bgcolor="#0d0f12",
        font=dict(color="#8b91a8", size=11), margin=dict(l=10, r=10, t=40, b=10),
        bargap=0, legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor="#2a2f3e", zerolinecolor="#2a2f3e")
    fig.update_yaxes(gridcolor="#2a2f3e", zerolinecolor="#2a2f3e")
    return fig
