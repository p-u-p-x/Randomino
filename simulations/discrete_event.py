"""
Discrete-event simulation — M/M/1 queue
Python replacement for R's simmer package.
Logic is identical: Poisson arrivals, Exp service, single server.
Exact translation of discrete_event.R
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def run(lam: float = 4.0, mu: float = 5.0,
        sim_hours: float = 8.0, seed: int = 42) -> dict:
    """
    lam  : arrival rate (customers / hr)
    mu   : service rate (customers / hr)
    Returns dict with wait times, queue lengths, and theoretical values.
    """
    if lam >= mu:
        return dict(error=True, msg=f"ρ = {lam/mu:.2f} ≥ 1 — queue is unstable (λ must be < μ)")

    rng = np.random.default_rng(seed)
    T_total = sim_hours * 60   # convert to minutes for readability
    lam_min = lam / 60         # per-minute rates
    mu_min = mu / 60

    t = 0.0
    server_free_at = 0.0
    wait_times = []
    arrival_times = []
    departure_times = []

    # Generate arrivals until T_total
    while True:
        inter = rng.exponential(1 / lam_min)
        t += inter
        if t > T_total:
            break
        arrival_times.append(t)
        start_svc = max(t, server_free_at)
        wait = start_svc - t
        svc_time = rng.exponential(1 / mu_min)
        server_free_at = start_svc + svc_time
        wait_times.append(wait)
        departure_times.append(server_free_at)

    wait_times = np.array(wait_times)
    arrival_times = np.array(arrival_times)
    departure_times = np.array(departure_times)

    # Queue length over 200 time snapshots
    snap_times = np.linspace(0, T_total, 200)
    q_lens = [
        int(np.sum((arrival_times <= t) & (departure_times >= t)))
        for t in snap_times
    ]

    avg_wait = wait_times.mean()
    theoretical = (lam / (mu * (mu - lam))) * 60   # Erlang formula → minutes
    rho = lam / mu

    return dict(
        error=False,
        avg_wait=avg_wait,
        theoretical=theoretical,
        rho=rho,
        n_customers=len(wait_times),
        wait_times=wait_times,
        snap_times=snap_times,
        q_lens=q_lens,
    )


def plot(res: dict) -> go.Figure:
    if res.get("error"):
        fig = go.Figure()
        fig.add_annotation(text=res["msg"], xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False,
                           font=dict(color="#E24B4A", size=14))
        fig.update_layout(paper_bgcolor="#141720", plot_bgcolor="#0d0f12", height=350)
        return fig

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Wait time distribution (min)", "Queue length over time"))

    wt = res["wait_times"]
    counts, edges = np.histogram(wt, bins=30, density=True)
    centers = (edges[:-1] + edges[1:]) / 2
    fig.add_trace(go.Bar(
        x=centers, y=counts,
        marker_color="rgba(216,90,48,0.45)",
        marker_line_color="#D85A30", marker_line_width=0.4,
        name="Wait times",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=res["snap_times"] / 60, y=res["q_lens"],
        mode="lines", fill="tozeroy",
        fillcolor="rgba(216,90,48,0.15)",
        line=dict(color="#F0997B", width=1.4),
        name="Queue length",
    ), row=1, col=2)

    fig.update_layout(
        height=380, paper_bgcolor="#141720", plot_bgcolor="#0d0f12",
        font=dict(color="#8b91a8", size=11), margin=dict(l=10, r=10, t=40, b=10),
        bargap=0, legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor="#2a2f3e", zerolinecolor="#2a2f3e")
    fig.update_yaxes(gridcolor="#2a2f3e", zerolinecolor="#2a2f3e")
    fig.update_xaxes(title_text="wait (min)", row=1, col=1)
    fig.update_xaxes(title_text="time (hr)", row=1, col=2)
    return fig
