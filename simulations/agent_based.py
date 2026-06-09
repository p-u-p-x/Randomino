"""
Agent-based SIR epidemic model
Exact translation of agent_based.R — no packages, pure logic.
"""
import numpy as np
import plotly.graph_objects as go


def run(N: int = 1000, beta: float = 0.3, gamma: float = 0.05,
        T: int = 100, I0: int = 5, seed: int = 2024) -> dict:
    """
    N     : population size
    beta  : transmission probability per contact
    gamma : daily recovery probability
    T     : simulation days
    I0    : initial infected
    seed  : for reproducibility
    """
    rng = np.random.default_rng(seed)

    # status: 0=S, 1=I, 2=R  (matches R's rep("S",N) + sample infected)
    status = np.zeros(N, dtype=int)
    initial_infected = rng.choice(N, size=I0, replace=False)
    status[initial_infected] = 1

    S_arr, I_arr, R_arr = [], [], []

    for t in range(T):
        infected_idx = np.where(status == 1)[0]
        suscept_idx = np.where(status == 0)[0]

        # Transmission: each S contacts a random I
        if len(suscept_idx) > 0 and len(infected_idx) > 0:
            # contacts <- sample(infected_idx, len(suscept_idx), replace=TRUE)
            # (contacts variable not used directly — only the probability matters)
            exposed = rng.random(len(suscept_idx)) < beta
            status[suscept_idx[exposed]] = 1

        # Recovery
        if len(infected_idx) > 0:
            recover = rng.random(len(infected_idx)) < gamma
            status[infected_idx[recover]] = 2

        S_arr.append(int(np.sum(status == 0)))
        I_arr.append(int(np.sum(status == 1)))
        R_arr.append(int(np.sum(status == 2)))

    S_arr, I_arr, R_arr = np.array(S_arr), np.array(I_arr), np.array(R_arr)
    peak_I = int(I_arr.max())
    peak_day = int(I_arr.argmax())
    R0 = beta / gamma

    return dict(
        S=S_arr, I=I_arr, R=R_arr,
        days=list(range(T)),
        peak_I=peak_I,
        peak_day=peak_day,
        final_R_pct=R_arr[-1] / N * 100,
        R0=R0,
        herd_threshold=(1 - 1 / R0) * 100 if R0 > 1 else 0,
    )


def plot(res: dict) -> go.Figure:
    days = res["days"]
    fig = go.Figure()

    for arr, name, color in [
        (res["S"], "Susceptible", "#378ADD"),
        (res["I"], "Infected", "#E24B4A"),
        (res["R"], "Recovered", "#1D9E75"),
    ]:
        fig.add_trace(go.Scatter(
            x=days, y=arr, mode="lines",
            line=dict(color=color, width=2.2),
            name=name,
        ))

    # Peak marker
    fig.add_vline(
        x=res["peak_day"],
        line_dash="dash", line_color="rgba(239,159,39,0.6)",
        annotation_text=f"Peak day {res['peak_day']}",
        annotation_font_color="#EF9F27",
    )

    fig.update_layout(
        title="Agent-based SIR epidemic",
        xaxis_title="Day", yaxis_title="Agents",
        height=430, paper_bgcolor="#141720", plot_bgcolor="#0d0f12",
        font=dict(color="#8b91a8", size=11), margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor="#2a2f3e")
    fig.update_yaxes(gridcolor="#2a2f3e")
    return fig
