"""
Randomino  —  Statistical Simulation Dashboard
Run:  streamlit run app.py
"""
import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="Randomino",
    page_icon="🎲",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
#  GLOBAL CSS  (blue accent, fixed sidebar scroll, original icon sizes)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #070a0f;
    color: #dde1ee;
}
.stApp { background: #070a0f; }

/* ── sidebar shell ── */
section[data-testid="stSidebar"] {
    background: #0b0f18;
    border-right: 1px solid #171e2e;
    min-width: 80px !important;
    max-width: 80px !important;
}
section[data-testid="stSidebar"] > div:first-child {
    padding: 0 !important;
    overflow-y: auto;                /* scroll only when icons overflow */
    height: 100vh;                   /* fill viewport, but scroll works within it */
}

/* ── icon nav ── */
.nav-shell {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 12px 0 16px;
    gap: 0;
    height: auto;                    /* no forced height → no extra empty space */
    min-height: min-content;
}
.nav-brand {
    display: flex;
    flex-direction: column;
    align-items: center;
    margin-bottom: 14px;
    gap: 3px;
}
.nav-brand svg { width: 28px; height: 28px; }
.nav-brand-lbl {
    font-size: 7px;
    letter-spacing: .14em;
    text-transform: uppercase;
    color: #3B82F6;
    font-family: 'JetBrains Mono', monospace;
}
.nav-sep {
    width: 40px;
    height: 1px;
    background: linear-gradient(90deg, transparent, #1a2235, transparent);
    margin: 8px 0 6px;
}
.nav-grp-lbl {
    font-size: 7.5px;
    letter-spacing: .14em;
    text-transform: uppercase;
    color: #2a3350;
    font-family: 'JetBrains Mono', monospace;
    margin: 4px 0 5px;
}
/* ── navigation links (original sizes, no underline) ── */
.ni {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    width: 58px;
    height: 54px;
    border-radius: 12px;
    cursor: pointer;
    transition: background .15s, box-shadow .15s, border-color .15s;
    background: transparent;
    border: 1px solid transparent;
    margin: 2px 0;
    gap: 3px;
    text-decoration: none;
    user-select: none;
}
.ni:hover {
    background: rgba(59,130,246,0.07);
    border-color: rgba(59,130,246,0.18);
}
.ni.on {
    background: rgba(59,130,246,0.11);
    border-color: rgba(59,130,246,0.32);
    box-shadow: 0 0 14px rgba(59,130,246,0.12);
}
.ni svg { width: 20px; height: 20px; }
.ni-lbl {
    font-size: 7px;
    color: #3a4565;
    letter-spacing: .03em;
    text-align: center;
    line-height: 1.25;
    max-width: 54px;
}
.ni.on .ni-lbl { color: #3B82F6; }

/* ── page header ── */
.ph {
    display: flex;
    align-items: flex-start;
    gap: 14px;
    padding: 18px 24px 14px;
    border-bottom: 1px solid #171e2e;
    margin-bottom: 18px;
}
.ph-icon-wrap {
    width: 42px; height: 42px;
    border-radius: 11px;
    background: rgba(59,130,246,0.09);
    border: 1px solid rgba(59,130,246,0.2);
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
}
.ph-icon-wrap svg { width: 22px; height: 22px; }
.ph-title { font-size: 17px; font-weight: 500; color: #dde1ee; margin: 0 0 3px; }
.ph-desc  { font-size: 12px; color: #4a5370; line-height: 1.5; margin: 0; max-width: 580px; }

/* ── glass panel ── */
.gp {
    background: linear-gradient(145deg, rgba(255,255,255,0.022) 0%, rgba(255,255,255,0.006) 100%);
    border: 1px solid #171e2e;
    border-radius: 14px;
    padding: 16px 16px 14px;
    margin-bottom: 12px;
}
.gp-title {
    font-size: 9.5px;
    font-weight: 500;
    letter-spacing: .12em;
    text-transform: uppercase;
    color: #2a3350;
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 13px;
    padding-bottom: 8px;
    border-bottom: 1px solid #131929;
}

/* ── metrics 2-col inside panel ── */
.mg { display: grid; grid-template-columns: 1fr 1fr; gap: 7px; }
.mc {
    background: rgba(7,10,15,0.7);
    border: 1px solid #171e2e;
    border-radius: 9px;
    padding: 10px 11px;
    position: relative;
    overflow: hidden;
}
.mc::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    border-radius: 9px 9px 0 0;
    background: #3B82F6;
}
.mc.am::before { background: #F0A855; }
.mc.bl::before { background: #5B9CF6; }
.mc.cr::before { background: #F0756A; }
.mc.pu::before { background: #A78BFA; }
.mv {
    font-family: 'JetBrains Mono', monospace;
    font-size: 16px;
    font-weight: 500;
    color: #3B82F6;
    line-height: 1.2;
}
.mc.am .mv { color: #F0A855; }
.mc.bl .mv { color: #5B9CF6; }
.mc.cr .mv { color: #F0756A; }
.mc.pu .mv { color: #A78BFA; }
.ml { font-size: 9.5px; color: #2a3350; margin-top: 4px; text-transform: uppercase; letter-spacing: .07em; }

/* ── formula strip ── */
.fs {
    background: rgba(59,130,246,0.04);
    border: 1px solid rgba(59,130,246,0.13);
    border-radius: 8px;
    padding: 8px 13px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: #3B82F6;
    margin: 0 0 16px;
    line-height: 1.7;
}

/* ── info / warn / ok strips ── */
.is { background: rgba(91,156,246,0.05); border-left: 2px solid #5B9CF6;
      border-radius: 0 7px 7px 0; padding: 8px 12px; font-size: 11.5px;
      color: #7a87aa; margin: 8px 0; line-height: 1.55; }
.ws { background: rgba(240,168,85,0.05); border-left: 2px solid #F0A855;
      border-radius: 0 7px 7px 0; padding: 8px 12px; font-size: 11.5px;
      color: #7a87aa; margin: 8px 0; line-height: 1.55; }
.os { background: rgba(59,130,246,0.05); border-left: 2px solid #3B82F6;
      border-radius: 0 7px 7px 0; padding: 8px 12px; font-size: 11.5px;
      color: #7a87aa; margin: 8px 0; line-height: 1.55; }

/* ── chart wrapper ── */
div[data-testid="stPlotlyChart"] {
    background: linear-gradient(145deg, rgba(255,255,255,0.018) 0%, rgba(255,255,255,0.005) 100%);
    border: 1px solid #171e2e;
    border-radius: 16px;
    overflow: hidden;
    padding: 6px 6px 2px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.35);
}

/* ── chart label ── */
.chart-label {
    font-size: 9.5px;
    font-weight: 500;
    letter-spacing: .12em;
    text-transform: uppercase;
    color: #2a3350;
    font-family: 'JetBrains Mono', monospace;
    margin: 0 0 6px 4px;
}

/* ── slider ── */
.stSlider > label { font-size: 11.5px !important; color: #4a5370 !important; }
.stSlider [data-baseweb="slider"] [role="slider"] {
    background: #3B82F6 !important;
    border: none !important;
    box-shadow: 0 0 8px rgba(59,130,246,0.5) !important;
}

/* ── run button ── */
.stButton > button {
    background: linear-gradient(135deg, #2563EB 0%, #3B82F6 100%);
    color: #070a0f;
    border: none;
    border-radius: 9px;
    font-weight: 600;
    font-size: 12px;
    padding: 10px 0;
    width: 100%;
    letter-spacing: .05em;
    transition: all .18s;
    box-shadow: 0 2px 14px rgba(59,130,246,0.22);
}
.stButton > button:hover {
    box-shadow: 0 4px 22px rgba(59,130,246,0.38);
    transform: translateY(-1px);
}
.stButton > button:active { transform: translateY(0); }

/* ── inputs ── */
.stTextInput input {
    background: #0b0f18 !important;
    border: 1px solid #1a2235 !important;
    border-radius: 8px !important;
    color: #dde1ee !important;
    font-size: 12px !important;
    font-family: 'JetBrains Mono', monospace !important;
}
.stTextInput label { font-size: 11px !important; color: #2a3350 !important; }
.stSelectbox > div > div {
    background: #0b0f18 !important;
    border: 1px solid #1a2235 !important;
    border-radius: 8px !important;
    color: #dde1ee !important;
    font-size: 12px !important;
}
.stSelectbox label { font-size: 11px !important; color: #2a3350 !important; }

/* ── headings override ── */
h1 { font-size: 17px !important; font-weight: 500 !important; color: #3B82F6 !important; margin: 0 !important; }
h2 { font-size: 14px !important; font-weight: 500 !important; color: #dde1ee !important; }

/* ── expander ── */
.stExpander {
    border: 1px solid #171e2e !important;
    border-radius: 10px !important;
    background: #0b0f18 !important;
}
.stExpander summary { font-size: 11.5px !important; color: #4a5370 !important; }

/* ── dataframe ── */
div[data-testid="stDataFrame"] {
    border-radius: 10px; overflow: hidden; border: 1px solid #171e2e;
}

/* ── spinner ── */
.stSpinner > div { border-top-color: #3B82F6 !important; }

/* ── hide default metric ── */
div[data-testid="metric-container"] { display: none; }

/* ── scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #070a0f; }
::-webkit-scrollbar-thumb { background: #1a2235; border-radius: 2px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  SVG icon library
# ─────────────────────────────────────────────────────────────────────────────
def svg(path_d, stroke="#3B82F6", fill="none", vb="0 0 24 24", extra=""):
    return (f'<svg viewBox="{vb}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" {extra}>'
            f'{path_d}</svg>')

ICONS = {
    "mc":    svg('<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="3"/>'
                 '<line x1="12" y1="2" x2="12" y2="5"/><line x1="12" y1="19" x2="12" y2="22"/>'),
    "boot":  svg('<polyline points="23 4 23 10 17 10"/><polyline points="1 20 1 14 7 14"/>'
                 '<path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/>'),
    "mcmc":  svg('<path d="M2 12h4l3-9 4 18 3-9h6"/>'),
    "des":   svg('<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>'),
    "icdf":  svg('<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>'),
    "rej":   svg('<circle cx="18" cy="18" r="3"/><circle cx="6" cy="6" r="3"/>'
                 '<path d="M6 9v12h12V9"/><path d="M6 9a6 6 0 0 0 6 6 6 6 0 0 0 6-6"/>'),
    "sir":   svg('<path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/>'
                 '<circle cx="9" cy="7" r="4"/>'
                 '<path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/>'),
    "perm":  svg('<line x1="8" y1="6" x2="21" y2="6"/><line x1="8" y1="12" x2="21" y2="12"/>'
                 '<line x1="8" y1="18" x2="21" y2="18"/>'
                 '<line x1="3" y1="6" x2="3.01" y2="6"/><line x1="3" y1="12" x2="3.01" y2="12"/>'
                 '<line x1="3" y1="18" x2="3.01" y2="18"/>'),
    "logit": svg('<path d="M2 20h20"/>'
                 '<path d="M6 20V10l6-8 6 8v10"/>'
                 '<path d="M6 12a6 6 0 0 0 12 0"/>'),
    "pois":  svg('<line x1="18" y1="20" x2="18" y2="10"/>'
                 '<line x1="12" y1="20" x2="12" y2="4"/>'
                 '<line x1="6" y1="20" x2="6" y2="14"/>'),
    "irls":  svg('<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>'),
    "brand": svg('<line x1="18" y1="20" x2="18" y2="10"/>'
                 '<line x1="12" y1="20" x2="12" y2="4"/>'
                 '<line x1="6" y1="20" x2="6" y2="14"/>',
                 stroke="#3B82F6", fill="none"),
}

# ─────────────────────────────────────────────────────────────────────────────
#  SIMULATION MODULES
# ─────────────────────────────────────────────────────────────────────────────
from simulations import (
    monte_carlo, bootstrap, mcmc, discrete_event,
    inverse_cdf, rejection_sampling, agent_based, permutation,
    glm_logistic, glm_poisson, glm_irls,
)

MODULES = [
    ("Monte Carlo",               "mc",    "Monte\nCarlo", "sim"),
    ("Bootstrap",                 "boot",  "Boot\nstrap",  "sim"),
    ("MCMC",                      "mcmc",  "MCMC",         "sim"),
    ("Discrete-Event (M/M/1)",    "des",   "M/M/1",        "sim"),
    ("Inverse CDF",               "icdf",  "Inv CDF",      "sim"),
    ("Rejection Sampling",        "rej",   "Reject",       "sim"),
    ("Agent-based SIR",           "sir",   "SIR",          "sim"),
    ("Permutation Test",          "perm",  "Permut.",      "sim"),
    ("Logistic Regression",       "logit", "Logistic",     "glm"),
    ("Poisson GLM",               "pois",  "Poisson",      "glm"),
    ("IRLS Convergence",          "irls",  "IRLS",         "glm"),
]

# ── Determine active module from URL query parameter ──
module = st.query_params.get("module", "Monte Carlo")

# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR – anchor links with JS to force same-tab navigation
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    _nav = ['<div class="nav-shell">',
            '<div class="nav-brand">',
            ICONS["brand"],
            '<span class="nav-brand-lbl">Randomino</span>',
            '</div>',
            '<div class="nav-sep"></div>',
            '<span class="nav-grp-lbl">SIM</span>']

    prev_group = "sim"
    for key, ik, lbl, grp in MODULES:
        if grp != prev_group:
            _nav.append('<div class="nav-sep"></div>')
            _nav.append('<span class="nav-grp-lbl">GLM</span>')
            prev_group = grp
        active = " on" if key == module else ""
        icon_svg = ICONS.get(ik, ICONS["mc"])
        _nav.append(
            f'<a class="ni{active}" href="?module={key}" target="_self">'  # target="_self" explicitly
            f'{icon_svg}'
            f'<span class="ni-lbl">{lbl}</span>'
            f'</a>'
        )

    _nav.append('</div>')
    st.markdown("".join(_nav), unsafe_allow_html=True)

    # JavaScript to ensure navigation stays in same tab (prevent any new window)
    st.markdown("""
    <script>
    (function() {
        const links = window.parent.document.querySelectorAll('.ni[href]');
        links.forEach(link => {
            link.addEventListener('click', function(e) {
                e.preventDefault();
                const url = this.getAttribute('href');
                if (url) {
                    window.parent.location.href = url;
                }
            });
        });
    })();
    </script>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  UI HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def page_header(icon_key, title, desc):
    svg_icon = ICONS.get(icon_key, ICONS["mc"])
    st.markdown(f"""
    <div class="ph">
      <div class="ph-icon-wrap">{svg_icon}</div>
      <div>
        <p class="ph-title">{title}</p>
        <p class="ph-desc">{desc}</p>
      </div>
    </div>""", unsafe_allow_html=True)

def formula(txt):       st.markdown(f'<div class="fs">{txt}</div>', unsafe_allow_html=True)
def info(txt):          st.markdown(f'<div class="is">{txt}</div>', unsafe_allow_html=True)
def warn(txt):          st.markdown(f'<div class="ws">{txt}</div>', unsafe_allow_html=True)
def ok(txt):            st.markdown(f'<div class="os">{txt}</div>', unsafe_allow_html=True)

def gp_open(title=""):
    html = '<div class="gp">'
    if title: html += f'<div class="gp-title">{title}</div>'
    st.markdown(html, unsafe_allow_html=True)

def gp_close():         st.markdown('</div>', unsafe_allow_html=True)

def metrics(*cards):
    html = '<div class="mg">'
    for val, lbl, cls in cards:
        html += f'<div class="mc {cls}"><div class="mv">{val}</div><div class="ml">{lbl}</div></div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

def chart_label(txt):   st.markdown(f'<p class="chart-label">{txt}</p>', unsafe_allow_html=True)
def cols():             return st.columns([1, 2.6], gap="medium")

def plotly_defaults(fig, height=380):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#4a5370", size=10),
        margin=dict(l=8, r=8, t=32, b=8),
        legend=dict(bgcolor="rgba(11,15,24,0.7)", bordercolor="#171e2e", borderwidth=1, font=dict(size=10)),
    )
    fig.update_xaxes(gridcolor="#131929", zerolinecolor="#1a2235", tickfont=dict(size=10), title_font=dict(size=10))
    fig.update_yaxes(gridcolor="#131929", zerolinecolor="#1a2235", tickfont=dict(size=10), title_font=dict(size=10))
    return fig

# ─────────────────────────────────────────────────────────────────────────────
#  SIMULATION PAGES (unchanged – all 11 modules)
# ─────────────────────────────────────────────────────────────────────────────
if module == "Monte Carlo":
    page_header("mc", "Monte Carlo — π Estimation",
                "Throw random points into a unit square. Count how many land inside the quarter-circle. Ratio × 4 → π.")
    formula("π  ≈  4 × (points inside circle) / (total points n)")

    left, right = cols()
    with left:
        gp_open("Parameters")
        n    = st.slider("Points  n", 500, 50000, 5000, 500)
        seed = st.slider("Random seed", 1, 100, 42)
        run_btn = st.button("▶  Simulate", key="mc_run")
        gp_close()

    if run_btn or "mc_res" not in st.session_state:
        with st.spinner("Simulating…"):
            st.session_state.mc_res = monte_carlo.run(n, seed)
    res = st.session_state.mc_res

    with left:
        gp_open("Statistics")
        metrics(
            (f"{res['pi_est']:.6f}", "π Estimate",      ""),
            (f"{res['error']:.6f}",  "Absolute error",  "am"),
            (f"{res['pct_inside']:.2f}%", "Points inside", "bl"),
            (f"{n:,}",               "Total points",    ""),
        )
        gp_close()
        info("Accuracy improves as <b>√n</b> — doubling accuracy needs 4× the points (Monte Carlo convergence rate).")

    with right:
        fig = monte_carlo.plot(res)
        plotly_defaults(fig, height=400)
        chart_label("Point scatter + π convergence")
        st.plotly_chart(fig, use_container_width=True)

elif module == "Bootstrap":
    page_header("boot", "Bootstrap — 95 % CI for the Median",
                "Resample observed data with replacement B times. Compute median each time. Middle 95 % of that distribution = confidence interval.")
    formula("CI  =  [ Q₂.₅%(bootstrap medians),  Q₉₇.₅%(bootstrap medians) ]")

    left, right = cols()
    with left:
        gp_open("Parameters")
        n    = st.slider("Sample size  n", 10, 300, 50, 5)
        B    = st.slider("Resamples  B", 500, 10000, 3000, 500)
        rate = st.slider("Exp rate  λ", 0.1, 2.0, 0.5, 0.1)
        seed = st.slider("Seed", 1, 200, 123)
        run_btn = st.button("▶  Bootstrap", key="boot_run")
        gp_close()

    if run_btn or "boot_res" not in st.session_state:
        with st.spinner("Resampling…"):
            st.session_state.boot_res = bootstrap.run(n, B, rate, seed)
    res = st.session_state.boot_res

    with left:
        gp_open("Statistics")
        metrics(
            (f"{res['sample_median']:.4f}",      "Sample median",    ""),
            (f"{res['theoretical_median']:.4f}", "True median ln2/λ","bl"),
            (f"{res['ci_lo']:.4f}",               "CI lower  2.5 %",  "am"),
            (f"{res['ci_hi']:.4f}",               "CI upper  97.5 %", "am"),
        )
        gp_close()
        ok(f"CI width = <b>{res['ci_width']:.4f}</b>. Larger n → narrower CI.")

    with right:
        fig = bootstrap.plot(res)
        plotly_defaults(fig, height=400)
        chart_label("Bootstrap distribution of median (red = 95 % CI bounds)")
        st.plotly_chart(fig, use_container_width=True)

elif module == "MCMC":
    page_header("mcmc", "MCMC — Metropolis-Hastings",
                "Sample the posterior of μ.  Prior: N(0,100).  Likelihood: data ~ N(μ,1).  Observed: [2.1, 1.8, 2.5, 1.9, 2.3].")
    formula("Accept proposal μ*  if  log U  <  log p(μ*|data) − log p(μᵗ|data)")

    left, right = cols()
    with left:
        gp_open("Parameters")
        proposal_sd  = st.slider("Proposal  σ", 0.05, 3.0, 0.5, 0.05)
        n_iter       = st.slider("Iterations", 1000, 20000, 10000, 1000)
        burnin_pct   = st.slider("Burn-in  %", 10, 50, 25, 5)
        seed         = st.slider("Seed", 1, 100, 7)
        run_btn      = st.button("▶  Run chain", key="mcmc_run")
        gp_close()

    if run_btn or "mcmc_res" not in st.session_state:
        with st.spinner("Running MCMC chain…"):
            st.session_state.mcmc_res = mcmc.run(n_iter, proposal_sd, burnin_pct, seed)
    res = st.session_state.mcmc_res

    with left:
        gp_open("Posterior Statistics")
        acc_col = "am" if res["accept_rate"] < 20 or res["accept_rate"] > 60 else ""
        metrics(
            (f"{res['posterior_mean']:.4f}", "Posterior mean",   ""),
            (f"{res['accept_rate']:.1f}%",   "Accept rate",      acc_col),
            (f"{res['ci_lo']:.4f}",           "95 % CrI lower",  "bl"),
            (f"{res['ci_hi']:.4f}",           "95 % CrI upper",  "bl"),
        )
        gp_close()
        warn("Ideal accept rate: 23–50 %.  Too high → tiny steps (slow mixing).  Too low → large steps (mostly rejected).  Tune σ.")

    with right:
        fig = mcmc.plot(res)
        plotly_defaults(fig, height=400)
        chart_label("Trace plot (chain history) + posterior histogram")
        st.plotly_chart(fig, use_container_width=True)

elif module == "Discrete-Event (M/M/1)":
    page_header("des", "Discrete-Event Simulation — M/M/1 Queue",
                "Arrivals: Poisson(λ).  Service: Exp(μ).  One server.  Compare simulated wait times to Erlang formula.")
    formula("ρ = λ/μ   (must be < 1)   |   Theoretical wait  W = λ / [ μ(μ−λ) ]  minutes")

    left, right = cols()
    with left:
        gp_open("Parameters")
        lam       = st.slider("Arrival rate  λ  (/hr)", 1.0, 9.0, 4.0, 0.5)
        mu        = st.slider("Service rate  μ  (/hr)", 2.0, 15.0, 6.0, 0.5)
        sim_hours = st.slider("Simulation hours", 2, 24, 8)
        seed      = st.slider("Seed", 1, 100, 42)
        run_btn   = st.button("▶  Simulate queue", key="des_run")
        gp_close()

    if lam >= mu:
        st.error(f"ρ = {lam/mu:.2f} ≥ 1 — queue diverges. Increase μ or decrease λ.")
        st.stop()

    if run_btn or "des_res" not in st.session_state:
        with st.spinner("Processing event log…"):
            st.session_state.des_res = discrete_event.run(lam, mu, sim_hours, seed)
    res = st.session_state.des_res

    if res.get("error"):
        st.error(res["msg"])
        st.stop()

    with left:
        gp_open("Statistics")
        rho_col = "cr" if res["rho"] > 0.85 else ("am" if res["rho"] > 0.7 else "")
        metrics(
            (f"{res['avg_wait']:.2f} min",   "Avg wait (sim)",    ""),
            (f"{res['theoretical']:.2f} min","Erlang theory",     "bl"),
            (f"{res['rho']:.3f}",             "Utilisation  ρ",   rho_col),
            (f"{res['n_customers']}",         "Customers served", ""),
        )
        gp_close()
        if res["rho"] > 0.85:
            warn(f"ρ = {res['rho']:.2f} — server near saturation. Expect long queues.")
        else:
            ok(f"System stable at ρ = {res['rho']:.2f}.")

    with right:
        fig = discrete_event.plot(res)
        plotly_defaults(fig, height=400)
        chart_label("Wait time histogram + queue length over time")
        st.plotly_chart(fig, use_container_width=True)

elif module == "Inverse CDF":
    page_header("icdf", "Inverse CDF — Triangular Distribution",
                "Draw U ~ Uniform(0,1).  Apply the analytically inverted CDF F⁻¹(U).  Exact samples, no rejection overhead.")
    formula("F⁻¹(u) = a + √[ u(b−a)(c−a) ]    if  u < (c−a)/(b−a)"
            "\nF⁻¹(u) = b − √[ (1−u)(b−a)(b−c) ]  otherwise")

    left, right = cols()
    with left:
        gp_open("Parameters")
        c_mode = st.slider("Mode  c  (peak, 0–1)", 0.05, 0.95, 0.30, 0.05)
        n      = st.slider("Samples  n", 1000, 50000, 20000, 1000)
        seed   = st.slider("Seed", 1, 200, 99)
        gp_close()

    res = inverse_cdf.run(n, c_mode, seed)

    with left:
        gp_open("Statistics")
        metrics(
            (f"{res['sample_mean']:.4f}",      "Sample mean",        ""),
            (f"{res['theoretical_mean']:.4f}", "Theory (a+b+c)/3",   "bl"),
            (f"{res['mode']:.2f}",              "Mode  c",            "am"),
            (f"{n:,}",                          "Samples",            ""),
        )
        gp_close()
        ok(f"Error = {abs(res['sample_mean']-res['theoretical_mean']):.5f}.  Inverse CDF is exact — no rejection waste.")

    with right:
        fig = inverse_cdf.plot(res)
        plotly_defaults(fig, height=400)
        chart_label("Histogram of samples vs theoretical triangular PDF")
        st.plotly_chart(fig, use_container_width=True)

elif module == "Rejection Sampling":
    page_header("rej", "Rejection Sampling — Beta(α, β)",
                "Propose from Uniform[0,1].  Accept if U ≤ f(x)/(M·g(x)).  M must upper-bound f(x)/g(x) everywhere.")
    formula("Accept  x_prop  if  U  ≤  f(x) / (M · g(x))     g(x) = Uniform[0,1]")

    left, right = cols()
    with left:
        gp_open("Parameters")
        alpha   = st.slider("α  (shape 1)", 1.0, 8.0, 3.0, 0.5)
        beta_v  = st.slider("β  (shape 2)", 1.0, 8.0, 4.0, 0.5)
        n_tgt   = st.slider("Target samples", 1000, 15000, 5000, 500)
        seed    = st.slider("Seed", 1, 100, 5)
        run_btn = st.button("▶  Sample", key="rej_run")
        gp_close()

    if run_btn or "rej_res" not in st.session_state:
        with st.spinner("Sampling…"):
            st.session_state.rej_res = rejection_sampling.run(alpha, beta_v, n_tgt, seed)
    res = st.session_state.rej_res

    with left:
        gp_open("Statistics")
        rate_col = "cr" if res["accept_rate"] < 20 else ("am" if res["accept_rate"] < 40 else "")
        metrics(
            (f"{res['accept_rate']:.1f}%",      "Accept rate",         rate_col),
            (f"{res['n_proposed']:,}",            "Total proposed",      ""),
            (f"{res['sample_mean']:.4f}",         "Sample mean",         ""),
            (f"{res['theoretical_mean']:.4f}",    "Theory  α/(α+β)",     "bl"),
        )
        gp_close()
        info(f"Efficiency = {res['accept_rate']:.1f}%.  Higher α+β → sharper peak → lower rate.  Tighter M → higher rate.")

    with right:
        fig = rejection_sampling.plot(res)
        plotly_defaults(fig, height=400)
        chart_label(f"Accepted samples vs Beta({alpha}, {beta_v}) PDF")
        st.plotly_chart(fig, use_container_width=True)

elif module == "Agent-based SIR":
    page_header("sir", "Agent-based SIR Epidemic Model",
                "Each agent is S / I / R.  Every day: each S contacts a random I and infects with prob β.  Infected recover with prob γ.")
    formula("R₀ = β / γ   |   Epidemic takes off if R₀ > 1   |   Herd immunity threshold = 1 − 1/R₀")

    left, right = cols()
    with left:
        gp_open("Parameters")
        N      = st.slider("Population  N", 100, 2000, 1000, 50)
        beta_v = st.slider("Transmission  β", 0.05, 0.90, 0.30, 0.05)
        gamma  = st.slider("Recovery  γ", 0.01, 0.30, 0.05, 0.01)
        I0     = st.slider("Initial infected", 1, 20, 5)
        T      = st.slider("Days  T", 30, 200, 100, 10)
        seed   = st.slider("Seed", 1, 100, 2024)
        run_btn = st.button("▶  Simulate", key="sir_run")
        gp_close()

    if run_btn or "sir_res" not in st.session_state:
        with st.spinner("Simulating epidemic…"):
            st.session_state.sir_res = agent_based.run(N, beta_v, gamma, T, I0, seed)
    res = st.session_state.sir_res

    with left:
        gp_open("Statistics")
        r0_col = "cr" if res["R0"] > 3 else ("am" if res["R0"] > 1 else "bl")
        metrics(
            (f"{res['R0']:.2f}",              "R₀  =  β/γ",          r0_col),
            (f"{res['peak_I']}",               "Peak infected",        "cr"),
            (f"Day {res['peak_day']}",         "Day of peak",          "am"),
            (f"{res['final_R_pct']:.1f}%",     "Final recovered %",    "bl"),
        )
        gp_close()
        if res["R0"] > 1:
            warn(f"R₀ = {res['R0']:.2f} > 1 — epidemic takes off.  Herd immunity threshold ≈ <b>{res['herd_threshold']:.1f}%</b>.")
        else:
            ok(f"R₀ = {res['R0']:.2f} ≤ 1 — disease dies out without major outbreak.")

    with right:
        fig = agent_based.plot(res)
        plotly_defaults(fig, height=420)
        chart_label("S (blue) / I (red) / R (green) agents over time")
        st.plotly_chart(fig, use_container_width=True)

elif module == "Permutation Test":
    page_header("perm", "Permutation Test — Two-sample Δ Means",
                "Pool both groups, shuffle B times, recompute Δ means.  Build the null distribution and find the exact p-value.")
    formula("p-value  =  #{ |Δ*| ≥ |Δ_obs| }  /  B     (two-tailed, no normality assumption)")

    left, right = cols()
    with left:
        gp_open("Data & Parameters")
        ga_str  = st.text_input("Group A values  (comma-separated)", "12.1,11.8,13.4,10.9,12.7,11.5")
        gb_str  = st.text_input("Group B values  (comma-separated)", "14.2,13.6,15.1,14.8,13.9")
        n_perm  = st.slider("Permutations  B", 1000, 20000, 5000, 1000)
        seed    = st.slider("Seed", 1, 100, 21)
        run_btn = st.button("▶  Run test", key="perm_run")
        gp_close()

    try:
        ga = [float(v.strip()) for v in ga_str.split(",") if v.strip()]
        gb = [float(v.strip()) for v in gb_str.split(",") if v.strip()]
        assert len(ga) >= 2 and len(gb) >= 2
    except Exception:
        st.error("Enter valid comma-separated numbers in both groups (≥ 2 values each).")
        st.stop()

    if run_btn or "perm_res" not in st.session_state:
        with st.spinner("Permuting…"):
            st.session_state.perm_res = permutation.run(ga, gb, n_perm, seed)
    res = st.session_state.perm_res

    with left:
        gp_open("Test Results")
        p_col = "cr" if res["significant"] else ""
        metrics(
            (f"{res['obs_stat']:.4f}",           "Observed  Δ",     "am"),
            (f"{res['p_val']:.4f}",               "p-value",         p_col),
            (f"{np.mean(ga):.3f}",                "Mean — Group A",  ""),
            (f"{np.mean(gb):.3f}",                "Mean — Group B",  "bl"),
        )
        gp_close()
        if res["significant"]:
            ok(f"p = {res['p_val']:.4f} < 0.05 — statistically significant difference.")
        else:
            info(f"p = {res['p_val']:.4f} ≥ 0.05 — no significant difference detected.")

    with right:
        fig = permutation.plot(res)
        plotly_defaults(fig, height=400)
        chart_label("Null distribution (observed Δ in red — two-tailed)")
        st.plotly_chart(fig, use_container_width=True)

elif module == "Logistic Regression":
    page_header("logit", "Logistic Regression — Binomial GLM",
                "Generate binary data from true β via logit link.  IRLS recovers β from scratch.  See sigmoid fit + Pearson residuals + convergence.")
    formula("logit(p) = β₀ + β₁x   →   p = eᶯ/(1+eᶯ)   |   IRLS:  β_new = (XᵀWX)⁻¹ XᵀWz   |   W = p(1−p)")

    left, right = cols()
    with left:
        gp_open("True Parameters  (data generation)")
        b0      = st.slider("True  β₀", -6.0, 2.0, -3.0, 0.5)
        b1      = st.slider("True  β₁", 0.1, 4.0, 1.5, 0.1)
        n       = st.slider("Sample size  n", 20, 300, 80, 10)
        seed    = st.slider("Seed", 1, 200, 42)
        run_btn = st.button("▶  Fit model", key="lg_run")
        gp_close()

    if run_btn or "lg_res" not in st.session_state:
        st.session_state.lg_res = glm_logistic.run(b0, b1, n, seed)
    res = st.session_state.lg_res
    bh = res["beta_hat"]

    with left:
        gp_open("Fitted Statistics")
        metrics(
            (f"{bh[0]:.4f}",             "Fitted  β₀",       ""),
            (f"{bh[1]:.4f}",             "Fitted  β₁",       ""),
            (f"{res['odds_ratio']:.4f}", "Odds ratio  e^β₁", "am"),
            (f"{res['n_iter']}",          "IRLS iterations",  "bl"),
        )
        gp_close()
        info(f"e^β₁ = <b>{res['odds_ratio']:.3f}</b>: each +1 unit in X multiplies the <i>odds</i> of Y=1 by {res['odds_ratio']:.3f}.")

    with right:
        fig = glm_logistic.plot(res)
        plotly_defaults(fig, height=620)
        chart_label("Sigmoid fit · Pearson residuals · β convergence · log-likelihood")
        st.plotly_chart(fig, use_container_width=True)

elif module == "Poisson GLM":
    page_header("pois", "Poisson GLM — Count Data",
                "Generate Poisson counts from true β via log link.  IRLS recovers β.  deviance/df ≈ 1 = good fit; >> 1 = overdispersion.")
    formula("log(μ) = β₀ + β₁x   →   μ = e^(β₀+β₁x)   |   Rate ratio = e^β₁   |   W = μ")

    left, right = cols()
    with left:
        gp_open("True Parameters  (data generation)")
        b0      = st.slider("True  β₀", 0.0, 3.0, 1.0, 0.1)
        b1      = st.slider("True  β₁", 0.0, 1.5, 0.5, 0.05)
        n       = st.slider("Sample size  n", 20, 200, 80, 10)
        seed    = st.slider("Seed", 1, 200, 42)
        run_btn = st.button("▶  Fit model", key="poi_run")
        gp_close()

    if run_btn or "poi_res" not in st.session_state:
        st.session_state.poi_res = glm_poisson.run(b0, b1, n, seed)
    res = st.session_state.poi_res
    bh = res["beta_hat"]

    with left:
        gp_open("Fitted Statistics")
        phi_col = "cr" if res["phi"] > 2 else ""
        metrics(
            (f"{bh[0]:.4f}",              "Fitted  β₀",       ""),
            (f"{bh[1]:.4f}",              "Fitted  β₁",       ""),
            (f"{res['rate_ratio']:.4f}",  "Rate ratio  e^β₁", "am"),
            (f"{res['phi']:.3f}",          "Deviance / df",    phi_col),
        )
        gp_close()
        if res["phi"] > 2:
            warn(f"Deviance/df = {res['phi']:.2f} — possible overdispersion. Consider quasi-Poisson or Negative Binomial.")
        else:
            ok(f"Deviance/df = {res['phi']:.2f} ≈ 1 — good Poisson fit.")

    with right:
        fig = glm_poisson.plot(res)
        plotly_defaults(fig, height=620)
        chart_label("Data + fitted curve · Pearson residuals · β convergence · log-likelihood")
        st.plotly_chart(fig, use_container_width=True)

elif module == "IRLS Convergence":
    page_header("irls", "IRLS Convergence — Step-by-step",
                "Step through every IRLS iteration. Watch β, weights W, working response z, and log-likelihood evolve from [0,0] to convergence.")
    formula("β_new = (XᵀWX)⁻¹ XᵀWz     |     z = η + (y−μ)/W     |     W = Var(μ)  (family-specific)")

    left, right = cols()
    with left:
        gp_open("Data & Family")
        family    = st.selectbox("GLM family", ["binomial", "poisson"])
        default_y = "0,0,1,1,1" if family == "binomial" else "1,2,5,8,12"
        y_str     = st.text_input("y  values", default_y)
        x_str     = st.text_input("x  values (predictor)", "1,2,3,4,5")
        fit_btn   = st.button("▶  Fit all iterations", key="irls_run")
        gp_close()

    try:
        y_in = [float(v.strip()) for v in y_str.split(",") if v.strip()]
        x_in = [float(v.strip()) for v in x_str.split(",") if v.strip()]
        assert len(y_in) == len(x_in) and len(y_in) >= 3
    except Exception:
        st.error("y and x must be equal-length comma-separated numbers (≥ 3).")
        st.stop()

    if fit_btn or "irls_res" not in st.session_state:
        st.session_state.irls_res = glm_irls.run(y_in, x_in, family)
    res = st.session_state.irls_res
    n_iters = len(res["all_iters"])

    if n_iters == 0:
        st.warning("IRLS did not converge — try different y/x values.")
        st.stop()

    selected = st.slider("Step through iteration", 0, n_iters - 1, 0, 1, key="irls_slider")
    cur = res["all_iters"][selected]

    with left:
        gp_open(f"Values at iteration {selected}")
        metrics(
            (f"{cur['beta_new'][0]:.5f}", "β₀",              ""),
            (f"{cur['beta_new'][1]:.5f}", "β₁",              ""),
            (f"{cur['ll']:.4f}",           "Log-likelihood",  "bl"),
            (f"{cur['delta']:.2e}",        "|Δβ|  change",    "am"),
        )
        gp_close()

        with st.expander("📋 Full per-observation table"):
            rows = []
            for i, (xi, yi, wi, zi, mi) in enumerate(
                zip(res["x"], res["y"], cur["W"], cur["z"], cur["mu"])
            ):
                rows.append({"i": i+1, "xᵢ": f"{xi:.2f}", "yᵢ": f"{yi:.0f}",
                              "ηᵢ": f"{cur['eta'][i]:.4f}", "μᵢ": f"{mi:.4f}",
                              "Wᵢ": f"{wi:.4f}", "zᵢ": f"{zi:.4f}"})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        final = res["all_iters"][-1]
        ok(f"Converged in <b>{n_iters}</b> iterations.  "
           f"Final β = [{final['beta_new'][0]:.4f},  {final['beta_new'][1]:.4f}]")

    with right:
        fig = glm_irls.plot(res, selected)
        plotly_defaults(fig, height=560)
        chart_label("β path · log-likelihood · weights W · working response z")
        st.plotly_chart(fig, use_container_width=True)