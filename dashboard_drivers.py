"""Alex Optimization Dashboard

Chart 1: Stacked horizontal bars by pathway (phone number) — % of calls by outcome.
Chart 2: Weekly True Resolution Rate — validated by 48h callback analysis.
Chart 3: Week Fixed Effects — LPM regression controlling for component mix.
Filters: date range, caller type, component, component:symptom.

Data: data/calls_drivers.csv (with callback_48h and true_resolution columns)
Usage: streamlit run dashboard_drivers.py
"""

from datetime import timedelta

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DRIVERS_PATH = PROJECT_ROOT / "data" / "calls_drivers.csv"

# Pathway → phone number mapping
PATHWAY_PHONES = {
    "Applicant Pathway (Original)": "+14153246039",
    "Full Support Pathway": "+17144521864",
    "Julian Applicant Pathway": "+16504139279",
}

# Pathways to exclude
EXCLUDE_PATHWAYS = {"Spanish Pathway"}

# Test phone number suffixes to exclude
TEST_PHONE_SUFFIXES = ("8787", "9121")

# Resolution colors — order matters for stacking
RES_ORDER = ["resolved", "partially_resolved", "transferred", "abandoned", "unresolved", "no_interaction"]
RES_COLORS = {
    "resolved": "#27ae60",
    "partially_resolved": "#82e0aa",
    "transferred": "#e74c3c",
    "abandoned": "#f39c12",
    "unresolved": "#e67e22",
    "no_interaction": "#bdc3c7",
}
RES_LABELS = {
    "resolved": "Resolved",
    "partially_resolved": "Partially Resolved",
    "transferred": "Transferred",
    "abandoned": "Abandoned",
    "unresolved": "Unresolved",
    "no_interaction": "No Interaction",
}


def pct(n, d):
    return round(n / d * 100, 1) if d else 0.0


@st.cache_data(ttl=timedelta(minutes=15))
def load_data():
    df = pd.read_csv(DRIVERS_PATH, encoding="utf-8")
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True, format="mixed")
    _local = df["created_at"].dt.tz_localize(None)
    df["date"] = _local.dt.normalize()
    df["week"] = df["date"] - pd.to_timedelta(_local.dt.dayofweek, unit="D")
    # Exclude test phone numbers (strip .0 suffix from float-stored phones)
    phone_col = df["caller_phone"].fillna("").astype(str).str.replace(r"\.0$", "", regex=True)
    is_test = phone_col.str.match(r".*(" + "|".join(TEST_PHONE_SUFFIXES) + r")$")
    df = df[~is_test].copy()
    df["call_length_min"] = pd.to_numeric(df["call_length_min"], errors="coerce").fillna(0)
    df["resolution"] = df["resolution"].fillna("no_interaction").str.strip().str.lower()
    df["callback_48h"] = df["callback_48h"].fillna(False).astype(bool)
    df["true_resolution"] = df["true_resolution"].fillna(False).astype(bool)
    # Map pathway names to phone numbers
    df["phone"] = df["pathway_name"].map(PATHWAY_PHONES).fillna("Unknown")
    df["bar_label"] = df.apply(
        lambda r: f"{PATHWAY_PHONES.get(r['pathway_name'], 'Unknown')}  ({r['pathway_name']})"
        if pd.notna(r['pathway_name']) else "Unknown", axis=1)
    # Build component:symptom field
    df["comp_symptom"] = df["component"].fillna("") + ": " + df["symptom_category"].fillna("")
    return df


def meaningful(df):
    """Filter to non-noise calls (exclude no_interaction and call routing)."""
    return df[~df["resolution"].isin(["no_interaction"]) & (df["component"] != "call routing")]


def main():
    st.set_page_config(page_title="Alex Optimization", layout="wide")
    st.title("Alex Optimization Dashboard")

    if not DRIVERS_PATH.exists():
        st.error("Missing data/calls_drivers.csv")
        return

    df = load_data()
    if df.empty:
        st.error("No data.")
        return

    latest = df["created_at"].max()
    if pd.notna(latest):
        st.caption(f"Data through **{latest.strftime('%B %d, %Y')}** · {len(df):,} calls · refreshes every 15 min")

    # ── Filters ─────────────────────────────────────────────────────────────
    f1, f2, f3, f4 = st.columns([1, 1, 1, 1])

    with f1:
        d_min, d_max = df["date"].min(), df["date"].max()
        d_range = st.date_input("Date range", value=(d_min, d_max),
                                min_value=d_min, max_value=d_max)

    with f2:
        caller_types = sorted(df["caller_type"].dropna().unique())
        sel_caller = st.multiselect("Caller type", caller_types, default=[],
                                    placeholder="All caller types")

    with f3:
        components = sorted(df["component"].dropna().unique())
        sel_comp = st.multiselect("Component", components, default=[],
                                  placeholder="All components")

    with f4:
        if sel_comp:
            symp_opts = sorted(df[df["component"].isin(sel_comp)]["comp_symptom"].unique())
        else:
            symp_opts = sorted(df["comp_symptom"].unique())
        sel_symptom = st.multiselect("Component: Symptom", symp_opts, default=[],
                                     placeholder="All symptoms")

    # Apply filters
    f = df[~df["pathway_name"].isin(EXCLUDE_PATHWAYS)].copy()
    if len(d_range) == 2:
        f = f[(f["date"] >= pd.Timestamp(d_range[0])) & (f["date"] <= pd.Timestamp(d_range[1]))]
    if sel_caller:
        f = f[f["caller_type"].isin(sel_caller)]
    if sel_comp:
        f = f[f["component"].isin(sel_comp)]
    if sel_symptom:
        f = f[f["comp_symptom"].isin(sel_symptom)]

    if f.empty:
        st.warning("No calls match filters.")
        return

    # ════════════════════════════════════════════════════════════════════════
    # CHART 1: OUTCOMES BY PATHWAY
    # ════════════════════════════════════════════════════════════════════════
    st.header("Outcomes by Pathway")

    counts = f.groupby(["bar_label", "resolution"]).size().reset_index(name="count")
    totals = counts.groupby("bar_label")["count"].sum()
    counts["pct"] = counts.apply(
        lambda r: round(r["count"] / totals[r["bar_label"]] * 100, 1), axis=1)

    bar_labels = totals.sort_values(ascending=True).index.tolist()

    fig = go.Figure()
    for res in RES_ORDER:
        res_data = counts[counts["resolution"] == res].set_index("bar_label")
        pcts = [res_data.loc[bl, "pct"] if bl in res_data.index else 0 for bl in bar_labels]
        raw = [int(res_data.loc[bl, "count"]) if bl in res_data.index else 0 for bl in bar_labels]
        if sum(raw) == 0:
            continue
        fig.add_trace(go.Bar(
            y=bar_labels, x=pcts,
            name=RES_LABELS[res], orientation="h",
            marker_color=RES_COLORS[res], customdata=raw,
            hovertemplate="%{y}<br>" + RES_LABELS[res] + ": %{x:.1f}% (%{customdata} calls)<extra></extra>",
        ))

    for bl in bar_labels:
        fig.add_annotation(
            x=102, y=bl,
            text=f"<b>n={int(totals[bl])}</b>",
            showarrow=False, font=dict(size=12), xanchor="left",
            bgcolor="#f0f0f0", bordercolor="#ccc", borderwidth=1, borderpad=3,
        )

    fig.update_layout(
        barmode="stack",
        height=max(250, len(bar_labels) * 80),
        margin=dict(l=0, r=80, t=10, b=40),
        xaxis_title="% of Calls",
        xaxis=dict(range=[0, 115], ticksuffix="%", dtick=20),
        yaxis=dict(automargin=True, tickfont=dict(size=13)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="y unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Summary line
    total = len(f)
    n_resolved = len(f[f["resolution"].isin(["resolved", "partially_resolved"])])
    n_transferred = len(f[f["resolution"] == "transferred"])
    st.caption(
        f"{total} calls shown · "
        f"{n_resolved} resolved ({pct(n_resolved, total)}%) · "
        f"{n_transferred} transferred ({pct(n_transferred, total)}%)"
    )

    # ════════════════════════════════════════════════════════════════════════
    # CHART 2: TRUE RESOLUTION RATE OVER TIME
    # ════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.header("Containment vs True Resolution")
    st.caption(
        "**Containment** = didn't transfer to a human. "
        "**True Resolution** = didn't transfer AND caller didn't call back within 48h. "
        "The gap = calls the bot 'contained' but didn't actually resolve."
    )

    m = meaningful(f)
    if m.empty:
        st.info("No meaningful calls in selection.")
        return

    # Overall KPIs
    k1, k2, k3 = st.columns(3)
    n_m = len(m)
    n_contained = len(m[~m["resolution"].isin(["transferred"])])
    n_true = len(m[m["true_resolution"]])
    n_callback = len(m[m["callback_48h"]])

    k1.metric("Containment Rate", f"{pct(n_contained, n_m)}%",
              help="% of calls that didn't transfer to a human (includes abandoned, unresolved)")
    k2.metric("True Resolution Rate", f"{pct(n_true, n_m)}%",
              help="% resolved with no callback within 48h")
    k3.metric("Gap", f"{pct(n_contained, n_m) - pct(n_true, n_m):.1f}pp",
              help="Contained but not truly resolved — callers who didn't transfer but called back or abandoned")

    # Granularity and pathway filters for trend chart
    tc1, tc2 = st.columns(2)
    with tc1:
        granularity = st.selectbox("Granularity", ["Week", "Day", "Month"],
                                   index=0, key="trend_granularity")
    with tc2:
        trend_pws = sorted(m["pathway_name"].dropna().unique())
        trend_pw = st.selectbox("Pathway", ["All"] + trend_pws,
                                index=0, key="trend_pathway")

    trend_m = m.copy()
    if trend_pw != "All":
        trend_m = trend_m[trend_m["pathway_name"] == trend_pw]

    # Group by selected granularity
    if granularity == "Day":
        trend_m["_period"] = trend_m["created_at"].dt.date
        min_period_calls = 1
    elif granularity == "Month":
        trend_m["_period"] = trend_m["created_at"].dt.tz_localize(None).dt.to_period("M").apply(
            lambda r: r.start_time.date())
        min_period_calls = 5
    else:  # Week
        trend_m["_period"] = trend_m["week"]
        min_period_calls = 5

    grouped = trend_m.groupby("_period").apply(lambda g: pd.Series({
        "calls": len(g),
        "containment": pct(len(g[~g["resolution"].isin(["transferred"])]), len(g)),
        "true_rate": pct(len(g[g["true_resolution"]]), len(g)),
    })).reset_index()
    grouped.columns = ["period", "calls", "containment", "true_rate"]
    grouped["period"] = pd.to_datetime(grouped["period"])
    grouped = grouped[grouped["calls"] >= min_period_calls]

    if not grouped.empty:
        fig_t = go.Figure()

        # Volume bars (background)
        fig_t.add_trace(go.Bar(
            x=grouped["period"], y=grouped["calls"], name="Call Volume",
            marker_color="#dfe6e9", yaxis="y2", opacity=0.3,
        ))

        if granularity == "Day":
            # Line chart for daily
            fig_t.add_trace(go.Scatter(
                x=grouped["period"], y=grouped["containment"],
                name="Containment Rate",
                line=dict(color="#3498db", width=2),
                mode="lines+markers",
            ))
            fig_t.add_trace(go.Scatter(
                x=grouped["period"], y=grouped["true_rate"],
                name="True Resolution Rate",
                line=dict(color="#27ae60", width=3),
                mode="lines+markers",
            ))
            # Shade the gap
            fig_t.add_trace(go.Scatter(
                x=list(grouped["period"]) + list(grouped["period"])[::-1],
                y=list(grouped["containment"]) + list(grouped["true_rate"])[::-1],
                fill="toself",
                fillcolor="rgba(231, 76, 60, 0.15)",
                line=dict(width=0),
                name="Gap (contained but not resolved)",
                hoverinfo="skip",
            ))
        else:
            # Bar chart for week and month
            fig_t.add_trace(go.Bar(
                x=grouped["period"], y=grouped["containment"],
                name="Containment Rate", marker_color="#3498db",
            ))
            fig_t.add_trace(go.Bar(
                x=grouped["period"], y=grouped["true_rate"],
                name="True Resolution Rate", marker_color="#27ae60",
            ))
            fig_t.update_layout(barmode="group")

        pw_label = f" — {trend_pw}" if trend_pw != "All" else ""
        period_unit = granularity[0].lower()
        fig_t.update_layout(
            height=400,
            yaxis=dict(title="Rate (%)", range=[0, 100]),
            yaxis2=dict(title=f"Calls/{period_unit}", overlaying="y", side="right",
                        showgrid=False, range=[0, grouped["calls"].max() * 2.5]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            hovermode="x unified",
            margin=dict(t=10, b=40),
        )
        st.plotly_chart(fig_t, use_container_width=True)

        st.caption(
            f"{granularity}s with <{min_period_calls} calls hidden. "
            "Blue = didn't transfer. Green = didn't transfer AND no callback within 48h. "
            "Red shaded gap = contained but caller called back — not truly resolved."
        )
    else:
        st.info(f"No {granularity.lower()}s with {min_period_calls}+ calls for the selected filters.")

    # ════════════════════════════════════════════════════════════════════════
    # CHART 3: WEEK FIXED EFFECTS (COMPONENT-CONTROLLED)
    # ════════════════════════════════════════════════════════════════════════
    chart_week_fes(f)


def chart_week_fes(f):
    """Chart 3: Week Fixed Effects from LPM controlling for component mix."""
    import statsmodels.formula.api as smf

    st.markdown("---")
    st.header("Week Fixed Effects (Component-Controlled)")
    st.caption(
        "Linear probability model: **Resolved or Partially Resolved ~ Week FEs + Component FEs**. "
        "Isolates whether the bot is improving over time after controlling for the mix of call types each week. "
        "Points show the estimated week effect (vs. reference week); error bars are 95% CIs with robust SEs."
    )

    # Chart-level filters: pathway, caller type, FE granularity, time granularity
    fc1, fc2, fc3, fc4 = st.columns(4)
    with fc1:
        fe_pathways = sorted(f["pathway_name"].dropna().unique())
        fe_pw = st.selectbox("Pathway", ["All"] + fe_pathways, index=0, key="fe_pathway")
    with fc2:
        fe_caller = st.selectbox("Condition on caller type", ["Both", "applicant", "operator"],
                                 index=0, key="fe_caller_type")
    with fc3:
        fe_control = st.selectbox("Control level", ["Component", "Component:Symptom"],
                                  index=0, key="fe_control_level")
    with fc4:
        fe_time = st.selectbox("Time granularity", ["Week", "Month"],
                               index=0, key="fe_time_granularity")

    fe_data = f.copy()
    if fe_pw != "All":
        fe_data = fe_data[fe_data["pathway_name"] == fe_pw]
    if fe_caller != "Both":
        fe_data = fe_data[fe_data["caller_type"] == fe_caller]

    # Filter to meaningful calls with resolved DV
    m = fe_data[~fe_data["resolution"].isin(["no_interaction"]) & (fe_data["component"] != "call routing")].copy()
    m["resolved"] = m["resolution"].isin(["resolved", "partially_resolved"]).astype(int)

    # Build time period label based on selected granularity
    if fe_time == "Month":
        m["period_label"] = m["created_at"].dt.strftime("%Y-%m")
        time_label = "month"
        min_period_obs = 5
    else:
        m["period_label"] = m["week"].dt.strftime("%m-%d")
        time_label = "week"
        min_period_obs = 3

    # Build the control variable based on selected granularity
    if fe_control == "Component:Symptom":
        m["fe_var"] = m["component"].fillna("") + ":" + m["symptom_category"].fillna("")
        fe_label = "component:symptom"
        min_fe_obs = 3
    else:
        m["fe_var"] = m["component"]
        fe_label = "component"
        min_fe_obs = 5

    # Drop periods and FE groups with too few obs for stable estimation
    period_counts = m["period_label"].value_counts()
    m = m[m["period_label"].isin(period_counts[period_counts >= min_period_obs].index)].copy()
    fe_counts = m["fe_var"].value_counts()
    m = m[m["fe_var"].isin(fe_counts[fe_counts >= min_fe_obs].index)].copy()

    periods_sorted = sorted(m["period_label"].unique())
    n_fe_levels = m["fe_var"].nunique()

    if len(periods_sorted) < 2 or n_fe_levels < 2:
        st.info(f"Not enough data for regression (need 2+ {time_label}s and 2+ control groups with sufficient obs).")
        return

    # Run LPM with HC1 robust SEs
    model = smf.ols("resolved ~ C(period_label) + C(fe_var)", data=m).fit(cov_type="HC1")

    # Extract period FE coefficients + CIs
    ref_period = periods_sorted[0]
    period_coefs = [0.0]
    period_ci_lo = [0.0]
    period_ci_hi = [0.0]
    period_pvals = [np.nan]
    period_ns = [int(m[m["period_label"] == ref_period].shape[0])]

    ci = model.conf_int()
    for per in periods_sorted[1:]:
        param = [p for p in model.params.index if per in p][0]
        period_coefs.append(model.params[param])
        period_ci_lo.append(ci.loc[param, 0])
        period_ci_hi.append(ci.loc[param, 1])
        period_pvals.append(model.pvalues[param])
        period_ns.append(int(m[m["period_label"] == per].shape[0]))

    period_coefs = np.array(period_coefs)
    period_ci_lo = np.array(period_ci_lo)
    period_ci_hi = np.array(period_ci_hi)

    # Joint F-test on all period FEs
    period_param_idx = [i for i, p in enumerate(model.params.index) if "period_label" in p]
    R = np.zeros((len(period_param_idx), len(model.params)))
    for j, idx in enumerate(period_param_idx):
        R[j, idx] = 1
    f_test = model.f_test(R)
    f_val = float(np.squeeze(f_test.fvalue))
    f_pval = float(np.squeeze(f_test.pvalue))

    # KPIs
    k1, k2, k3 = st.columns(3)
    k1.metric("N (meaningful calls)", f"{int(model.nobs)}")
    k2.metric("R-squared", f"{model.rsquared:.3f}")
    k3.metric(f"Joint F-test ({time_label}s)", f"F={f_val:.2f}, p={f_pval:.3f}",
              help=f"Tests whether all {time_label} effects are jointly zero. p > 0.05 means no significant time trend.")

    # Build plotly figure
    if fe_time == "Month":
        x_dates = pd.to_datetime([f"{per}-01" for per in periods_sorted])
    else:
        x_dates = pd.to_datetime([f"2026-{per}" for per in periods_sorted])
    labels = [f"{per}<br>n={n}" for per, n in zip(periods_sorted, period_ns)]

    fig = go.Figure()

    # CI shading
    fig.add_trace(go.Scatter(
        x=list(x_dates) + list(x_dates)[::-1],
        y=list(period_ci_hi) + list(period_ci_lo)[::-1],
        fill="toself",
        fillcolor="rgba(37, 99, 235, 0.12)",
        line=dict(width=0),
        name="95% CI",
        hoverinfo="skip",
    ))

    # Point estimates
    colors = ["#6b7280" if i == 0 else "#2563eb" for i in range(len(periods_sorted))]
    fig.add_trace(go.Scatter(
        x=x_dates,
        y=period_coefs,
        mode="markers+lines",
        marker=dict(size=10, color=colors, line=dict(width=1, color="white")),
        line=dict(color="#2563eb", width=1.5, dash="dot"),
        name=f"{time_label.title()} FE",
        customdata=list(zip(periods_sorted, period_ns,
                            [f"{c:+.3f}" for c in period_coefs],
                            [f"{p:.3f}" if not np.isnan(p) else "ref" for p in period_pvals])),
        hovertemplate=f"{time_label.title()} " + "%{customdata[0]}<br>n=%{customdata[1]}<br>"
                      "Coef: %{customdata[2]}<br>p-value: %{customdata[3]}<extra></extra>",
    ))

    # Zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

    # Reference period annotation
    fig.add_annotation(
        x=x_dates[0], y=0.02,
        text="ref", showarrow=False,
        font=dict(size=10, color="#6b7280"),
    )

    fig.update_layout(
        height=400,
        yaxis=dict(title="Effect on Resolution Probability", zeroline=True),
        xaxis=dict(
            tickvals=x_dates,
            ticktext=labels,
            tickangle=0,
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(t=10, b=60),
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)

    sig_note = "not statistically significant" if f_pval > 0.05 else "statistically significant"
    st.caption(
        f"LPM with {n_fe_levels} {fe_label} FEs and {len(periods_sorted)} {time_label} FEs. "
        f"Robust (HC1) standard errors. Reference {time_label}: {ref_period}. "
        f"Joint F-test on {time_label} FEs is **{sig_note}** (p={f_pval:.3f}) — "
        + ("the bot's performance has not meaningfully changed over time after controlling for call mix."
           if f_pval > 0.05 else
           "there is a significant time trend in resolution rates beyond what component mix explains.")
    )


if __name__ == "__main__":
    main()
