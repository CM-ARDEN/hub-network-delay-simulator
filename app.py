import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.data_generator import generate_synthetic_network_data
from src.model import (
    build_network_graph,
    train_propagation_model,
    simulate_unit_cascade,
)

st.set_page_config(
    page_title="Hub-and-Spoke Network Delay Simulator",
    page_icon="🕸️",
    layout="wide",
)

st.title("Hub-and-Spoke Network Delay Propagation & AI Control Panel")
st.caption(
    "Graph Theory (NetworkX) & XGBoost Decision Support Simulator on 100% Synthetic Operations Data"
)

# Sidebar controls
st.sidebar.header("Simulation Parameters")
num_units = st.sidebar.slider("Active Fleet Units", min_value=12, max_value=40, value=24, step=2)
num_nodes = st.sidebar.slider("Spoke Nodes (Stations)", min_value=6, max_value=20, value=12, step=1)
sim_days = st.sidebar.slider("Simulation Horizon (Days)", min_value=7, max_value=30, value=14, step=1)
seed = st.sidebar.number_input("Random Seed", min_value=1, max_value=9999, value=42)


@st.cache_data
def load_simulation(u: int, n: int, d: int, s: int):
    data = generate_synthetic_network_data(num_units=u, num_nodes=n, sim_days=d, seed=s)
    _, node_df = build_network_graph(data)
    ml_artifacts = train_propagation_model(data)
    return data, node_df, ml_artifacts


df, node_metrics_df, ml_results = load_simulation(num_units, num_nodes, sim_days, seed)

# Top KPI Row
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Simulated Operations", f"{len(df):,}")
k2.metric("Mean Arrival Delay", f"{df['total_arr_delay_min'].mean():.1f} min")
k3.metric("Disruption Rate (>=15 min)", f"{df['is_delayed_15'].mean() * 100:.1f}%")
k4.metric("XGBoost Propagation R²", f"{ml_results['r2_score']:.4f}")

st.divider()

tab1, tab2, tab3 = st.tabs(
    [
        "1. Unit Rotation Cascade Simulator",
        "2. Hub-and-Spoke Graph Topology",
        "3. XGBoost Feature & Shock Analytics",
    ]
)

with tab1:
    st.subheader("Single-Operation Shock Cascade Analysis")
    st.markdown(
        "Select a synthetic date, fleet unit (`UNIT-Axx`), and inject a primary delay shock to observe downstream ripple effects across chained rotations."
    )

    c1, c2, c3, c4 = st.columns(4)
    selected_date = c1.selectbox("Simulation Date", sorted(df["sim_date"].unique()))
    day_df = df[df["sim_date"] == selected_date]
    selected_unit = c2.selectbox("Fleet Unit ID", sorted(day_df["unit_id"].unique()))

    unit_day_df = day_df[day_df["unit_id"] == selected_unit].sort_values("leg_sequence")
    selected_leg = c3.selectbox(
        "Inject Shock at Leg Sequence",
        unit_day_df["leg_sequence"].tolist(),
        format_func=lambda x: f"Leg {x} ({unit_day_df[unit_day_df['leg_sequence']==x]['op_id'].values[0]})",
    )
    injected_shock = c4.slider("Injected Primary Delay (min)", min_value=0, max_value=180, value=65, step=5)

    sim_result = simulate_unit_cascade(unit_day_df, selected_leg, float(injected_shock))

    m1, m2, m3 = st.columns(3)
    m1.metric("Selected Unit", f"{selected_unit} ({unit_day_df['unit_class'].iloc[0]})")
    m2.metric("Daily Chained Legs", f"{len(sim_result)} Operations")
    m3.metric(
        "Total Downstream Delay Impact",
        f"{sim_result['sim_arr_delay'].sum():.1f} min",
        delta=f"{sim_result['sim_arr_delay'].sum() - sim_result['total_arr_delay_min'].sum():+.1f} min vs baseline",
        delta_color="inverse",
    )

    fig_cascade = go.Figure()
    fig_cascade.add_trace(
        go.Bar(
            x=sim_result["op_id"] + " (" + sim_result["origin"] + "→" + sim_result["destination"] + ")",
            y=sim_result["total_arr_delay_min"],
            name="Baseline Arrival Delay (min)",
            marker_color="#636EFA",
        )
    )
    fig_cascade.add_trace(
        go.Bar(
            x=sim_result["op_id"] + " (" + sim_result["origin"] + "→" + sim_result["destination"] + ")",
            y=sim_result["sim_arr_delay"],
            name="Simulated Shock Delay (min)",
            marker_color="#EF553B",
        )
    )
    fig_cascade.update_layout(
        barmode="group",
        title=f"Baseline vs. Simulated Delay Propagation for {selected_unit} on {selected_date}",
        xaxis_title="Operation Sequence & Corridor",
        yaxis_title="Arrival Delay (Minutes)",
        height=400,
    )
    st.plotly_chart(fig_cascade, use_container_width=True)

    st.dataframe(
        sim_result[
            [
                "leg_sequence",
                "op_id",
                "origin",
                "destination",
                "scheduled_buffer_min",
                "absorbable_slack_min",
                "total_arr_delay_min",
                "sim_arr_delay",
                "sim_propagated_out",
            ]
        ],
        use_container_width=True,
    )

with tab2:
    st.subheader("Hub-and-Spoke Centrality & Bottleneck Metrics")
    col_a, col_b = st.columns([1.2, 1])

    with col_a:
        fig_nodes = px.bar(
            node_metrics_df,
            x="node_id",
            y="avg_node_delay_min",
            color="disruption_rate_pct",
            title="Average Arrival Delay & Disruption Rate by Network Node",
            labels={
                "node_id": "Network Node",
                "avg_node_delay_min": "Avg Arrival Delay (min)",
                "disruption_rate_pct": "Disruption Rate (%)",
            },
            color_continuous_scale="YlOrRd",
        )
        st.plotly_chart(fig_nodes, use_container_width=True)

    with col_b:
        st.markdown("**Node Centrality Table (`HUB-ALPHA` & `NODE-01..N`)**")
        st.dataframe(node_metrics_df, use_container_width=True, height=380)

with tab3:
    st.subheader("XGBoost Propagation Predictor Diagnostics")
    c_left, c_right = st.columns(2)

    with c_left:
        fig_imp = px.bar(
            ml_results["feature_importance"],
            x="importance",
            y="feature",
            orientation="h",
            title=f"Feature Importance (Test MAE: {ml_results['mae_min']} min | R²: {ml_results['r2_score']})",
            color_discrete_sequence=["#00CC96"],
        )
        fig_imp.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_imp, use_container_width=True)

    with c_right:
        cause_df = (
            df[df["disruption_cause"] != "NONE"]
            .groupby("disruption_cause")
            .agg(
                event_count=("op_id", "count"),
                mean_shock_min=("primary_delay_min", "mean"),
            )
            .reset_index()
        )
        fig_cause = px.pie(
            cause_df,
            names="disruption_cause",
            values="event_count",
            title="Synthetic Primary Shock Distribution by Category",
            hole=0.4,
        )
        st.plotly_chart(fig_cause, use_container_width=True)