import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


def build_network_graph(df: pd.DataFrame) -> tuple[nx.DiGraph, pd.DataFrame]:
    """
    Constructs a directed weighted network graph of stations and computes
    node centrality and congestion metrics.
    """
    G = nx.DiGraph()

    edge_stats = (
        df.groupby(["origin", "destination"])
        .agg(
            op_count=("op_id", "count"),
            avg_arr_delay=("total_arr_delay_min", "mean"),
            delayed_rate=("is_delayed_15", "mean"),
        )
        .reset_index()
    )

    for _, row in edge_stats.iterrows():
        G.add_edge(
            row["origin"],
            row["destination"],
            weight=row["op_count"],
            avg_delay=round(row["avg_arr_delay"], 2),
            delayed_rate=round(row["delayed_rate"] * 100, 1),
        )

    betweenness = nx.betweenness_centrality(G, weight="weight")
    in_degree = dict(G.in_degree(weight="weight"))
    out_degree = dict(G.out_degree(weight="weight"))

    node_rows = []
    for node in G.nodes():
        node_df = df[(df["origin"] == node) | (df["destination"] == node)]
        node_rows.append(
            {
                "node_id": node,
                "in_flow": in_degree.get(node, 0),
                "out_flow": out_degree.get(node, 0),
                "betweenness_centrality": round(betweenness.get(node, 0.0), 4),
                "avg_node_delay_min": round(node_df["total_arr_delay_min"].mean(), 2),
                "disruption_rate_pct": round(node_df["is_delayed_15"].mean() * 100, 1),
            }
        )

    node_metrics_df = pd.DataFrame(node_rows).sort_values(
        by="in_flow", ascending=False
    ).reset_index(drop=True)

    return G, node_metrics_df


def train_propagation_model(df: pd.DataFrame) -> dict:
    """
    Trains an XGBoost regressor to predict downstream propagated delay
    given current operation state, slack buffer, and primary shock.
    """
    feature_cols = [
        "leg_sequence",
        "min_turnaround_min",
        "scheduled_buffer_min",
        "absorbable_slack_min",
        "primary_delay_min",
        "propagated_delay_in_min",
        "total_dep_delay_min",
    ]

    X = pd.get_dummies(
        df[feature_cols + ["unit_class"]],
        columns=["unit_class"],
        drop_first=False,
    )
    y = df["propagated_delay_out_min"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBRegressor(
        n_estimators=120,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.9,
        random_state=42,
    )
    model.fit(X_train, y_train)

    preds = np.maximum(0.0, model.predict(X_test))
    r2 = round(float(r2_score(y_test, preds)), 4)
    mae = round(float(mean_absolute_error(y_test, preds)), 2)

    importances = pd.DataFrame(
        {"feature": X.columns, "importance": model.feature_importances_}
    ).sort_values(by="importance", ascending=False)

    return {
        "model": model,
        "feature_columns": list(X.columns),
        "r2_score": r2,
        "mae_min": mae,
        "feature_importance": importances,
    }


def simulate_unit_cascade(
    unit_day_df: pd.DataFrame,
    target_leg_seq: int,
    injected_shock_min: float,
) -> pd.DataFrame:
    """
    Re-simulates a single unit's daily chain when an artificial primary
    delay shock is injected at a chosen leg sequence.
    """
    sim_df = unit_day_df.sort_values("leg_sequence").copy().reset_index(drop=True)
    carried_delay = 0.0

    for idx, row in sim_df.iterrows():
        seq = row["leg_sequence"]
        if seq < target_leg_seq:
            carried_delay = row["propagated_delay_out_min"]
            sim_df.loc[idx, "sim_dep_delay"] = row["total_dep_delay_min"]
            sim_df.loc[idx, "sim_arr_delay"] = row["total_arr_delay_min"]
            sim_df.loc[idx, "sim_propagated_out"] = row["propagated_delay_out_min"]
        else:
            shock = injected_shock_min if seq == target_leg_seq else row["primary_delay_min"]
            dep_delay = round(carried_delay + shock, 1)
            arr_delay = max(0.0, round(dep_delay - 2.0, 1)) if dep_delay > 0 else 0.0
            prop_out = max(0.0, round(arr_delay - row["absorbable_slack_min"], 1))

            sim_df.loc[idx, "sim_dep_delay"] = dep_delay
            sim_df.loc[idx, "sim_arr_delay"] = arr_delay
            sim_df.loc[idx, "sim_propagated_out"] = prop_out
            carried_delay = prop_out

    return sim_df