# Hub-and-Spoke Network Delay Propagation & AI Control Panel

An interactive decision-support and network simulation platform built with **Python**, **NetworkX**, **XGBoost**, and **Streamlit**. The system models cascading delay propagation across hub-and-spoke operational networks using 100% synthetic data.

## Key Capabilities
* **Stochastic Synthetic Data Engine (`src/data_generator.py`):** Simulates multi-day fleet rotations (`HUB-ALPHA` to `NODE-01..N`) with Poisson-triggered primary shocks and Weibull recovery distributions.
* **Graph Topology & Bottleneck Analytics (`src/model.py`):** Constructs directed weighted network graphs via `NetworkX` to compute node in/out flows, betweenness centrality, and disruption rates.
* **Machine Learning Propagation Predictor:** Trains an `XGBoost` regressor ($R^2 \approx 0.99$) to predict downstream propagated delays based on scheduled turnaround buffers and absorbable slack.
* **Interactive Shock Cascade Simulator (`app.py`):** Allows operators to inject artificial delay shocks at any leg sequence and visualize real-time ripple effects across subsequent operations.

## Tech Stack
* **Frontend & Visualization:** Streamlit, Plotly
* **Data & Graph Modeling:** Pandas, NumPy, NetworkX
* **Machine Learning:** XGBoost, Scikit-Learn

## Quick Start

1. Install dependencies:
   pip install -r requirements.txt

2. Launch the simulation dashboard:
   streamlit run app.py