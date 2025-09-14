import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore

# --- Detect anomalies using Z-score > 3 ---
def detect_anomalies(data):
    numeric_cols = data.select_dtypes(include="number").columns
    anomalies = pd.DataFrame()
    
    for col in numeric_cols:
        col_data = data[col]
        z_scores = (col_data - col_data.mean()) / col_data.std(ddof=0)
        mask = abs(z_scores) > 3
        mask = mask.fillna(False)  # align with data length
        outliers = data.loc[mask, [col]].copy()
        if not outliers.empty:
            outliers["Anomaly_Column"] = col
            anomalies = pd.concat([anomalies, outliers], ignore_index=True)
    
    return anomalies

# --- Scatter plot for anomalies ---
def anomaly_scatter_plot(data, x_col, y_col, anomalies):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(data[x_col], data[y_col], label="Data", alpha=0.6)
    
    if not anomalies.empty and x_col in data.columns and y_col in data.columns:
        mask = anomalies[x_col].notna() & anomalies[y_col].notna()
        ax.scatter(anomalies.loc[mask, x_col], anomalies.loc[mask, y_col],
                   color="red", label="Anomalies", alpha=0.8)
        ax.set_title(f"Scatter Plot with Anomalies ({x_col} vs {y_col})")
        ax.legend()
    else:
        ax.set_title(f"Scatter Plot ({x_col} vs {y_col})")
    
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    return fig

# --- Box plot for anomalies ---
def anomaly_box_plot(data, col, anomalies):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(data[col].dropna(), vert=False)
    
    if not anomalies.empty and col in anomalies["Anomaly_Column"].values:
        anomaly_values = anomalies[anomalies["Anomaly_Column"] == col][col]
        ax.scatter(anomaly_values, [1]*len(anomaly_values), color="red", label="Anomalies")
        ax.set_title(f"Box Plot with Anomalies ({col})")
        ax.legend()
    else:
        ax.set_title(f"Box Plot ({col})")
    
    return fig
