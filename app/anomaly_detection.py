import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore

# Detect anomalies using Z-Score
def detect_anomalies(df, threshold=3):
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return pd.DataFrame()  # No numeric data

    z_scores = np.abs(zscore(numeric_df, nan_policy="omit"))
    anomalies = numeric_df[(z_scores > threshold).any(axis=1)]
    return anomalies

# Scatter plot with anomalies highlighted
def anomaly_scatter_plot(df, x_col, y_col, anomalies):
    fig, ax = plt.subplots(figsize=(6, 4))

    # Ensure both are numeric
    if not (pd.api.types.is_numeric_dtype(df[x_col]) and pd.api.types.is_numeric_dtype(df[y_col])):
        ax.text(0.5, 0.5, f"No numeric data in '{x_col}' or '{y_col}'", 
                ha="center", va="center", fontsize=12)
        return fig

    ax.scatter(df[x_col], df[y_col], label="Normal", alpha=0.6)
    
    if not anomalies.empty and x_col in anomalies.columns and y_col in anomalies.columns:
        ax.scatter(anomalies[x_col], anomalies[y_col], color="red", label="Anomaly")
    
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"Scatter Plot: {x_col} vs {y_col}")
    ax.legend()
    return fig

# Box plot with anomalies highlighted
def anomaly_box_plot(df, column, anomalies):
    fig, ax = plt.subplots(figsize=(6, 4))

    # Ensure column is numeric
    if not pd.api.types.is_numeric_dtype(df[column]):
        ax.text(0.5, 0.5, f"No numeric data in '{column}'", 
                ha="center", va="center", fontsize=12)
        return fig

    # Plot boxplot
    sns.boxplot(y=df[column], ax=ax, color="lightblue")

    # Overlay anomalies if present
    if not anomalies.empty and column in anomalies.columns:
        ax.scatter([0]*len(anomalies), anomalies[column], color="red", label="Anomaly")
        ax.legend()

    ax.set_title(f"Box Plot for {column}")
    return fig
