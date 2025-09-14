import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def missing_values_heatmap(data):
    # Calculate percentage of missing values per column
    missing_percent = (data.isnull().sum() / len(data)) * 100
    missing_df = pd.DataFrame(missing_percent, columns=["Missing (%)"])

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        missing_df.T,  # transpose so columns appear horizontally
        annot=True, fmt=".1f", cmap="Reds", cbar_kws={"label": "% Missing"},
        ax=ax
    )
    ax.set_title("Percentage of Missing Values per Column")
    return fig

def data_quality_summary(data):
    """Return duplicated rows count and schema."""
    duplicated_count = data.duplicated().sum()
    schema = pd.DataFrame({
        "Column": data.columns,
        "DataType": [str(dtype) for dtype in data.dtypes],
        "Non-Null Count": data.notnull().sum()
    })
    return duplicated_count, schema