import matplotlib.pyplot as plt
import seaborn as sns

def missing_values_heatmap(data):
    missing_pct = data.isnull().mean().to_frame(name="Missing %") * 100
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(missing_pct.T, annot=True, cmap="Reds", cbar=False, ax=ax)
    ax.set_title("Missing Values Percentage by Column", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig
