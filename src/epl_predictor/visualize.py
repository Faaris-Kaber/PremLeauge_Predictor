"""
visualization module

generates plots for model analysis and saves them to figures/.

usage:
    python -m epl_predictor.visualize
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from epl_predictor import config


# plot style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12


def ensure_figures_dir() -> Path:
    """create figures directory if needed."""
    figures_dir = config.PROJECT_ROOT / 'figures'
    figures_dir.mkdir(exist_ok=True)
    return figures_dir


def plot_class_distribution(df: pd.DataFrame, save_path: Path) -> None:
    """bar chart of match outcome distribution."""
    print("Creating class distribution plot...")
    
    # count outcomes
    counts = df['FTR'].value_counts()
    labels = {'H': 'Home Win', 'D': 'Draw', 'A': 'Away Win'}
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = ['#2ecc71', '#f39c12', '#e74c3c']  # Green, Orange, Red
    bars = ax.bar(
        [labels.get(x, x) for x in counts.index], 
        counts.values,
        color=colors,
        edgecolor='black',
        linewidth=1.2
    )
    
    # add count labels on bars
    for bar, count in zip(bars, counts.values):
        height = bar.get_height()
        ax.annotate(f'{count}\n({count/len(df)*100:.1f}%)',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontweight='bold', fontsize=11)
    
    ax.set_xlabel('Match Outcome', fontsize=12)
    ax.set_ylabel('Number of Matches', fontsize=12)
    ax.set_title('Premier League Match Outcome Distribution\n(2020-2025 Seasons)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_confusion_matrix(save_path: Path) -> None:
    """confusion matrix heatmap from model predictions."""
    print("Creating confusion matrix plot...")
    
    # load data and model for predictions
    from epl_predictor.models.train_model import EPLPredictor
    
    predictor = EPLPredictor()
    df = predictor.load_features()
    
    if df is None:
        print("  Error: Could not load features. Skipping confusion matrix.")
        return
    
    X, y, dates = predictor.prepare_data(df)
    X_train, X_val, X_test, y_train, y_val, y_test = predictor.split_chronologically(X, y, dates)
    
    # load trained model
    import torch
    from epl_predictor.models.train_model import EPLNet
    
    model_path = config.PROJECT_ROOT / 'saved_models' / 'epl_predictor.pth'
    if not model_path.exists():
        print("  Error: No trained model found. Run train_model.py first.")
        return
    
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model = EPLNet(input_dim=len(predictor.feature_columns))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    scaler = checkpoint['scaler']
    
    # get predictions
    X_test_scaled = scaler.transform(X_test)
    with torch.no_grad():
        outputs = model(torch.FloatTensor(X_test_scaled))
        predictions = outputs.argmax(dim=1).numpy()
    
    # build confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, predictions)
    
    # normalize
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    labels = ['Home Win', 'Draw', 'Away Win']
    
    # plot heatmap
    im = ax.imshow(cm_normalized, cmap='Blues', aspect='auto')
    
    # colorbar
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Proportion', fontsize=11)
    
    # set ticks
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticklabels(labels, fontsize=11)
    
    # text annotations
    for i in range(len(labels)):
        for j in range(len(labels)):
            count = cm[i, j]
            pct = cm_normalized[i, j]
            text_color = 'white' if pct > 0.5 else 'black'
            ax.text(j, i, f'{count}\n({pct:.0%})',
                    ha='center', va='center',
                    color=text_color, fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title('Confusion Matrix - Test Set Performance', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_feature_correlations(df: pd.DataFrame, save_path: Path) -> None:
    """correlation plot of features vs match outcome."""
    print("Creating feature correlations plot...")
    
    # encode target
    target_map = {'H': 0, 'D': 1, 'A': 2}
    df = df.copy()
    df['Target'] = df['FTR'].map(target_map)
    
    # get feature columns
    feature_cols = [col for col in df.columns if 'RollingAvg' in col or '_Diff' in col]
    
    # correlations with target
    correlations = df[feature_cols + ['Target']].corr()['Target'].drop('Target')
    correlations = correlations.sort_values(ascending=True)
    
    # top correlated features
    top_features = pd.concat([correlations.head(8), correlations.tail(8)])
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # color by sign
    colors = ['#e74c3c' if x < 0 else '#2ecc71' for x in top_features.values]
    
    # shorten feature names
    short_names = []
    for name in top_features.index:
        name = name.replace('_RollingAvg', '')
        name = name.replace('Home_', 'H:')
        name = name.replace('Away_', 'A:')
        short_names.append(name)
    
    bars = ax.barh(range(len(top_features)), top_features.values, color=colors, edgecolor='black')
    
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(short_names, fontsize=10)
    ax.set_xlabel('Correlation with Outcome', fontsize=12)
    ax.set_title('Feature Correlations with Match Outcome\n(Positive = favors Away Win, Negative = favors Home Win)', 
                 fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linewidth=0.8)
    
    # value labels
    for bar, val in zip(bars, top_features.values):
        x_pos = val + 0.01 if val >= 0 else val - 0.01
        ha = 'left' if val >= 0 else 'right'
        ax.text(x_pos, bar.get_y() + bar.get_height()/2, f'{val:.3f}',
                ha=ha, va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def main() -> None:
    """generate all visualizations."""
    print("="*50)
    print("EPL Predictor - Generating Visualizations")
    print("="*50)
    
    figures_dir = ensure_figures_dir()
    
    # load data
    features_path = config.PROCESSED_DATA_DIR / "features_dataset.csv"
    if not features_path.exists():
        print(f"Error: Features file not found at {features_path}")
        print("Run: python -m epl_predictor.features.build_features")
        return
    
    df = pd.read_csv(features_path, parse_dates=['Date'])
    print(f"\nLoaded {len(df)} matches")
    
    # generate plots
    print()
    plot_class_distribution(df, figures_dir / 'class_distribution.png')
    plot_confusion_matrix(figures_dir / 'confusion_matrix.png')
    plot_feature_correlations(df, figures_dir / 'feature_correlations.png')
    
    print(f"\nAll visualizations saved to: {figures_dir}/")
    print("="*50)


if __name__ == "__main__":
    main()

