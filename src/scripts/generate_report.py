#!/usr/bin/env python3
"""Generate comprehensive training report from tensorboard logs."""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tbparse import SummaryReader
import base64
from datetime import datetime

# Configuration
RAPORT_DIR = "raport"
TENSORBOARD_DIR = os.path.join(RAPORT_DIR, "tensorboard")
CSV_FILE = os.path.join(RAPORT_DIR, "training_metrics.csv")
OUT_DIR = os.path.join(RAPORT_DIR, "html_report")
os.makedirs(OUT_DIR, exist_ok=True)

# Model colors for consistent plotting
MODEL_COLORS = {
    "KAN_FAST": "#FF6B6B",
    "resnet50": "#4ECDC4",
    "vgg16": "#45B7D1",
    "densenet121": "#96CEB4",
    "mobilenet_v2": "#FFEAA7",
    "efficientnet_b0": "#DDA15E",
    "vit_b_16": "#BC6C25",
}


def extract_all_model_data(tensorboard_dir: str) -> dict:
    """Extract metrics from all model runs."""
    model_data = {}

    for model_dir in Path(tensorboard_dir).iterdir():
        if not model_dir.is_dir():
            continue

        model_name = model_dir.name
        print(f"📊 Reading {model_name}...")

        try:
            reader = SummaryReader(str(model_dir), pivot=True)
            df = reader.scalars

            if not df.empty:
                model_data[model_name] = df
                print(f"   ✅ Found {len(df)} steps")
            else:
                print("   ⚠️  No data found")
        except Exception as e:
            print(f"   ❌ Error: {e}")

    return model_data


def create_comparison_plots(model_data: dict, output_dir: str) -> list:
    """Create comparison plots for all models."""
    plots = []
    sns.set_style("whitegrid")

    # 1. Training Loss Comparison
    plt.figure(figsize=(14, 7))
    for model_name, df in model_data.items():
        if "train_loss_epoch" in df.columns:
            color = MODEL_COLORS.get(model_name, None)
            plt.plot(
                df.index,
                df["train_loss_epoch"],
                label=model_name,
                marker="o",
                markersize=4,
                linewidth=2,
                color=color,
            )

    plt.title("Training Loss Comparison", fontsize=18, fontweight="bold")
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.legend(fontsize=11, loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, "train_loss_comparison.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    plots.append(("Training Loss Comparison", plot_path))

    # 2. Validation Loss Comparison
    plt.figure(figsize=(14, 7))
    for model_name, df in model_data.items():
        if "val_loss" in df.columns:
            color = MODEL_COLORS.get(model_name, None)
            plt.plot(
                df.index,
                df["val_loss"],
                label=model_name,
                marker="s",
                markersize=4,
                linewidth=2,
                color=color,
            )

    plt.title("Validation Loss Comparison", fontsize=18, fontweight="bold")
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.legend(fontsize=11, loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, "val_loss_comparison.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    plots.append(("Validation Loss Comparison", plot_path))

    # 3. Training Accuracy Comparison
    plt.figure(figsize=(14, 7))
    for model_name, df in model_data.items():
        if "train_accuracy_epoch" in df.columns:
            color = MODEL_COLORS.get(model_name, None)
            plt.plot(
                df.index,
                df["train_accuracy_epoch"] * 100,
                label=model_name,
                marker="o",
                markersize=4,
                linewidth=2,
                color=color,
            )

    plt.title("Training Accuracy Comparison", fontsize=18, fontweight="bold")
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Accuracy (%)", fontsize=14)
    plt.legend(fontsize=11, loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, "train_acc_comparison.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    plots.append(("Training Accuracy Comparison", plot_path))

    # 4. Validation Accuracy Comparison
    plt.figure(figsize=(14, 7))
    for model_name, df in model_data.items():
        if "val_accuracy" in df.columns:
            color = MODEL_COLORS.get(model_name, None)
            plt.plot(
                df.index,
                df["val_accuracy"] * 100,
                label=model_name,
                marker="s",
                markersize=4,
                linewidth=2,
                color=color,
            )

    plt.title("Validation Accuracy Comparison", fontsize=18, fontweight="bold")
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Accuracy (%)", fontsize=14)
    plt.legend(fontsize=11, loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, "val_acc_comparison.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    plots.append(("Validation Accuracy Comparison", plot_path))

    # 5. Final Metrics Bar Chart
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    metrics = [
        ("train_loss_epoch", "Final Training Loss", axes[0, 0], False),
        ("val_loss", "Final Validation Loss", axes[0, 1], False),
        ("train_accuracy_epoch", "Final Training Accuracy", axes[1, 0], True),
        ("val_accuracy", "Final Validation Accuracy", axes[1, 1], True),
    ]

    for metric_col, title, ax, is_accuracy in metrics:
        model_names = []
        values = []
        colors = []

        for model_name, df in model_data.items():
            if metric_col in df.columns and len(df) > 0:
                final_value = df[metric_col].iloc[-1]
                if is_accuracy:
                    final_value *= 100
                model_names.append(model_name)
                values.append(final_value)
                colors.append(MODEL_COLORS.get(model_name, "#95a5a6"))

        bars = ax.bar(range(len(model_names)), values, color=colors)
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=45, ha="right")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_ylabel("Accuracy (%)" if is_accuracy else "Loss", fontsize=12)
        ax.grid(True, alpha=0.3, axis="y")

        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "final_metrics_comparison.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    plots.append(("Final Metrics Comparison", plot_path))

    return plots


def create_summary_table(model_data: dict) -> pd.DataFrame:
    """Create summary table with final metrics for all models."""
    summary = []

    for model_name, df in model_data.items():
        if len(df) == 0:
            continue

        final_metrics = df.iloc[-1]

        row = {
            "Model": model_name,
            "Train Loss": final_metrics.get("train_loss_epoch", None),
            "Val Loss": final_metrics.get("val_loss", None),
            "Train Acc (%)": (
                final_metrics.get("train_accuracy_epoch", None) * 100
                if "train_accuracy_epoch" in final_metrics
                else None
            ),
            "Val Acc (%)": (
                final_metrics.get("val_accuracy", None) * 100
                if "val_accuracy" in final_metrics
                else None
            ),
            "Epochs": len(df),
        }
        summary.append(row)

    return pd.DataFrame(summary)


def create_html_report(
    model_data: dict, plots: list, summary_df: pd.DataFrame, output_dir: str
):
    """Generate comprehensive HTML report."""

    html = [
        "<!DOCTYPE html>",
        "<html><head>",
        "<meta charset='utf-8'>",
        "<title>Model Training Report</title>",
        "<style>",
        "body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }",
        ".container { max-width: 1400px; margin: 0 auto; background: white; padding: 40px; border-radius: 15px; box-shadow: 0 10px 40px rgba(0,0,0,0.3); }",
        "h1 { color: #2c3e50; border-bottom: 4px solid #3498db; padding-bottom: 15px; font-size: 2.5em; margin-bottom: 10px; }",
        ".subtitle { color: #7f8c8d; font-size: 1.1em; margin-bottom: 30px; }",
        "h2 { color: #34495e; margin-top: 40px; padding: 15px; background: linear-gradient(90deg, #3498db 0%, #2ecc71 100%); color: white; border-radius: 8px; }",
        "h3 { color: #2c3e50; margin-top: 30px; font-size: 1.5em; border-left: 5px solid #3498db; padding-left: 15px; }",
        ".summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 30px 0; }",
        ".metric-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 25px; border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); color: white; text-align: center; }",
        ".metric-card h4 { margin: 0 0 10px 0; font-size: 1.1em; opacity: 0.9; }",
        ".metric-card .value { font-size: 2.5em; font-weight: bold; margin: 10px 0; }",
        ".metric-card .model-name { font-size: 0.9em; opacity: 0.8; margin-top: 10px; }",
        "img { max-width: 100%; height: auto; margin: 25px 0; box-shadow: 0 5px 20px rgba(0,0,0,0.15); border-radius: 10px; }",
        "table { border-collapse: collapse; width: 100%; background: white; margin: 25px 0; box-shadow: 0 2px 10px rgba(0,0,0,0.1); border-radius: 8px; overflow: hidden; }",
        "th, td { border: none; padding: 15px 20px; text-align: left; }",
        "th { background: linear-gradient(90deg, #3498db 0%, #2ecc71 100%); color: white; font-weight: 600; text-transform: uppercase; font-size: 0.9em; letter-spacing: 0.5px; }",
        "tr:nth-child(even) { background-color: #f8f9fa; }",
        "tr:hover { background-color: #e8f4f8; transition: background-color 0.3s; }",
        "td { color: #2c3e50; }",
        ".best-value { background-color: #d4edda; font-weight: bold; color: #155724; padding: 5px 10px; border-radius: 5px; }",
        ".footer { text-align: center; margin-top: 50px; padding: 20px; color: #7f8c8d; border-top: 2px solid #ecf0f1; }",
        ".plot-section { margin: 40px 0; padding: 20px; background: #f8f9fa; border-radius: 10px; }",
        "</style>",
        "</head><body>",
        "<div class='container'>",
        "<h1>🚀 Model Training Report</h1>",
        f"<p class='subtitle'>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>",
    ]

    # Best performers section
    html.append("<h2>🏆 Best Performers</h2>")
    html.append("<div class='summary-grid'>")

    # Best validation accuracy
    best_val_acc_row = summary_df.loc[summary_df["Val Acc (%)"].idxmax()]
    html.append("<div class='metric-card'>")
    html.append("<h4>Best Validation Accuracy</h4>")
    html.append(f"<div class='value'>{best_val_acc_row['Val Acc (%)']:.2f}%</div>")
    html.append(f"<div class='model-name'>{best_val_acc_row['Model']}</div>")
    html.append("</div>")

    # Lowest validation loss
    best_val_loss_row = summary_df.loc[summary_df["Val Loss"].idxmin()]
    html.append("<div class='metric-card'>")
    html.append("<h4>Lowest Validation Loss</h4>")
    html.append(f"<div class='value'>{best_val_loss_row['Val Loss']:.4f}</div>")
    html.append(f"<div class='model-name'>{best_val_loss_row['Model']}</div>")
    html.append("</div>")

    # Total models
    html.append("<div class='metric-card'>")
    html.append("<h4>Models Compared</h4>")
    html.append(f"<div class='value'>{len(model_data)}</div>")
    html.append("<div class='model-name'>Total Architectures</div>")
    html.append("</div>")

    html.append("</div>")

    # Summary table
    html.append("<h2>📊 Summary Table</h2>")

    # Highlight best values
    styled_df = summary_df.copy()

    # Convert to HTML with styling
    table_html = styled_df.to_html(
        index=False,
        border=0,
        escape=False,
        float_format=lambda x: f"{x:.4f}" if pd.notna(x) else "N/A",
    )

    # Add best value highlighting
    best_val_acc = summary_df["Val Acc (%)"].max()
    best_val_loss = summary_df["Val Loss"].min()

    table_html = table_html.replace(
        f"{best_val_acc:.4f}", f"<span class='best-value'>{best_val_acc:.2f}</span>"
    )
    table_html = table_html.replace(
        f"{best_val_loss:.4f}", f"<span class='best-value'>{best_val_loss:.4f}</span>"
    )

    html.append(table_html)

    # Comparison plots
    html.append("<h2>📈 Training Curves</h2>")
    for plot_name, plot_path in plots:
        html.append("<div class='plot-section'>")
        html.append(f"<h3>{plot_name}</h3>")
        with open(plot_path, "rb") as f:
            b64_img = base64.b64encode(f.read()).decode("utf-8")
        html.append(f"<img src='data:image/png;base64,{b64_img}' alt='{plot_name}'>")
        html.append("</div>")

    # Footer
    html.append("<div class='footer'>")
    html.append("<p><strong>SNT-KAN-CNN Project</strong></p>")
    html.append(f"<p>Report generated from {len(model_data)} model runs</p>")
    html.append("</div>")

    html.append("</div></body></html>")

    report_path = os.path.join(output_dir, "report.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html))

    return report_path


def generate():
    """Main report generation function."""
    print("=" * 60)
    print("🎯 SNT-KAN-CNN Training Report Generator")
    print("=" * 60)
    print()

    if not os.path.exists(TENSORBOARD_DIR):
        print(f"❌ Error: TensorBoard directory not found: {TENSORBOARD_DIR}")
        return

    # Extract data from all models
    print("📊 Extracting data from all models...")
    model_data = extract_all_model_data(TENSORBOARD_DIR)

    if not model_data:
        print("❌ No model data found!")
        return

    print(f"\n✅ Successfully loaded {len(model_data)} models")
    print()

    # Create summary table
    print("📋 Creating summary table...")
    summary_df = create_summary_table(model_data)
    print(summary_df.to_string(index=False))
    print()

    # Generate comparison plots
    print("📈 Generating comparison plots...")
    plots = create_comparison_plots(model_data, OUT_DIR)
    print(f"✅ Generated {len(plots)} plots")
    print()

    # Create HTML report
    print("📝 Creating HTML report...")
    report_path = create_html_report(model_data, plots, summary_df, OUT_DIR)
    print(f"✅ Report saved: {report_path}")
    print()

    # Save summary CSV
    csv_path = os.path.join(OUT_DIR, "models_summary.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"💾 Summary CSV saved: {csv_path}")
    print()

    print("=" * 60)
    print("🎉 Done! Open the report:")
    print(f"   file://{os.path.abspath(report_path)}")
    print("=" * 60)


if __name__ == "__main__":
    generate()
