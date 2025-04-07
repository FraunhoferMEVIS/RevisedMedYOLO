import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_metrics(folder):
    training_file = os.path.join(folder, "training_loss.csv")
    validation_file = os.path.join(folder, "validation_loss.csv")

    if not os.path.exists(training_file) or not os.path.exists(validation_file):
        print("Error: Training or validation file not found in the specified folder.")
        return

    try:
        train_df = pd.read_csv(training_file)
        val_df = pd.read_csv(validation_file)
    except Exception as e:
        print(f"Error reading CSV files: {e}")
        return

    output_path = os.path.join(folder, "loss_plot.png")

    loss_metrics = ["total_loss", "bounding_box_loss", "objectness_loss", "classification_loss"]
    map_metrics = ["mAP@0.1", "mAP@0.1:0.95"]

    fig, ax1 = plt.subplots(figsize=(14, 8))
    ax2 = ax1.twinx()

    all_metrics_combined = sorted(list(set(loss_metrics + map_metrics)))
    colors = sns.color_palette("tab10", len(all_metrics_combined))
    color_map = {metric: colors[i] for i, metric in enumerate(all_metrics_combined)}

    lines_handles = []
    labels_list = []

    for metric in loss_metrics:
        if metric in train_df.columns and 'epoch' in train_df.columns:
            line, = ax1.plot(train_df["epoch"], train_df[metric], linestyle="dashed", color=color_map.get(metric, 'black'), label=f"Train {metric}")
            lines_handles.append(line)
            labels_list.append(f"Train {metric}")

    for metric in loss_metrics + map_metrics:
        if metric in val_df.columns and 'epoch' in val_df.columns:
            if metric in loss_metrics:
                line, = ax1.plot(val_df["epoch"], val_df[metric], linestyle="solid", color=color_map.get(metric, 'black'), label=f"Val {metric}")
                lines_handles.append(line)
                labels_list.append(f"Val {metric}")
            elif metric in map_metrics:
                line, = ax2.plot(val_df["epoch"], val_df[metric], linestyle="solid", color=color_map.get(metric, 'grey'), label=f"Val {metric}")
                lines_handles.append(line)
                labels_list.append(f"Val {metric}")

    best_map_metric = "mAP@0.1:0.95"
    vline_handle = None
    vline_label = None
    if best_map_metric in val_df.columns and 'epoch' in val_df.columns and not val_df[best_map_metric].isnull().all():
        try:
            best_epoch_idx = val_df[best_map_metric].idxmax()
            best_epoch = val_df.loc[best_epoch_idx, "epoch"]
            vline_label = f'Best Model (Epoch {best_epoch:.0f})'
            vline_color = color_map.get(best_map_metric, 'black')
            vline_handle = ax1.axvline(x=best_epoch, color=vline_color, linestyle='dotted', label=vline_label)
        except (ValueError, KeyError):
             print(f"Warning: Could not determine best epoch based on '{best_map_metric}'.")

    if vline_handle and vline_label:
        lines_handles.append(vline_handle)
        labels_list.append(vline_label)

    
    upper_y_lim_loss = np.percentile(val_df['total_loss'], 95)
    ax1.set_xlim(0, max(val_df['epoch']))
    ax1.set_ylim(0, upper_y_lim_loss)
    ax2.set_ylim(0, 1)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss Value")
    ax2.set_ylabel("Metric Value (mAP)")

    plt.title("Training and Validation Metrics per Epoch")

    ax2.legend(handles=lines_handles, labels=labels_list, loc="upper left")

    ax2.grid(True)

    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training and validation losses from CSV logs in a given folder.")
    parser.add_argument("folder", type=str, help="Path to folder containing training_loss.csv and validation_loss.csv.")
    args = parser.parse_args()

    plot_metrics(args.folder)