import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_metrics(folder):
    training_file = os.path.join(folder, "training_loss.csv")
    validation_file = os.path.join(folder, "validation_loss.csv")
    
    if not os.path.exists(training_file) or not os.path.exists(validation_file):
        print("Error: Training or validation file not found in the specified folder.")
        return
    
    train_df = pd.read_csv(training_file)
    val_df = pd.read_csv(validation_file)
    
    output_path = os.path.join(folder, "loss_plot.png")
    
    metrics = ["total_loss", "bounding_box_loss", "objectness_loss", "classification_loss"]
    val_metrics = metrics + ["mAP@0.1", "mAP@0.1:0.95"]
    
    plt.figure(figsize=(10, 6))
    
    all_metrics = sorted(list(set(metrics + val_metrics)))
    colors = sns.color_palette("tab10", len(all_metrics))
    color_map = {metric: colors[i] for i, metric in enumerate(all_metrics)}
    
    for metric in metrics:
        if metric in train_df.columns:
            plt.plot(train_df["epoch"], train_df[metric], linestyle="dashed", color=color_map[metric], label=f"Train {metric}")
    
    for metric in val_metrics:
        if metric in val_df.columns:
            plt.plot(val_df["epoch"], val_df[metric], linestyle="solid", color=color_map[metric], label=f"Val {metric}")
    
    if "mAP@0.1:0.95" in val_df.columns:
        best_epoch = val_df.loc[val_df["mAP@0.1:0.95"].idxmax(), "epoch"]
        plt.axvline(x=best_epoch, color=color_map["mAP@0.1:0.95"], linestyle='dotted', label=f'Best Model (Epoch {best_epoch})')
    
    plt.xlabel("Epoch")
    plt.ylabel("Loss / Metric Value")
    plt.title("Training and Validation Metrics per Epoch")
    plt.legend()
    plt.grid(True)
    
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training and validation losses from CSV logs in a given folder.")
    parser.add_argument("folder", type=str, help="Path to folder containing training_loss.csv and validation_loss.csv.")
    args = parser.parse_args()
    
    plot_metrics(args.folder)