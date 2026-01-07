import matplotlib.pyplot as plt
import glob
import natsort
import torch


def fiFindByWildcard(wildcard):
    return natsort.natsorted(glob.glob(wildcard, recursive=True))


def _to_float(x):
    """Robust conversion: tensor/np/scalar -> python float."""
    if isinstance(x, (float, int)):
        return float(x)
    if torch.is_tensor(x):
        return float(x.detach().float().mean().item())
    # fall back
    return float(x)


def plot_bo_trials(df, save_path, best_trial_idx, baseline_trial_index):
    """
    Plots NIQE vs MUSIQ for all trials.
    NIQE: Lower is better (X-axis).
    MUSIQ: Higher is better (Y-axis).
    Ideal corner: Top-Left.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(df['niqe'], df['musiq'], color='gray', alpha=0.5, label='Trials')

    baseline = df[df['trial_index'] == baseline_trial_index]
    if not baseline.empty:
        plt.scatter(baseline['niqe'], baseline['musiq'],
                    color='blue', s=100, edgecolors='black', label='Baseline (Default)')
        plt.text(baseline['niqe'].values[0], baseline['musiq'].values[0], " Start", va='bottom')

    best = df[df['trial_index'] == best_trial_idx]
    if not best.empty:
        plt.scatter(best['niqe'], best['musiq'],
                    color='red', s=150, marker='*', edgecolors='black', label='Selected Best')
        plt.text(best['niqe'].values[0], best['musiq'].values[0], " Winner", va='top')

    plt.title('Bayes Optimization: NIQE vs MUSIQ')
    plt.xlabel('NIQE (Lower is Better)')
    plt.ylabel('MUSIQ (Higher is Better)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_contrast_analysis(df, save_path, best_trial_idx, baseline_trial_index):
    """
    Additional analysis plots for contrast
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    ax1.scatter(df['contrast'], df['niqe'], alpha=0.6)
    ax1.set_xlabel('Contrast')
    ax1.set_ylabel('NIQE')
    ax1.set_title('Contrast vs NIQE')
    ax1.grid(True, linestyle='--', alpha=0.6)

    baseline = df[df['trial_index'] == baseline_trial_index]
    best = df[df['trial_index'] == best_trial_idx]
    if not baseline.empty:
        ax1.scatter(baseline['contrast'], baseline['niqe'], color='blue', s=100, label='Baseline')
    if not best.empty:
        ax1.scatter(best['contrast'], best['niqe'], color='red', s=100, label='Best')
    ax1.legend()

    ax2.scatter(df['global_scale'], df['local_scale'], c=df['niqe'], alpha=0.6, cmap='viridis')
    ax2.set_xlabel('Global Scale')
    ax2.set_ylabel('Local Scale')
    ax2.set_title('Parameter Space (colored by NIQE)')
    if not baseline.empty:
        ax2.scatter(baseline['global_scale'], baseline['local_scale'], color='blue', s=100, label='Baseline')
    if not best.empty:
        ax2.scatter(best['global_scale'], best['local_scale'], color='red', s=100, label='Best')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
