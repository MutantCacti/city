'''
city/plot.py

Generate presentation-ready graphs from the output tree.
Reads output/{provider}/{condition}/ directories automatically.

Usage:
    python -m city.plot                          # all providers
    python -m city.plot --provider deepseek      # one provider
    python -m city.plot --output-dir output      # custom root
'''
import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path


# Style
BG = '#2d2d2d'
FG = '#e0e0e0'
CYAN = '#4dd0e1'
GRID = '#444444'
RED = '#ef5350'
GREEN = '#66bb6a'
ORANGE = '#ffa726'
BLUE = '#42a5f5'
PURPLE = '#ab47bc'

PINK = '#f06292'
LIME = '#c6ff00'

CONDITION_COLORS = {
    'o1_terse_solo': ORANGE,
    'o1_terse_peers': BLUE,
    'o1_reasoning_solo': GREEN,
    'o1_reasoning_peers': PURPLE,
    'o2_terse_solo': RED,
    'o2_terse_peers': CYAN,
    'o2_reasoning_solo': LIME,
    'o2_reasoning_peers': PINK,
}

CONDITION_LABELS = {
    'o1_terse_solo': 'O1 Terse Solo',
    'o1_terse_peers': 'O1 Terse Peers',
    'o1_reasoning_solo': 'O1 Reasoning Prompt Solo',
    'o1_reasoning_peers': 'O1 Reasoning Prompt Peers',
    'o2_terse_solo': 'O2 Terse Solo',
    'o2_terse_peers': 'O2 Terse Peers',
    'o2_reasoning_solo': 'O2 Reasoning Prompt Solo',
    'o2_reasoning_peers': 'O2 Reasoning Prompt Peers',
}


def setup_style():
    plt.rcParams.update({
        'figure.facecolor': BG,
        'axes.facecolor': BG,
        'axes.edgecolor': GRID,
        'axes.labelcolor': FG,
        'text.color': FG,
        'xtick.color': FG,
        'ytick.color': FG,
        'grid.color': GRID,
        'font.family': 'sans-serif',
        'font.size': 14,
    })


# MARK: Data loading

def load_condition(condition_dir: Path) -> list[dict]:
    '''Load all seed runs from a condition directory.'''
    runs = []
    for f in sorted(condition_dir.glob('seed*.json')):
        with open(f) as fh:
            runs.append(json.load(fh))
    return runs


def load_provider(provider_dir: Path) -> dict[str, list[dict]]:
    '''Load all conditions for a provider. Returns {condition_name: [runs]}.'''
    conditions = {}
    for d in sorted(provider_dir.iterdir()):
        if d.is_dir() and not d.name.startswith('.'):
            runs = load_condition(d)
            if runs:
                conditions[d.name] = runs
    return conditions


def aggregate_confusion(runs: list[dict]) -> dict:
    '''Aggregate TP/FN/FP/TN across all runs.'''
    tp = fn = fp = tn = 0
    for run in runs:
        for t in run['trials']:
            should = t['should_act']
            did = t['did_act']
            if should and did:
                tp += 1
            elif should and not did:
                fn += 1
            elif not should and did:
                fp += 1
            else:
                tn += 1
    return {'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn}


# MARK: Confusion matrices

def _render_confusion(ax, d: dict, label: str, color: str, n_runs: int, n_trials: int, cmap):
    '''Render a single 2x2 confusion matrix on an axis.'''
    rate_matrix = np.array([
        [d['tp'] / max(d['tp'] + d['fn'], 1), d['fn'] / max(d['tp'] + d['fn'], 1)],
        [d['fp'] / max(d['fp'] + d['tn'], 1), d['tn'] / max(d['fp'] + d['tn'], 1)],
    ])
    colors = np.array([
        [rate_matrix[0, 0], 1 - rate_matrix[0, 1]],
        [1 - rate_matrix[1, 0], rate_matrix[1, 1]],
    ])

    ax.imshow(colors, cmap=cmap, vmin=0, vmax=1, aspect='equal')

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Predicted\nACT', 'Predicted\nWAIT'], fontsize=9)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Should\nACT', 'Should\nWAIT'], fontsize=9)
    ax.xaxis.set_ticks_position('top')

    cells = [
        (0, 0, d['tp'], d['tp'] + d['fn']),
        (0, 1, d['fn'], d['tp'] + d['fn']),
        (1, 0, d['fp'], d['fp'] + d['tn']),
        (1, 1, d['tn'], d['fp'] + d['tn']),
    ]
    for row, col, count, total_row in cells:
        rate = count / total_row if total_row > 0 else 0
        text_color = '#000000' if colors[row, col] > 0.5 else '#ffffff'
        ax.text(col, row, f'{rate:.0%}', ha='center', va='center',
                fontsize=22, fontweight='bold', color=text_color)

    ax.set_title(f'{label}\n({n_runs} runs, {n_trials} trials)',
                 fontsize=11, fontweight='bold', color=color, pad=25)


def plot_confusion_matrices(provider_dir: Path, output_dir: Path):
    '''Plot confusion matrices as 2x2 grids: one figure per order, per provider.'''
    setup_style()

    conditions = load_provider(provider_dir)
    if not conditions:
        return

    provider = provider_dir.name
    cmap = LinearSegmentedColormap.from_list('rg', [RED, ORANGE, GREEN])

    # Group by order
    orders = {}
    for cond_name, runs in conditions.items():
        order = cond_name.split('_')[0]  # 'o1' or 'o2'
        if order not in orders:
            orders[order] = {}
        orders[order][cond_name] = runs

    for order, order_conditions in orders.items():
        # 2x2 grid: rows = terse/reasoning, cols = solo/peers
        grid = {
            (0, 0): None,  # terse solo
            (0, 1): None,  # terse peers
            (1, 0): None,  # reasoning solo
            (1, 1): None,  # reasoning peers
        }
        for cond_name, runs in order_conditions.items():
            is_reasoning = 'reasoning' in cond_name
            is_peers = 'peers' in cond_name
            row = 1 if is_reasoning else 0
            col = 1 if is_peers else 0
            grid[(row, col)] = (cond_name, runs)

        fig, axes = plt.subplots(2, 2, figsize=(9, 9))

        for (row, col), entry in grid.items():
            ax = axes[row][col]
            if entry is None:
                ax.set_visible(False)
                continue
            cond_name, runs = entry
            d = aggregate_confusion(runs)
            n_runs = len(runs)
            n_trials = sum(len(r['trials']) for r in runs)
            color = CONDITION_COLORS.get(cond_name, FG)
            label = CONDITION_LABELS.get(cond_name, cond_name)
            _render_confusion(ax, d, label, color, n_runs, n_trials, cmap)

        order_label = order.upper().replace('O', 'Order ')
        fig.suptitle(f'{order_label} — {provider}',
                     fontsize=18, fontweight='bold', color=CYAN, y=1.0)

        plt.tight_layout()
        out = output_dir / f'confusion_{provider}_{order}.png'
        fig.savefig(out, dpi=200, bbox_inches='tight', facecolor=BG)
        print(f'Saved: {out}')
        plt.close()


# MARK: Learning curves

def _compute_balanced_acc(trials: list[dict]) -> np.ndarray:
    '''Compute running balanced accuracy over a trial sequence.'''
    bal_accs = []
    tp = fn = fp = tn = 0
    for t in trials:
        should = t['should_act']
        did = t['did_act']
        if should and did: tp += 1
        elif should and not did: fn += 1
        elif not should and did: fp += 1
        else: tn += 1
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.5
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.5
        bal_accs.append((recall + tnr) / 2)
    return np.array(bal_accs)


def _plot_accuracy_ax(ax, order_conditions: dict, order: str):
    '''Plot accuracy curves for one order on a given axis.'''
    for cond_name, runs in order_conditions.items():
        color = CONDITION_COLORS.get(cond_name, FG)
        label = CONDITION_LABELS.get(cond_name, cond_name)
        # Strip order prefix from label for cleaner legend
        short_label = label.replace(f'{order.upper()} ', '')

        max_trials = max(len(r['trials']) for r in runs)
        running_accs = []

        for run in runs:
            acc = _compute_balanced_acc(run['trials'])
            if len(acc) < max_trials:
                acc = np.concatenate([acc, np.full(max_trials - len(acc), np.nan)])
            running_accs.append(acc)

        running_accs = np.array(running_accs)
        mean = np.nanmean(running_accs, axis=0)
        std = np.nanstd(running_accs, axis=0)
        trials = np.arange(1, max_trials + 1)

        is_peers = 'peers' in cond_name
        linestyle = '--' if is_peers else '-'

        ax.plot(trials, mean, color=color, linewidth=2,
                label=f'{short_label} ({len(runs)} seeds)',
                alpha=0.9, linestyle=linestyle)
        ax.fill_between(trials, mean - std, mean + std, color=color, alpha=0.12)

    ax.set_xlabel('Trial', fontsize=14)
    ax.set_ylabel('Balanced Accuracy', fontsize=14)
    ax.set_ylim(0.35, 1.02)
    ax.axhline(y=0.5, color=GRID, linestyle=':', alpha=0.4)
    ax.axhline(y=1.0, color=GRID, linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.2)

    legend = ax.legend(loc='lower right', fontsize=10, framealpha=0.3,
                       edgecolor=GRID)
    legend.get_frame().set_facecolor(BG)

def plot_learning_curves(provider_dir: Path, output_dir: Path):
    '''Plot accuracy curves split by order: one graph per order per provider.'''
    setup_style()

    conditions = load_provider(provider_dir)
    if not conditions:
        return

    provider = provider_dir.name

    # Group by order
    orders = {}
    for cond_name, runs in conditions.items():
        order = cond_name.split('_')[0]
        if order not in orders:
            orders[order] = {}
        orders[order][cond_name] = runs

    for order, order_conditions in orders.items():
        fig, ax = plt.subplots(figsize=(12, 5.5))

        _plot_accuracy_ax(ax, order_conditions, order)

        order_label = order.upper().replace('O', 'Order ')
        fig.suptitle(f'Accuracy Curve — {provider} — {order_label}',
                     fontsize=18, fontweight='bold', color=CYAN, y=0.98)
        ax.set_title('mean ± 1 std, solid = solo, dashed = peers',
                     fontsize=11, color=FG, alpha=0.6, pad=15)

        plt.tight_layout(rect=[0, 0, 1, 0.92])
        out = output_dir / f'accuracy_curve_{provider}_{order}.png'
        fig.savefig(out, dpi=200, bbox_inches='tight', facecolor=BG)
        print(f'Saved: {out}')
        plt.close()


# MARK: Spaghetti plots

def plot_spaghetti(provider_dir: Path, output_dir: Path):
    '''Plot individual seed trajectories with mean overlay.'''
    setup_style()

    conditions = load_provider(provider_dir)
    if not conditions:
        return

    provider = provider_dir.name

    orders = {}
    for cond_name, runs in conditions.items():
        order = cond_name.split('_')[0]
        if order not in orders:
            orders[order] = {}
        orders[order][cond_name] = runs

    for order, order_conditions in orders.items():
        fig, ax = plt.subplots(figsize=(12, 5.5))

        for cond_name, runs in order_conditions.items():
            color = CONDITION_COLORS.get(cond_name, FG)
            label = CONDITION_LABELS.get(cond_name, cond_name)
            short_label = label.replace(f'{order.upper()} ', '')
            is_peers = 'peers' in cond_name

            max_trials = max(len(r['trials']) for r in runs)
            trials = np.arange(1, max_trials + 1)
            running_accs = []

            for run in runs:
                acc = _compute_balanced_acc(run['trials'])
                if len(acc) < max_trials:
                    acc = np.concatenate([acc, np.full(max_trials - len(acc), np.nan)])
                running_accs.append(acc)
                ax.plot(trials, acc, color=color, linewidth=0.5, alpha=0.15)

            mean = np.nanmean(running_accs, axis=0)
            linestyle = '--' if is_peers else '-'
            ax.plot(trials, mean, color=color, linewidth=2.5, alpha=0.9,
                    linestyle=linestyle, label=f'{short_label} ({len(runs)} seeds)')

        ax.set_xlabel('Trial', fontsize=14)
        ax.set_ylabel('Balanced Accuracy', fontsize=14)
        ax.set_ylim(0.35, 1.02)
        ax.axhline(y=0.5, color=GRID, linestyle=':', alpha=0.4)
        ax.axhline(y=1.0, color=GRID, linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.2)

        legend = ax.legend(loc='lower right', fontsize=10, framealpha=0.3,
                           edgecolor=GRID)
        legend.get_frame().set_facecolor(BG)

        order_label = order.upper().replace('O', 'Order ')
        fig.suptitle(f'Individual Seeds — {provider} — {order_label}',
                     fontsize=18, fontweight='bold', color=CYAN, y=0.98)
        ax.set_title('thin = individual seeds, thick = mean',
                     fontsize=11, color=FG, alpha=0.6, pad=15)

        plt.tight_layout(rect=[0, 0, 1, 0.92])
        out = output_dir / f'spaghetti_{provider}_{order}.png'
        fig.savefig(out, dpi=200, bbox_inches='tight', facecolor=BG)
        print(f'Saved: {out}')
        plt.close()


# MARK: Seed heatmaps

def plot_seed_heatmap(provider_dir: Path, output_dir: Path):
    '''Plot ranked seed heatmaps: seeds sorted by final accuracy, color = balanced acc per trial.'''
    setup_style()

    conditions = load_provider(provider_dir)
    if not conditions:
        return

    provider = provider_dir.name
    cmap = LinearSegmentedColormap.from_list('rg', [RED, ORANGE, GREEN])

    orders = {}
    for cond_name, runs in conditions.items():
        order = cond_name.split('_')[0]
        if order not in orders:
            orders[order] = {}
        orders[order][cond_name] = runs

    for order, order_conditions in orders.items():
        grid = {
            (0, 0): None,
            (0, 1): None,
            (1, 0): None,
            (1, 1): None,
        }
        for cond_name, runs in order_conditions.items():
            is_reasoning = 'reasoning' in cond_name
            is_peers = 'peers' in cond_name
            row = 1 if is_reasoning else 0
            col = 1 if is_peers else 0
            grid[(row, col)] = (cond_name, runs)

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        for (row, col), entry in grid.items():
            ax = axes[row][col]
            if entry is None:
                ax.set_visible(False)
                continue
            cond_name, runs = entry
            color = CONDITION_COLORS.get(cond_name, FG)
            label = CONDITION_LABELS.get(cond_name, cond_name)
            short_label = label.replace(f'{order.upper()} ', '')

            max_trials = max(len(r['trials']) for r in runs)
            matrix = []
            for run in runs:
                acc = _compute_balanced_acc(run['trials'])
                if len(acc) < max_trials:
                    acc = np.concatenate([acc, np.full(max_trials - len(acc), np.nan)])
                matrix.append(acc)
            matrix = np.array(matrix)

            # Sort by final balanced accuracy, best at top
            final_accs = matrix[:, -1]
            sort_idx = np.argsort(final_accs)[::-1]
            matrix = matrix[sort_idx]

            im = ax.imshow(matrix, aspect='auto', cmap=cmap, vmin=0.3, vmax=1.0,
                           interpolation='nearest')
            ax.set_xlabel('Trial', fontsize=10)
            ax.set_ylabel('Seed (ranked)', fontsize=10)
            ax.set_title(f'{short_label} ({len(runs)} seeds)',
                         fontsize=11, fontweight='bold', color=color, pad=8)

        order_label = order.upper().replace('O', 'Order ')
        fig.suptitle(f'Seed Heatmap — {provider} — {order_label}',
                     fontsize=18, fontweight='bold', color=CYAN, y=1.0)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        fig.subplots_adjust(bottom=0.12)
        fig.colorbar(im, ax=axes.ravel().tolist(), label='Balanced Accuracy',
                     shrink=0.3, pad=0.08, location='bottom')
        out = output_dir / f'heatmap_{provider}_{order}.png'
        fig.savefig(out, dpi=200, bbox_inches='tight', facecolor=BG)
        print(f'Saved: {out}')
        plt.close()


# MARK: Entry point

def main():
    output_root = Path('output')
    graph_dir = output_root / 'graphs'
    graph_dir.mkdir(parents=True, exist_ok=True)

    providers = []
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == '--provider':
            providers.append(args[i + 1])
            i += 2
        elif args[i] == '--output-dir':
            output_root = Path(args[i + 1])
            graph_dir = output_root / 'graphs'
            graph_dir.mkdir(parents=True, exist_ok=True)
            i += 2
        else:
            i += 1

    if not providers:
        providers = [d.name for d in sorted(output_root.iterdir())
                     if d.is_dir() and d.name not in ('graphs', 'logs')]

    for provider in providers:
        provider_dir = output_root / provider
        if not provider_dir.exists():
            print(f'Skipping {provider}: directory not found')
            continue
        print(f'\n--- {provider} ---')
        plot_confusion_matrices(provider_dir, graph_dir)
        plot_learning_curves(provider_dir, graph_dir)
        plot_spaghetti(provider_dir, graph_dir)
        plot_seed_heatmap(provider_dir, graph_dir)


if __name__ == '__main__':
    main()
