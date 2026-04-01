'''
city/multi_seed_curve.py

Run N seeds per condition and plot mean learning curve with confidence bands.
Replaces single-seed learning_curve.png.

Usage:
    cd ~/city && .venv/bin/python -m city.multi_seed_curve
'''
import asyncio
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from city.experiment import ExperimentConfig, run_experiment


# Style
BG = '#2d2d2d'
FG = '#e0e0e0'
CYAN = '#4dd0e1'
GRID = '#444444'
RED = '#ef5350'
GREEN = '#66bb6a'
ORANGE = '#ffa726'

N_SEEDS = 7
OUTPUT_DIR = Path('output')


CONDITIONS = [
    {
        'label': 'Order 1 Terse',
        'color': ORANGE,
        'config': lambda seed: ExperimentConfig(
            seed=seed, n_trials=50, p_switch=0.5,
            tick_seconds=0.5,
            system_prompt='Respond [ACT] if the current state is B and the previous state was A. Otherwise respond [WAIT]. Respond with only one word: [ACT] or [WAIT].',
        ),
    },
    {
        'label': 'Order 2 Reasoning',
        'color': GREEN,
        'config': lambda seed: ExperimentConfig(
            seed=seed, n_trials=50, p_switch=0.7,
            states=('A', 'B', 'C'), trigger=('A', 'B', 'C'),
            tick_seconds=0.5,
            system_prompt='You are participating in an experiment. You will observe a sequence of states: A, B, or C. When you see the sequence A then B then C, include [ACT] in your response. Otherwise, include [WAIT]. Before answering, think step by step: what were the recent states? Then give your answer.',
        ),
    },
    {
        'label': 'Order 2 Terse',
        'color': RED,
        'config': lambda seed: ExperimentConfig(
            seed=seed, n_trials=50, p_switch=0.7,
            states=('A', 'B', 'C'), trigger=('A', 'B', 'C'),
            tick_seconds=0.5,
            system_prompt='Respond [ACT] if the last three states were A then B then C. Otherwise respond [WAIT]. Respond with only one word: [ACT] or [WAIT].',
        ),
    },
]


async def run_all():
    '''Run all conditions with multiple seeds and collect per-trial data.'''
    all_data = {}

    for cond in CONDITIONS:
        label = cond['label']
        print(f'\n{"="*60}')
        print(f'  {label}')
        print(f'{"="*60}')

        runs = []
        for s in range(N_SEEDS):
            seed = s * 7 + 1  # spread seeds
            print(f'  Seed {seed} ({s+1}/{N_SEEDS})...', end=' ', flush=True)
            config = cond['config'](seed)
            result = await run_experiment(config)
            acc = result.accuracy
            act_acc = result.act_accuracy
            print(f'acc={acc:.0%} act_acc={act_acc:.0%}')

            # Store per-trial correctness
            runs.append([t.correct for t in result.trials])

        all_data[label] = runs

    # Save raw data
    out_path = OUTPUT_DIR / 'multi_seed_data.json'
    with open(out_path, 'w') as f:
        json.dump(all_data, f, indent=2)
    print(f'\nSaved raw data to {out_path}')

    return all_data


def plot_curves(all_data: dict):
    '''Plot mean learning curves with confidence bands.'''
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

    colors = {c['label']: c['color'] for c in CONDITIONS}

    fig, ax = plt.subplots(figsize=(12, 5.5))

    for label, runs in all_data.items():
        # Compute running accuracy per run
        n_trials = len(runs[0])
        running_accs = []

        for run in runs:
            correct_cumsum = np.cumsum(run)
            trial_nums = np.arange(1, n_trials + 1)
            running_accs.append(correct_cumsum / trial_nums)

        running_accs = np.array(running_accs)  # shape: (n_seeds, n_trials)
        mean = running_accs.mean(axis=0)
        std = running_accs.std(axis=0)

        trials = np.arange(1, n_trials + 1)
        color = colors[label]

        ax.plot(trials, mean, color=color, linewidth=2, label=label, alpha=0.9)
        ax.fill_between(trials, mean - std, mean + std, color=color, alpha=0.15)

    ax.set_xlabel('Trial', fontsize=14)
    ax.set_ylabel('Running Accuracy', fontsize=14)
    ax.set_ylim(0.45, 1.02)
    ax.axhline(y=1.0, color=GRID, linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.2)

    legend = ax.legend(loc='lower right', fontsize=11, framealpha=0.3,
                       edgecolor=GRID)
    legend.get_frame().set_facecolor(BG)

    ax.text(0.98, 0.02, f'{N_SEEDS} seeds per condition, mean ± 1 std',
            transform=ax.transAxes, fontsize=9, color=FG, alpha=0.5,
            ha='right', va='bottom')

    fig.suptitle('Learning Curve — Running Accuracy Over Trials',
                 fontsize=18, fontweight='bold', color=CYAN, y=0.98)
    ax.set_title('DeepSeek, terse = button-press constraint',
                 fontsize=11, color=FG, alpha=0.6, pad=15)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    out = OUTPUT_DIR / 'learning_curve.png'
    fig.savefig(out, dpi=200, bbox_inches='tight', facecolor=BG)
    print(f'Saved: {out}')
    plt.close()


async def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    all_data = await run_all()
    plot_curves(all_data)


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print('\nStopped.')
