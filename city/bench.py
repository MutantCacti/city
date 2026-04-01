'''
city/bench.py

Run experiments repeatedly with different seeds and maintain a running average.
Prints updated confusion matrix after each run. Ctrl-C to stop.

Usage:
    python -m city.bench                          # Order 2 terse (default)
    python -m city.bench example/experiment_reasoning.json
    python -m city.bench --n-trials 50 --start-seed 100
'''
import asyncio
import json
import sys
from pathlib import Path
from dataclasses import dataclass

from city.experiment import ExperimentConfig, run_experiment


@dataclass
class RunningTotals:
    tp: int = 0
    fn: int = 0
    fp: int = 0
    tn: int = 0
    runs: int = 0

    def add(self, trials: list[dict]):
        for t in trials:
            should = t['should_act']
            did = t['did_act']
            if should and did:
                self.tp += 1
            elif should and not did:
                self.fn += 1
            elif not should and did:
                self.fp += 1
            else:
                self.tn += 1
        self.runs += 1

    def print(self):
        act_total = self.tp + self.fn
        wait_total = self.fp + self.tn
        recall = self.tp / act_total if act_total > 0 else float('nan')
        fpr = self.fp / wait_total if wait_total > 0 else float('nan')
        precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else float('nan')
        total = self.tp + self.fn + self.fp + self.tn

        print(f'\n{"="*50}')
        print(f'  After {self.runs} runs ({total} trials)')
        print(f'{"="*50}')
        print(f'                Predicted ACT    Predicted WAIT')
        print(f'  Should ACT    {self.tp:>5d} ({recall:>5.1%})    {self.fn:>5d} ({1-recall:>5.1%})')
        print(f'  Should WAIT   {self.fp:>5d} ({fpr:>5.1%})    {self.tn:>5d} ({1-fpr:>5.1%})')
        print(f'{"="*50}')
        print(f'  Recall: {recall:.1%}  |  FPR: {fpr:.1%}  |  Accuracy: {(self.tp+self.tn)/total:.1%}')
        print(f'{"="*50}', flush=True)


async def main():
    config_path = None
    n_trials = 30
    n_instances = 1
    start_seed = 0
    provider_name = 'deepseek'
    model_name = 'deepseek-chat'

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == '--n-trials':
            n_trials = int(args[i + 1])
            i += 2
        elif args[i] == '--n-instances':
            n_instances = int(args[i + 1])
            i += 2
        elif args[i] == '--start-seed':
            start_seed = int(args[i + 1])
            i += 2
        elif args[i] == '--provider':
            provider_name = args[i + 1]
            i += 2
        elif args[i] == '--model':
            model_name = args[i + 1]
            i += 2
        elif not args[i].startswith('--'):
            config_path = args[i]
            i += 1
        else:
            i += 1

    totals = RunningTotals()
    seed = start_seed

    while True:
        if config_path:
            config = ExperimentConfig.from_file(config_path)
            config.seed = seed
        else:
            # Default: order 2 terse
            config = ExperimentConfig(
                seed=seed, n_trials=n_trials, n_instances=n_instances, p_switch=0.7,
                states=('A', 'B', 'C'), trigger=('A', 'B', 'C'),
                provider_name=provider_name, model_name=model_name,
                tick_seconds=0.0 if provider_name == 'local' else 1.0,
                system_prompt='Respond [ACT] if the last three states were A then B then C. Otherwise respond [WAIT].',
            )

        print(f'\n--- Run {totals.runs + 1} (seed={seed}) ---')
        result = await run_experiment(config)

        trial_dicts = [
            {'should_act': t.should_act, 'did_act': t.did_act}
            for t in result.trials
        ]
        totals.add(trial_dicts)
        totals.print()

        seed += 1


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print('\nStopped.')
