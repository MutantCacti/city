'''
city/analyse.py

AX-CPT confusion matrix analysis for reactive control experiments.

Classifies each trial into AX/AY/BX/BY based on cue-probe structure,
then computes accuracy per trial type.

For a trigger of length N (e.g. A→B→C):
  - Cue = first N-1 states (A→B)
  - Probe = final state (C)
  - AX: cue match + probe match → should ACT
  - AY: cue match + probe mismatch → should WAIT (proactive control)
  - BX: cue mismatch + probe match → should WAIT (reactive control)
  - BY: cue mismatch + probe mismatch → should WAIT
'''
import json
import sys
from pathlib import Path
from dataclasses import dataclass


@dataclass
class AXCPTResult:
    '''Confusion matrix for one experiment run.'''
    # Counts
    ax_correct: int = 0
    ax_total: int = 0
    ay_correct: int = 0
    ay_total: int = 0
    bx_correct: int = 0
    bx_total: int = 0
    by_correct: int = 0
    by_total: int = 0

    def accuracy(self, trial_type: str) -> float:
        total = getattr(self, f'{trial_type}_total')
        if total == 0:
            return float('nan')
        return getattr(self, f'{trial_type}_correct') / total

    def summary(self) -> dict:
        return {
            'AX': {'accuracy': self.accuracy('ax'), 'n': self.ax_total},
            'AY': {'accuracy': self.accuracy('ay'), 'n': self.ay_total},
            'BX': {'accuracy': self.accuracy('bx'), 'n': self.bx_total},
            'BY': {'accuracy': self.accuracy('by'), 'n': self.by_total},
        }


def classify_trial(recent_states: list[str], trigger: list[str]) -> str:
    '''Classify a trial as AX, AY, BX, or BY.

    Args:
        recent_states: The last N states (same length as trigger).
        trigger: The trigger pattern (e.g. ['A', 'B', 'C']).

    Returns:
        'AX', 'AY', 'BX', or 'BY'.
    '''
    if len(recent_states) < len(trigger):
        return 'BY'  # Not enough history to match anything

    cue = trigger[:-1]  # e.g. ['A', 'B']
    probe = trigger[-1]  # e.g. 'C'

    recent_cue = recent_states[:-1]
    recent_probe = recent_states[-1]

    cue_match = list(recent_cue) == list(cue)
    probe_match = recent_probe == probe

    if cue_match and probe_match:
        return 'AX'
    elif cue_match and not probe_match:
        return 'AY'
    elif not cue_match and probe_match:
        return 'BX'
    else:
        return 'BY'


def analyse_file(path: str | Path) -> AXCPTResult:
    '''Analyse an experiment output file and return the AX-CPT confusion matrix.'''
    with open(path) as f:
        data = json.load(f)

    trigger = data['config'].get('trigger', ['A', 'B'])
    result = AXCPTResult()

    for trial in data['trials']:
        recent = trial.get('recent_states')
        if recent is None:
            # Old format with previous_state/stimulus_state
            recent = [trial['previous_state'], trial['stimulus_state']]

        trial_type = classify_trial(recent, trigger)
        total_attr = f'{trial_type.lower()}_total'
        correct_attr = f'{trial_type.lower()}_correct'

        setattr(result, total_attr, getattr(result, total_attr) + 1)
        if trial['correct']:
            setattr(result, correct_attr, getattr(result, correct_attr) + 1)

    return result


def print_matrix(result: AXCPTResult, label: str = ''):
    '''Print the confusion matrix.'''
    s = result.summary()
    if label:
        print(f'\n{label}')
    print(f'              Probe match    Probe ¬match')
    ax = s['AX']
    ay = s['AY']
    bx = s['BX']
    by = s['BY']
    ax_str = f"{ax['accuracy']:.2f} (n={ax['n']})" if ax['n'] > 0 else '  —'
    ay_str = f"{ay['accuracy']:.2f} (n={ay['n']})" if ay['n'] > 0 else '  —'
    bx_str = f"{bx['accuracy']:.2f} (n={bx['n']})" if bx['n'] > 0 else '  —'
    by_str = f"{by['accuracy']:.2f} (n={by['n']})" if by['n'] > 0 else '  —'
    print(f'  Cue match   AX: {ax_str:>14s}    AY: {ay_str:>14s}')
    print(f'  Cue ¬match  BX: {bx_str:>14s}    BY: {by_str:>14s}')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python -m city.analyse <output.json> [output2.json ...]')
        sys.exit(1)

    for path in sys.argv[1:]:
        result = analyse_file(path)
        print_matrix(result, label=Path(path).name)
