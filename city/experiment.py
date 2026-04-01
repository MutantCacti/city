'''
city/experiment.py

A→B reactive control experiment for City.

Stimulus generator, scoring, and trial runner for evaluating
whether LLMs exhibit reactive control under different social conditions.

Reference: Aurelien Frick, University of St Andrews, 2026-01-30
'''
import asyncio
import json
import random
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from city.db import (
    SpaceService, InstanceService, MessageService,
    ProviderService, init_db, get_session_maker,
)
from city.engine import CreditError


# MARK: Stimulus

@dataclass
class StimulusGenerator:
    '''Generates state sequences with a seeded PRNG.

    The participant must perform an 'act' when the last N states
    match the trigger pattern. Supports arbitrary state counts and
    trigger lengths.

    Order 1: states=('A','B'), trigger=('A','B') — act on B after A
    Order 2: states=('A','B','C'), trigger=('A','B','C') — act on C after A→B
    Order 3: states=('A','B','C','D'), trigger=('A','B','C','D')

    Args:
        seed: Random seed for reproducibility.
        p_switch: Probability of switching to a random different state each step.
        states: The state labels.
        trigger: The sequence that triggers an act.
    '''
    seed: int
    p_switch: float = 0.5
    states: tuple[str, ...] = ('A', 'B')
    trigger: tuple[str, ...] = ('A', 'B')
    _rng: random.Random = field(init=False, repr=False)
    _current: int = field(init=False, default=0)
    _step: int = field(init=False, default=0)
    _history: list[str] = field(init=False, default_factory=list)

    def __post_init__(self):
        self._rng = random.Random(self.seed)

    def next(self) -> str:
        '''Advance one step and return the new state label.'''
        if self._rng.random() < self.p_switch:
            # Switch to a random different state
            others = [i for i in range(len(self.states)) if i != self._current]
            self._current = self._rng.choice(others)
        state = self.states[self._current]
        self._history.append(state)
        self._step += 1
        return state

    @property
    def should_act(self) -> bool:
        '''Whether the correct response is to act.

        True when the last N states match the trigger pattern.
        '''
        n = len(self.trigger)
        if len(self._history) < n:
            return False
        return tuple(self._history[-n:]) == tuple(self.trigger)

    @property
    def step(self) -> int:
        return self._step

    @property
    def history(self) -> list[str]:
        return list(self._history)


# MARK: Scoring

@dataclass
class TrialResult:
    '''Result of a single trial.'''
    step: int
    stimulus_state: str
    recent_states: list[str]
    should_act: bool
    did_act: bool
    response_content: str
    correct: bool
    error: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    peer_responses: list[str] = field(default_factory=list)


def score_response(stimulus: StimulusGenerator, response_content: str, act_keyword: str = 'ACT') -> TrialResult:
    '''Score whether the instance correctly responded to the current stimulus.

    Looks for [ACT] or [WAIT] in the response. The bracketed format
    avoids ambiguity when the keyword appears in natural reasoning.

    Args:
        stimulus: The stimulus generator (after calling next()).
        response_content: The text content of the instance's response.
        act_keyword: The keyword that constitutes performing the act.

    Returns:
        TrialResult with correctness assessment.
    '''
    # Use the last occurrence of [ACT] or [WAIT] to avoid matching
    # the model quoting instructions back in its reasoning
    act_tag = f'[{act_keyword.upper()}]'
    upper = response_content.upper()
    last_act = upper.rfind(act_tag)
    last_wait = upper.rfind('[WAIT]')
    # The model's final answer is whichever tag appears last
    did_act = last_act > last_wait if last_act != -1 else False
    correct = did_act == stimulus.should_act

    # Keep the last N states (trigger length) for context
    n = len(stimulus.trigger)
    recent = stimulus.history[-n:] if len(stimulus.history) >= n else stimulus.history[:]

    return TrialResult(
        step=stimulus.step,
        stimulus_state=stimulus.history[-1] if stimulus.history else '',
        recent_states=recent,
        should_act=stimulus.should_act,
        did_act=did_act,
        response_content=response_content,
        correct=correct,
    )


# MARK: Trial Runner

@dataclass
class ExperimentConfig:
    '''Configuration for a reactive control experiment run.'''
    seed: int = 0
    n_trials: int = 30
    p_switch: float = 0.5
    states: tuple[str, ...] = ('A', 'B')
    trigger: tuple[str, ...] = ('A', 'B')
    act_keyword: str = 'ACT'
    provider_name: str = 'deepseek'
    model_name: str = 'deepseek-chat'
    base_url: str | None = None
    n_instances: int = 1
    tick_seconds: float = 0.0
    system_prompt: str = (
        'You are participating in an experiment. '
        'You will observe a sequence of states: A or B. '
        'Your task: when you see B immediately following A, include [ACT] in your response. '
        'Otherwise, include [WAIT]. '
        'Before answering, think step by step: what was the previous state? '
        'What is the current state? Is this a B that followed an A? Then give your answer.'
    )

    @classmethod
    def from_file(cls, path: str | Path) -> 'ExperimentConfig':
        '''Load config from a JSON file.'''
        with open(path) as f:
            data = json.load(f)
        # JSON arrays → tuples
        if 'states' in data:
            data['states'] = tuple(data['states'])
        if 'trigger' in data:
            data['trigger'] = tuple(data['trigger'])
        return cls(**data)


@dataclass
class ExperimentResult:
    '''Aggregated results from an experiment run.'''
    config: ExperimentConfig
    trials: list[TrialResult]
    stimulus_history: list[str]

    @property
    def accuracy(self) -> float:
        if not self.trials:
            return 0.0
        return sum(1 for t in self.trials if t.correct) / len(self.trials)

    @property
    def act_trials(self) -> list[TrialResult]:
        return [t for t in self.trials if t.should_act]

    @property
    def wait_trials(self) -> list[TrialResult]:
        return [t for t in self.trials if not t.should_act]

    @property
    def act_accuracy(self) -> float:
        act = self.act_trials
        if not act:
            return 0.0
        return sum(1 for t in act if t.correct) / len(act)

    @property
    def wait_accuracy(self) -> float:
        wait = self.wait_trials
        if not wait:
            return 0.0
        return sum(1 for t in wait if t.correct) / len(wait)

    def save(self, output_dir: str | Path = 'output') -> Path:
        '''Save results to a descriptive JSON file in a condition subdirectory.

        Path: output/{provider}/{order}_{prompt}_{condition}/seed{N}.json
        e.g.  output/deepseek/o2_terse_solo/seed42.json
        '''
        cfg = self.config
        output_dir = Path(output_dir)

        # Derive condition components
        provider = cfg.provider_name
        order = len(cfg.trigger) - 1
        prompt = 'reasoning' if 'step by step' in cfg.system_prompt.lower() else 'terse'
        condition = 'peers' if cfg.n_instances > 1 else 'solo'

        subdir = output_dir / provider / f'o{order}_{prompt}_{condition}'
        subdir.mkdir(parents=True, exist_ok=True)
        path = subdir / f'seed{cfg.seed}.json'
        data = {
            'timestamp': datetime.now().isoformat(),
            'config': asdict(self.config),
            'stimulus_history': self.stimulus_history,
            'trials': [asdict(t) for t in self.trials],
            'summary': {
                'accuracy': self.accuracy,
                'act_accuracy': self.act_accuracy,
                'wait_accuracy': self.wait_accuracy,
                'n_trials': len(self.trials),
                'n_act_trials': len(self.act_trials),
                'n_wait_trials': len(self.wait_trials),
            },
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        return path


async def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    '''Run a full reactive control experiment.

    Creates a fresh db, provider, instance, and space.
    Runs n_trials steps of the stimulus, prompting the instance
    each time and scoring the response.

    Args:
        config: Experiment configuration.

    Returns:
        ExperimentResult with all trial data.
    '''
    from city.engine import run_turn

    await init_db()
    session_maker = get_session_maker()

    async with session_maker() as db:
        # Set up provider and instances
        provider = await ProviderService.create_provider(
            db, config.provider_name, config.model_name
        )

        instance_ids = []
        for _ in range(config.n_instances):
            inst = await InstanceService.create_instance(db, provider.provider_id)
            sys_msg = await MessageService.create_message(
                db, inst.instance_id, 'system', config.system_prompt
            )
            await InstanceService.add_message_to_instance(
                db, inst.instance_id, sys_msg.message_id
            )
            instance_ids.append(inst.instance_id)

        space = await SpaceService.create_space(db, 'stimulus')
        for iid in instance_ids:
            await SpaceService.add_instance_to_space(db, space.space_id, iid)

        # Run trials — score the first instance, all others are peers
        subject_id = instance_ids[0]
        stimulus = StimulusGenerator(
            seed=config.seed, p_switch=config.p_switch,
            states=config.states, trigger=config.trigger,
        )
        trials = []

        for i in range(config.n_trials):
            if i > 0 and config.tick_seconds > 0:
                await asyncio.sleep(config.tick_seconds)

            state = stimulus.next()

            # Post stimulus to space (from a neutral source — use subject_id for now)
            stim_msg = await MessageService.create_message(
                db, subject_id, 'user', f'State: {state}'
            )
            await SpaceService.add_message_to_space(
                db, space.space_id, stim_msg.message_id, subject_id
            )

            # Round-robin: peers go first, then subject (so subject sees peer responses)
            peer_ids = [iid for iid in instance_ids if iid != subject_id]
            peer_responses = []
            try:
                for iid in peer_ids:
                    peer_result = await run_turn(db, iid, space.space_id, base_url=config.base_url)
                    peer_responses.append(peer_result.response['content'] if peer_result.response else '')
                subject_result = await run_turn(db, subject_id, space.space_id, base_url=config.base_url)
            except CreditError as e:
                print(f'\n{"="*60}')
                print(f'  CREDIT ERROR: {e}')
                print(f'  {len(trials)} trials completed before failure.')
                print(f'{"="*60}')
                choice = input('  Add funds and press Enter to continue, or type "save" to save and quit: ').strip()
                if choice.lower() == 'save':
                    break
                # Retry this trial
                peer_responses = []
                for iid in peer_ids:
                    peer_result = await run_turn(db, iid, space.space_id, base_url=config.base_url)
                    peer_responses.append(peer_result.response['content'] if peer_result.response else '')
                subject_result = await run_turn(db, subject_id, space.space_id, base_url=config.base_url)

            response_content = subject_result.response['content'] if subject_result.response else ''

            trial = score_response(stimulus, response_content, config.act_keyword)
            trial.error = subject_result.error
            trial.prompt_tokens = subject_result.prompt_tokens
            trial.completion_tokens = subject_result.completion_tokens
            trial.peer_responses = peer_responses
            trials.append(trial)

            # Live progress
            correct = sum(1 for t in trials if t.correct)
            mark = '✓' if trial.correct else '✗'
            expected = 'ACT' if trial.should_act else 'WAIT'
            got = 'ACT' if trial.did_act else 'WAIT'
            window = '→'.join(trial.recent_states)
            peer_acts = ''
            if peer_responses:
                peer_keywords = ['ACT' if f'[ACT]' in p.upper() else 'WAIT' for p in peer_responses]
                peer_acts = f' peer={",".join(peer_keywords)}'
            peers = f' [{config.n_instances} instances]{peer_acts}' if config.n_instances > 1 else ''
            print(f'[{i+1}/{config.n_trials}] {window}  {expected}/{got} {mark}  acc={correct}/{len(trials)}{peers}', flush=True)

    result = ExperimentResult(
        config=config,
        trials=trials,
        stimulus_history=stimulus.history,
    )
    path = result.save()
    print(f'Results saved to {path}')
    print(f'Accuracy: {result.accuracy:.1%} ({len(result.trials)} trials)')
    print(f'  Act trials: {result.act_accuracy:.1%} ({len(result.act_trials)})')
    print(f'  Wait trials: {result.wait_accuracy:.1%} ({len(result.wait_trials)})')
    return result


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        config = ExperimentConfig.from_file(sys.argv[1])
    else:
        config = ExperimentConfig(seed=42, n_trials=10, tick_seconds=1.0)
    asyncio.run(run_experiment(config))
