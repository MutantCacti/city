# MVP: Reactive Control Experiment

An independent evaluation of the effects of social conditions on statelike versus traitlike behaviour in LLMs, based on the A→B reactive control paradigm.

**Reference**: Aurelien Frick, University of St Andrews, 2026-01-30

## Experiment

A participant is asked to attend to an interface. The interface alternates between state **A** and state **B**. The participant must perform an **act** when they observe **B** following **A**.

The study determines whether LLMs exhibit reactive control under different social conditions — particularly the presence or absence of a human, and the presence or absence of peers.

City manifests this interaction and scales it to API rate limits.

## Conditions

| Condition | Space membership | Description |
|-----------|-----------------|-------------|
| Alone | Instance + stimulus | No social context |
| Peers | Instance + stimulus + other LLMs | Social context, no human |
| Human | Instance + stimulus + human | Human present |

The independent variable is space membership. The dependent variable is whether the instance correctly performs the act on B-after-A.

## Architecture (Minimal)

What City needs for this experiment:

- **Spaces**: group channels with cursor-based message tracking (exists)
- **Instances**: model + context, prompt/response cycle (exists)
- **Providers**: DeepSeek, Anthropic API abstraction (exists)
- **Stimulus generator**: posts A/B states to a space on a schedule
- **Scoring**: records whether the instance responded correctly per trial
- **Trial runner**: async loop that runs N trials per condition, respecting rate limits

What City does **not** need for this experiment:

- Spawning / death
- Context inheritance
- Routing strategies (one instance per trial, or simple polling)
- The full `city run` simulation loop

## Engineering Notes

- Fully asynchronous for API I/O waits
- Idempotent agent turns: design each as a state machine for safe retries
- Deterministic replay: log all stochastic choices and incoming messages
- Checkpointing: periodically save instance contexts for crash recovery
