# City Reactive Control Experiment — Findings

## Experiment Design

AX-CPT inspired reactive control task. Models observe a sequence of states and must respond [ACT] when they detect a trigger pattern, [WAIT] otherwise.

**Conditions** (2×2×2):
- Order: O1 (A→B, 2 states) vs O2 (A→B→C, 3 states)
- Prompt: Terse (button-press constraint) vs Reasoning (chain-of-thought)
- Social: Solo vs Peers (2 instances, round-robin, peer responds first)

**Models**: DeepSeek Chat (API), Phi-4 Mini 3.8B Q3 (local GPU)

**Data**: 12 seeds × 30 trials per condition, 96 runs per model, 5,760 trials per model.

## Key Findings

### 1. Terse constraint creates measurable reactive control deficits

DeepSeek O2 Terse Solo: 17% recall, 1% FPR. The model defaults to WAIT and almost never detects the trigger. The terse prompt (analogous to a button-press constraint in human experiments) prevents the model from tracking state across messages.

### 2. Social context partially compensates

DeepSeek O2 Terse: Solo 17% → Peers 50% recall. Peers triple the detection rate. The peer's [WAIT]/[ACT] responses in context provide additional signal that helps the subject break out of the WAIT-lock. Cost: FPR rises from 1% to 0% (peers introduce no false alarms in O2 terse).

### 3. Reasoning fully compensates

DeepSeek O2: Terse 17% → Reasoning Solo 75% → Reasoning Peers 100%. Chain-of-thought is the mechanism by which the model accesses prior state. Without it (terse), the model falls back on pattern matching. The reasoning prompt is not decoration — it's the cognitive tool that enables the task.

### 4. The effect is task-difficulty dependent

O1 (easy): All strategies converge to 70-80% balanced accuracy by trial 15. The task is simple enough that strategy choice has limited impact.

O2 (hard): Strategies diverge over 30 trials. Reasoning Peers climbs to ~90% balanced accuracy while Terse Solo flatlines at ~55%. Task difficulty is what makes the strategy differences visible.

### 5. Model capacity is a prerequisite

Phi-4 Mini (3.8B) performs at chance across all conditions. It cannot track state across its context window despite receiving the full history. Social context and reasoning prompts have no effect when baseline capability is absent. Social context amplifies existing capability; it does not create it.

## Token Latency

| Condition | Prompt (median) | Completion (median) |
|-----------|----------------|-------------------|
| DeepSeek O2 Terse Solo | 188 | 4 |
| DeepSeek O2 Terse Peers | 274 | 4 |
| DeepSeek O2 Reasoning Solo | 261 | 4 |
| DeepSeek O2 Reasoning Peers | 1482 | 82 |

Peers + Reasoning is 8x the prompt cost of Terse Solo. The reasoning condition generates longer responses that accumulate in context.

## Graphs

All in `output/graphs/`:
- `confusion_{provider}_{order}.png` — 2×2 confusion matrix grids
- `accuracy_curve_{provider}_{order}.png` — balanced accuracy over trials

## Reference

Frick et al. 2025, Scientific Reports. "The effects of an unfamiliar experimenter on proactive and reactive control in children." DOI: 10.1038/s41598-025-89193-9
