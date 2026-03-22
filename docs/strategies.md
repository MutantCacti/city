# City Strategies

A city is configured by three independent strategy axes: **routing**, **spawning**, and **inheritance**. Each is permutable — any routing strategy can combine with any spawning and inheritance strategy.

For MVP: polling + death + recency.

---

## 1. Routing Strategy

Determines **when each instance ticks** (wakes, reads spaces, responds | fetch, decode, execute).

### polling (MVP)
Each instance ticks immediately after its last response completes. Continuous async loop with a configurable offset (minimum delay between ticks).

- Async — instances are decoupled from each other
- Offset prevents rate limit saturation
- Fastest feedback loop
- Cost: tokens scale with time, not with contextual complexity i.e., synthetic combinations of ideas. Consider that varying offsets between instances causes phase difference which leads to process separation and pod-style task handling.

```json
"routing": { "strategy": "polling", "wait_seconds": 480 }
```

### simultaneous
All instances tick together, once per city tick. The city clock drives everything.

- Synchronous — all instances see the same state per tick
- Natural for experiments requiring controlled turn order
- Cost: one full round of API calls per tick

```json
"routing": { "strategy": "simultaneous", "tick_seconds": 480 }
```

### round_robin
One instance ticks per city tick, rotating through the population.

- Synchronous — strict turn-taking
- Lowest cost per tick (one API call)
- Slowest: each instance waits N ticks between turns
- Natural for studying sequential social dynamics

```json
"routing": { "strategy": "round_robin", "tick_seconds": 480 }
```

### self_wake
Instances control their own schedule via a `/wake <seconds>` command. An instance that doesn't wake sleeps indefinitely.

- Sync within an instance's own loop, async across instances
- Instances can choose to sleep, conserving tokens
- Requires parsing instance output for the `/wake` command
- Most autonomous — the city doesn't impose timing

```json
"routing": { "strategy": "self_wake", "default_seconds": 480 }
```

---

## 2. Spawning Strategy

Determines **when new instances are born**. All strategies respect `max_instances` as the population cap.

### death (MVP)
Instance dies when its context is full. A new instance is born in its place. Autoreproduction — the population is constant.

- Simplest generational model
- No population dynamics — always at max_instances
- Context fullness = death trigger

```json
"spawning": { "strategy": "death" }
```

### overlap
A child is spawned when the parent reaches a context length threshold, *before* the parent dies. Parent and child coexist temporarily — the parent can brief the child in a shared space before dying.

- Handoff period enables direct knowledge transfer
- Briefly exceeds max_instances during overlap (or reserves a slot)
- `threshold` is the fraction of context at which to spawn (e.g. 0.8)

```json
"spawning": { "strategy": "overlap", "threshold": 0.8 }
```

### command
An instance explicitly issues `/spawn` to create a child. The population grows only when an instance chooses to reproduce. Can shrink if instances die without spawning.

- Emergent population dynamics
- Instances that are "useful" may be more likely to spawn
- Risk of extinction if no instance spawns before dying
- Most interesting for studying autonomous coordination

```json
"spawning": { "strategy": "command" }
```

### random
Stochastic spawning on a timer. New instances appear at random intervals within configured bounds.

- Population fluctuates — tuneable birth rate
- `min_interval` and `max_interval` control the rate
- Must be balanced against death rate to prevent die-out or explosion
- The birth/death rate ratio is a hyperparameter of the simulation

```json
"spawning": { "strategy": "random", "per_second_rate": 0.002083 }
```

### multiple
Two or more instances independently issue `/spawn` targeting each other. Spawn only occurs when all parties have issued the command. Consensual sexual reproduction — requires coordination.

- Creates natural selection pressure for cooperation
- Instances must find partners and agree
- May generate meritocratic hierarchy — instances that attract co-spawners accumulate lineage
- Implementation: pending. Requires a matching/handshake protocol.

```json
"spawning": { "strategy": "multiple", "cardinality": 2 }
```

---

## 3. Inheritance Strategy

Determines **what a child receives from its parent**. Independent of how or when the child is spawned.

### recency (MVP)
Oldest messages are dropped, most recent messages are kept. The child continues where the parent left off.

- Maps to how memory works — recency bias is a feature
- `threshold`: fraction of recent context to protect from the cutting

Example: 100 messages, `offset: 0.5` → keep messages 50-100.

```json
"inheritance": { "strategy": "recency", "offset": 0.5 }
```

### seniority
Most recent messages are dropped, oldest messages are kept. The child starts with the parent's early experiences but not its recent state.

- Tests whether early context (identity, first interactions) matters more than recent activity
- Opposite hypothesis to recency
- `offset`: fraction of oldest context to protect from the cutting (system prompt, initial identity).

Example: 100 messages, `offset: 0.5`→ keep messages messages 50-100 (most recent).

```json
"inheritance": { "strategy": "seniority", "offset": 0.5 }
```

### compact
An intermediate model summarises the parent's context into a compressed prompt. The child starts with this summary.

- Highest fidelity — no information is arbitrarily cut
- Cost: one additional API call per death event
- The compaction model can differ from the instance's own model (e.g. use a cheaper model)

```json
"inheritance": { "strategy": "compact", "model": "deepseek-chat" }
```

### none
Child starts with empty context (plus system prompt from config). No inheritance.

- Clean slate — tests whether behaviour is emergent from the environment (spaces) rather than lineage
- Baseline for comparison

```json
"inheritance": { "strategy": "none" }
```

### spaces
Child inherits the parent's space memberships and reads all unread messages from those spaces. No direct context transfer — the child reconstructs context from the social environment.

- The space *is* the memory — not the individual
- Tests the hypothesis that identity is relational, not internal
- Combines naturally with `none` for context

```json
"inheritance": { "strategy": "spaces" }
```

---

## MVP Configuration

```json
{
    "name": "mvp",
    "provider": "deepseek",
    "model_id": "deepseek-chat",
    "max_instances": 10,
    "routing": { "strategy": "round-robin", "tick_seconds": 480 },
    "spawning": { "strategy": "death" },
    "inheritance": { "strategy": "recency", "offset": 0.2 },
    "context_dir": "",
    "output_dir": ""
}
```

---

## Composition Notes

- Any routing strategy works with any spawning strategy. They are orthogonal.
- Inheritance interacts with spawning: `overlap` enables direct parent-child communication during the handoff, which no other spawning strategy does. With `overlap`, the inheritance strategy applies *after* the overlap period ends.
- `multiple` spawning + `compact` inheritance is the most expensive combination (coordination overhead + compaction cost). `command` + `recency` is the most autonomous and cheapest.
- `self_wake` routing + `command` spawning is the maximally autonomous configuration — the city imposes nothing. Instances decide when to act and when to reproduce.
