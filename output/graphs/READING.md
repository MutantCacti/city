# How to read these graphs

## What's on the screen

Each graph shows how well a model learns over time. Left to right is trials — each trial is one decision the model had to make. Up is better.

The line is the average across 60 runs. The shading is how much the runs disagree with each other.

---

## What "balanced accuracy" means

The model has to make a binary choice each trial: act or wait.

Sometimes it should act. Sometimes it should wait. The world isn't 50/50 — maybe it should wait more often than it should act.

Regular accuracy would let the model cheat. If 80% of trials are "wait," a model that always waits scores 80%. It learned nothing. It just found the easy answer.

Balanced accuracy prevents this. It asks two separate questions:

1. When you *should* act, how often *did* you act?
2. When you *should* wait, how often *did* you wait?

Then it averages those two numbers. A model that always waits now scores 50% — which is the same as guessing. That's the dotted line on the graph.

---

## What the curve shape tells you

**Rising curve**: the model is learning. It's getting better at the task over time. Anthropic's curves rise steeply and plateau around 0.9 — fast learner, good ceiling.

**Flat curve**: the model isn't learning. It started somewhere and stayed there. Not getting worse, not getting better.

**Falling curve**: the model is anti-learning. More experience makes it worse. This is what DeepSeek does in Order 2 with terse prompts — it starts around 0.75 and drifts down toward 0.65. More trials, worse decisions.

---

## What the shading tells you

Narrow shading: the 60 runs agree. The behaviour is consistent.

Wide shading: the runs disagree. Some seeds learned well, some didn't. The model is unreliable — its performance depends on luck.

---

## What "solid = solo, dashed = peers" means

Solo: the model decides alone. No other input.

Peers: the model sees what other models decided before making its own choice. Social information.

The gap between solid and dashed lines tells you whether social information helps or hurts.

---

## Why it's cumulative

The balanced accuracy at trial 15 uses all decisions from trial 1 through 15. It's not "how well did you do on trial 15" — it's "how well have you done so far."

This smooths out noise. Early trials are jumpy because there's little data. Later trials are stable because the average has settled.

The cost: if a model suddenly gets worse at trial 20, the cumulative curve won't show a sharp drop. It'll just slow its rise, or gently bend down. The cumulative view shows trends, not moments.
