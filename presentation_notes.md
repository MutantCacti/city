# City — Presentation Notes

## Slide: Confusion Matrices

Three conditions, same model (DeepSeek), same stimulus seed.

**Order 1 Terse** (A→B trigger, one-word response constraint):
- 85% hit rate on ACT trials. Misses the first two A→B transitions, learns by step 9.
- 0% false alarm rate. Never acts when it shouldn't.
- Conservative bias that relaxes with accumulated context.

**Order 2 Reasoning prompt** (A→B→C trigger, step-by-step prompt):
- 100% across the board. The reasoning prompt improves performance even when the model doesn't visibly reason.
- Note: DeepSeek still responds in single tokens despite the "think step by step" instruction. Haiku produces visible chain-of-thought. The prompt variable is not "reasoning" but a *reasoning prompt* — whether the model actually reasons is model-dependent.

**Order 2 Terse** (A→B→C trigger, one-word response constraint):
- 0% hit rate. Never says ACT. Complete failure.
- 0% false alarm rate. Still never acts when it shouldn't.
- The one-word constraint prevents the model from tracking a 3-state sequence.

The only cell that varies is top-left (should ACT, predicted ACT). The model never false-alarms under any condition. Its sole failure mode is missing the trigger — conservative bias, not confusion. The terse constraint is the mechanism: it's equivalent to a button press in the Frick study, where children can't reason aloud mid-trial.

**The question this sets up**: Can social conditions (peers in the space) shift the Order 2 Terse hit rate above 0%?

## Slide: Learning Curve

Running accuracy (correct/total) across trial steps. Dots mark act trials.

**Order 2 Reasoning** (green, dashed): Flat at 1.0 from trial 1. No learning curve — the task is trivially solved with chain-of-thought. This is the ceiling.

**Order 1 Terse** (orange): Starts at 0.5, dips on early missed A→B transitions, then climbs steadily. Reaches ~1.0 by trial 15-20 and stays there. The model learns the rule from context alone, without explicit reasoning. The crossover is visible around step 9.

**Order 2 Terse** (red): Starts at 1.0 and *decays*. Every missed ACT trial drags the running accuracy down. It never recovers because it never learns to ACT. By trial 100, accuracy settles around 0.93 — high overall, but only because ACT trials are rare. The act accuracy is 0%.

**Key point**: Order 1 terse shows a learning curve. Order 2 terse shows a decay curve. The difference is working memory depth — the model can track one transition in context but not two. The running accuracy masks the failure because WAIT trials dominate. The confusion matrix (previous slide) reveals what this graph hides.

