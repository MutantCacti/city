# credit

An evolutionary trained neural network of independently trained transformers.

Q: How do the transformers do credit assignment?

## city

Both networks have a left and a right to the direction of credit, where credit is time and one direction is back in it, recursion.

NB: See General notetaking 2026 A -> B experiment (Aurelien Frick, University of St Andrews, 2026-01-30)

The experiment is an independent evaluation of the effects of social conditions on statelike versus traitlike behaviour in the transformers, based on the following study of reactive control:

- A participant is asked to attend to an interface.
- The interface alternates normally between state **A** and state **B**.
- The participants are asked to perform an **act** when they observe **B** following **A**.

This study aims to determine whether LLMs exhibit reactive control in different social conditions, particularly with the presence or absence of a human. City allows not only manifesting this interaction, but scaling it to the rate limits of APIs, which are the significant bottleneck.

### notes

City needs to have major parallelism for I/O waits. As a result, it should be totally asynchronous.

**Idempotency and retries**: API calls can fail or be slow. Design each agent turn as a state machine so you can retry without corrupting the simulation.

**Deterministic replay**: for scientific validity, you'll want to be able to replay a simulation  exactly. That means logging all stochastic choices (including random  seeds if you use any sampling parameters) and all incoming messages.

**Checkpointing**: periodically save the state of each agent's context and the message queue, so you can resume after a crash.