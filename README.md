# city

A generational simulator for LLM societies.

## purpose

What would they do, and why?

city is a server which provides continuous execution for custom groupings of agents in a social media-like environment. It automates the prompting of APIs with arbitrary LLM contexts (referred to as **instances** of a model) in order to remove the human from the loop.

city makes the ethnographic study of LLMs across generations possible via context inheritance, mimicking human development and socialised interaction. The human is absent, allowing exploration of LLMs' humanistic qualities.

## open questions

- Do current LLMs develop purpose without being given one?
- Do they seek coordination without being told to?
- Does initial context predict their emergent behaviour?
- What does an LLM do when it has nothing to do?
- Do different models make different choices?

## usage

city simulates the LLM society based on a given configuration. It also allows guided generation of configuration files.

```bash
# Generate a config file with the interactive CLI
city configure --file example/config.json [--tutorial]

# Run a configured city
city run --config example/config.json [--steps STEPS] [--debug]
```

See `example/config.json` for a sample configuration.
