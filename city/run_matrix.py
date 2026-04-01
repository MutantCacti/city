'''
city/run_matrix.py

Run the full experiment matrix: all configs × N seeds × provider.
Supports parallel seed execution for throughput.

Usage:
    python -m city.run_matrix --provider deepseek --seeds 12
    python -m city.run_matrix --provider local --model phi-4-mini --seeds 12 --parallel 3
    python -m city.run_matrix --configs example/matrix/o2_terse_solo.json --seeds 5
'''
import asyncio
import argparse
import glob
import json
import os

from dotenv import load_dotenv
load_dotenv()
import sys
import time
from pathlib import Path

from city.experiment import ExperimentConfig, run_experiment


async def run_seed(config_path: str, seed: int, provider: str, model: str, base_url: str = None):
    '''Run one seed of one config.'''
    config = ExperimentConfig.from_file(config_path)
    config.seed = seed
    config.provider_name = provider
    config.model_name = model
    if base_url:
        config.base_url = base_url
    if provider == 'local':
        config.tick_seconds = 0.0
    elif provider == 'anthropic':
        config.tick_seconds = 1.2

    try:
        result = await run_experiment(config)
        return result
    except Exception as e:
        print(f'  ERROR seed={seed}: {e}', flush=True)
        return None


async def run_config(config_path: str, seeds: int, start_seed: int, provider: str,
                     model: str, parallel: int, base_url: str = None):
    '''Run all seeds for one config, with parallelism.'''
    name = Path(config_path).stem
    seed_range = f'{start_seed}-{start_seed + seeds - 1}'
    print(f'\n{"="*60}', flush=True)
    print(f'  {name}  ({provider}/{model}, seeds {seed_range}, parallel={parallel})', flush=True)
    print(f'{"="*60}', flush=True)

    semaphore = asyncio.Semaphore(parallel)

    async def bounded_run(seed):
        async with semaphore:
            return await run_seed(config_path, seed, provider, model, base_url)

    tasks = [bounded_run(seed) for seed in range(start_seed, start_seed + seeds)]
    results = await asyncio.gather(*tasks)

    completed = sum(1 for r in results if r is not None)
    print(f'\n  {name}: {completed}/{seeds} seeds completed', flush=True)
    return results


async def main():
    parser = argparse.ArgumentParser(description='Run full experiment matrix')
    parser.add_argument('--provider', type=str, default='deepseek')
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--base-url', type=str, default=None)
    parser.add_argument('--api-key', type=str, default=None,
                        help='API key or slot number (1/2/3). Slot looks up {PROVIDER}_API_KEY_{N} from .env.')
    parser.add_argument('--seeds', type=int, default=12)
    parser.add_argument('--start-seed', type=int, default=0)
    parser.add_argument('--parallel', type=int, default=1,
                        help='Max concurrent seeds. >1 requires DB isolation (not yet implemented).')
    parser.add_argument('--configs', type=str, nargs='*', default=None,
                        help='Specific config files. Default: all in example/matrix/')
    args = parser.parse_args()

    # Resolve API key from slot number or literal
    if args.api_key:
        if args.api_key in ('1', '2', '3'):
            env_key = f'{args.provider.upper()}_API_KEY_{args.api_key}'
            resolved = os.environ.get(env_key)
            if not resolved:
                print(f'Error: {env_key} not found in environment')
                sys.exit(1)
            os.environ[f'{args.provider.upper()}_API_KEY'] = resolved
            print(f'Using {env_key}')
        else:
            os.environ[f'{args.provider.upper()}_API_KEY'] = args.api_key
            print(f'Using provided API key')

    if args.model is None:
        args.model = {
            'deepseek': 'deepseek-chat',
            'anthropic': 'claude-sonnet-4-5-20250929',
            'local': 'phi-4-mini',
        }.get(args.provider, 'deepseek-chat')

    if args.configs:
        config_files = args.configs
    else:
        config_files = sorted(glob.glob('example/matrix/*.json'))

    if not config_files:
        print('No config files found in example/matrix/')
        sys.exit(1)

    print(f'Experiment matrix: {len(config_files)} conditions × {args.seeds} seeds (start={args.start_seed})')
    print(f'Provider: {args.provider}/{args.model}')
    print(f'Parallelism: {args.parallel}')
    print(f'Total runs: {len(config_files) * args.seeds}')

    start = time.time()

    for config_path in config_files:
        await run_config(
            config_path, args.seeds, args.start_seed, args.provider, args.model,
            args.parallel, args.base_url,
        )

    elapsed = time.time() - start
    print(f'\n{"="*60}')
    print(f'  Matrix complete. {len(config_files) * args.seeds} runs in {elapsed/60:.1f} minutes')
    print(f'{"="*60}')


if __name__ == '__main__':
    asyncio.run(main())
