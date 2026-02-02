'''
city/city/__main__.py

Entry point and parser for the city command line.

Created: 2026-02-02
 Author: Maxence Morel Dierckx
'''
import argparse
import json
from pathlib import Path

HERE = Path(__file__).parent.resolve()
ROOT = HERE.parent

# MARK: configure

def configure(args):
    # TODO: interactive config generator
    print(args.file)
    pass

def run(args):
    # TODO: entry point to the full run through (bounded by instance count)
    print(args.config, args.steps, args.debug)
    pass

def main():
    parser = argparse.ArgumentParser(
        prog='city',
        description='A generational simulator for LLM societies'
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    # configure subcommand
    configure_parser = subparsers.add_parser(
        'configure',
        help='Generate a config file with the interactive CLI'
    )
    configure_parser.add_argument(
        '--file', '-f',
        required=True,
        help='Path to config file, new or existing'
    )
    configure_parser.set_defaults(func=configure)

    # run subcommand
    run_parser = subparsers.add_parser(
        'run',
        help='Run a configured city'
    )
    run_parser.add_argument(
        '--config', '-c',
        required=True,
        help='Path to city config file'
    )
    run_parser.add_argument(
        '--steps', '-s',
        type=int,
        default=-1,
        help='Number of steps to run (overrides instance count)'
    )
    run_parser.add_argument(
        '--debug', '-d',
        action='store_true',
        help='Show debug information'
    )
    run_parser.set_defaults(func=run)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
