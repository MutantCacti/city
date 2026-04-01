'''
city/llm/server.py

OpenAI-compatible inference server for City.
Single file, Starlette + uvicorn, backed by llama_cpp.

Usage:
    cd ~/city && python llm/server.py
    cd ~/city && python llm/server.py --port 8126 --workers 2
    cd ~/city && python llm/server.py --model /path/to/model.gguf
'''
import asyncio
import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from llama_cpp import Llama

DEFAULT_MODEL = Path.home() / 'sg-code/llm/model/microsoft_Phi-4-mini-instruct-Q3_K_M.gguf'
DEFAULT_PORT = 8126
DEFAULT_CTX = 4096
DEFAULT_WORKERS = 2

llm = None
executor = None


def load_model(model_path: str, n_ctx: int = DEFAULT_CTX) -> Llama:
    print(f'Loading model: {model_path}')
    start = time.time()
    model = Llama(model_path=str(model_path), n_ctx=n_ctx, n_gpu_layers=-1, verbose=False)
    print(f'Model loaded in {time.time() - start:.1f}s')
    return model


async def health(request: Request) -> JSONResponse:
    return JSONResponse({'status': 'ok', 'model': llm.model_path if llm else None})


async def chat_completions(request: Request) -> JSONResponse:
    body = await request.json()
    messages = body.get('messages', [])
    model_name = body.get('model', 'phi-4-mini')

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        executor,
        lambda: llm.create_chat_completion(messages=messages),
    )

    return JSONResponse(result)


routes = [
    Route('/', health, methods=['GET']),
    Route('/v1/chat/completions', chat_completions, methods=['POST']),
]

app = Starlette(routes=routes)


def main():
    global llm, executor

    parser = argparse.ArgumentParser(description='City LLM inference server')
    parser.add_argument('--model', type=str, default=str(DEFAULT_MODEL))
    parser.add_argument('--port', type=int, default=DEFAULT_PORT)
    parser.add_argument('--n-ctx', type=int, default=DEFAULT_CTX)
    parser.add_argument('--workers', type=int, default=DEFAULT_WORKERS)
    args = parser.parse_args()

    llm = load_model(args.model, args.n_ctx)
    executor = ThreadPoolExecutor(max_workers=args.workers)

    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=args.port)


if __name__ == '__main__':
    main()
