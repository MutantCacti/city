'''
city/providers.py

LLM provider classes.

Created: 2026-02-02
 Author: Maxence Morel Dierckx
'''
import os
from abc import ABC, abstractmethod
from typing import Optional
from openai import OpenAI
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

class Provider(ABC):
    '''Base provider interface'''

    @abstractmethod
    def __init__(self, model):
        '''Initialise the client'''
        return

    @abstractmethod
    def transform_context(self, context: list[dict]) -> dict:
        '''Create and return a chat completion from passed messages'''
        return

    @abstractmethod
    def get_name(self) -> str:
        '''Get the provider name'''
        return


class DeepSeekProvider(Provider):
    def __init__(self, model):
        '''Initialise the OpenAI client'''

        base_url = 'https://api.deepseek.com'
        api_key = os.environ.get('DEEPSEEK_API_KEY')
        if not api_key:
            raise ValueError('DEEPSEEK_API_KEY must be set')

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def transform_context(self, context: list[dict]) -> dict:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=context,
            stream=False
        )
        msg = response.choices[0].message
        return {'role': msg.role, 'content': msg.content}

    def get_name(self) -> str:
        return "deepseek"


class AnthropicProvider(Provider):
    def __init__(self, model):
        '''Initialise the Anthropic client'''

        api_key = os.environ.get('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError('ANTHROPIC_API_KEY must be set')

        self.client = Anthropic()
        self.model = model

    def transform_context(self, context: list[dict]) -> dict:
        system = None
        messages = context
        if context[0]['role'] == 'system':
            system = context[0]['content']
            messages = context[1:]

        kwargs = dict(model=self.model, messages=messages, max_tokens=1024)
        if system:
            kwargs['system'] = system

        response = self.client.messages.create(**kwargs)
        return {'role': 'assistant', 'content': response.content[0].text}

    def get_name(self) -> str:
        return "anthropic"


PROVIDERS = {
    'deepseek': DeepSeekProvider,
    'anthropic': AnthropicProvider,
}

def get_provider(name: str, model: str) -> Provider:
    return PROVIDERS[name](model)
