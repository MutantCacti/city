'''
city/city/providers.py

LLM provider classes.

Created: 2026-02-2
 Author: Maxence Morel Dierckx
'''
import os
from abc import ABC, abstractmethod
from typing import Optional
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class Provider(ABC):
    '''Base provider interface'''
    pass

    @abstractmethod
    def __init__(self, model):
        '''Initialise the client'''
        return

    @abstractmethod
    def transform_context(self, context: list[dict]) -> list[dict]:
        '''Create and append a chat completion to passed messages'''
        return  

    @abstractmethod
    def get_name(self) -> str:
        '''Get the provider name'''
        return


class DeepSeekProvider(Provider):
    def __init__(self, model):
        '''Initialise the OpenAI client'''

        base_url = 'https://api.deepseek.com'
        try:
            api_key = os.environ['DEEPSEEK_API_KEY']
        except KeyError as e:
            print(KeyError, e)
            print('DEEPSEEK_API_KEY must be set')
        
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def transform_context(self, context: list[dict]) -> list[dict]:
        response = self.client.chat.completions.create(
            model=self.model,
            thinking='disabled',
            messages=context,
            stream=False
        )
        context.append(response.choices[0].message)
        return context

    def get_name(self) -> str:
        return "deepseek"


PROVIDERS = {
    'deepseek': DeepSeekProvider
}

def get_provider(name: str, model: str) -> Provider:
    return PROVIDERS[name](model)
