'''
city/city/primitives.py

Primitives for the city network.

Created: 2026-02-05
 Author: Maxence Morel Dierckx
'''
from typing import Set


class Message:
    '''A message to or from an LLM instance. Contains a role and content'''

    def __init__(self, role: str, content: str) -> None:
        self.role = role
        self.content = content

    def to_dict(self) -> dict[str, str]:
        return {'role': self.role, 'content': self.content}


class Context:
    '''A list of messages'''

    def __init__(self, initial_prompt: Message | None = None) -> None:
        self.messages = []
        if initial_prompt is not None:
            self.messages.append(initial_prompt)

    def add(self, message: Message) -> None:
        self.messages.append(message)

    def get_last(self) -> Message:
        return self.messages[-1]

    def to_dicts(self) -> list[dict[str, str]]:
        return [msg.to_dict() for msg in self.messages]


class Instance:
    '''A composed model and context constituting an individual in the city'''

    def __init__(self, provider, context: Context | None = None) -> None:
        self.provider = provider
        self.context = context if context is not None else Context()

    def prompt(self, message: Message) -> None:
        self.context.add(message)

    def get_response(self) -> Message:
        '''Get response from provider, converting Context <-> list[dict]'''
        dicts = self.context.to_dicts()
        response = self.provider.transform_context(dicts)
        self.context.add(Message(response['role'], response['content']))
        return response

    def to_dict(self) -> dict:
        return {
            'provider': self.provider.get_name(),
            'model': self.provider.model,
            'context': self.context.to_dicts()
        }


class Space:
    '''A group chat with multiple instances'''

    def __init__(self, name: str):
        self.name = name
        self.instances: Set[Instance] = set()

    def add_instance(self, instance: Instance):
        self.instances.add(instance)

    def add_instances(self, instances: list[Instance] | Set[Instance]):
        self.instances.update(instances)

    def remove_instance(self, instance: Instance):
        try:
            self.instances.remove(instance)
        except KeyError:
            return

    def clear(self):
        self.instances.clear()
