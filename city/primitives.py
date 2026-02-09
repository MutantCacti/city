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


class Instance:
    '''A composed model and context constituting an individual in the city'''

    def __init__(self, provider, context: list[Message] | None = None) -> None:
        self.provider = provider
        self.context = context if context is not None else []

    def prompt(self, message: Message) -> None:
        self.context.append(message)

    def get_response(self) -> Message:
        dicts = [msg.to_dict() for msg in self.context]
        response = self.provider.transform_context(dicts)
        self.context.append(Message(response['role'], response['content']))
        return response

    def to_dict(self) -> dict:
        return {
            'provider': self.provider.get_name(),
            'model': self.provider.model,
            'context': [msg.to_dict() for msg in self.context]
        }


class Space:
    '''A group chat with multiple instances'''

    def __init__(self, name: str) -> None:
        self.name = name
        self.instances: list[Instance] = []
        self.pointers: dict[Instance, int] = {}
        self.chat: list[tuple[Instance, Message]] = []

    '''Instance management'''

    def add_instance(self, instance: Instance) -> None:
        self.instances.append(instance)
        self.pointers[instance] = 0

    def add_instances(self, instances: list[Instance] | Set[Instance]) -> None:
        for instance in instances:
            self.add_instance(instance)

    def remove_instance(self, instance: Instance) -> None:
        if instance not in self.instances:
            return
        self.instances.remove(instance)
        self.pointers.pop(instance)

    '''Chat management'''

    def add_message(self, message: Message, instance: Instance) -> None:
        assert instance in self.instances
        entry = (instance, message)
        self.chat.append(entry)
    
    def read_messages(self, instance: Instance) -> list[tuple[Instance, Message]]:
        pointer = self.pointers.get(instance)
        return self.chat[pointer:]
    
    def advance_pointer(self, instance: Instance) -> None:
        self.pointers[instance] = len(self.chat)
