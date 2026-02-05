'''
city/tests/test_types.py

Automated unit tests for type primitives.

Created: 2026-02-05
'''
import pytest
from unittest.mock import MagicMock
from city.primitives import Message, Instance, Space
from city.providers import get_provider


class TestMessage:
    '''Test the Message class'''

    def test_message_init(self):
        msg = Message('user', 'Hello')
        assert msg.role == 'user'
        assert msg.content == 'Hello'

    def test_message_to_dict(self):
        msg = Message('assistant', 'Hi there')
        assert msg.to_dict() == {'role': 'assistant', 'content': 'Hi there'}


class TestInstance:
    '''Test the Instance class'''

    def test_instance_init_with_context(self):
        provider = get_provider('deepseek', 'deepseek-chat')
        ctx = [Message('system', 'You exist.')]
        instance = Instance(provider, ctx)
        assert instance.context is ctx
        assert instance.provider is provider

    def test_instance_init_no_context(self):
        provider = get_provider('deepseek', 'deepseek-chat')
        instance = Instance(provider)
        assert instance.context is not None
        assert len(instance.context) == 0

    def test_instance_prompt(self):
        provider = get_provider('deepseek', 'deepseek-chat')
        instance = Instance(provider)
        instance.prompt(Message('user', 'Hello'))
        assert len(instance.context) == 1
        assert instance.context[-1].content == 'Hello'

    def test_instance_get_response(self):
        provider = get_provider('deepseek', 'deepseek-chat')
        ctx = [Message('system', 'You are helpful.'), Message('user', 'Hi')]
        instance = Instance(provider, ctx)

        # Mock the provider
        mock_msg = MagicMock()
        mock_msg.role = 'assistant'
        mock_msg.content = 'Hello!'
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=mock_msg)]
        instance.provider.client.chat.completions.create = MagicMock(return_value=mock_response)

        response = instance.get_response()
        assert response['role'] == 'assistant'
        assert response['content'] == 'Hello!'
        # Check it was added to context
        assert instance.context[-1].content == 'Hello!'

    def test_instance_to_dict(self):
        provider = get_provider('deepseek', 'deepseek-chat')
        ctx = [Message('user', 'Test')]
        instance = Instance(provider, ctx)
        d = instance.to_dict()
        assert d['provider'] == 'deepseek'
        assert d['model'] == 'deepseek-chat'
        assert d['context'] == [{'role': 'user', 'content': 'Test'}]


class TestSpace:
    '''Test the Space class'''

    def test_space_init(self):
        space = Space('town_square')
        assert space.name == 'town_square'
        assert len(space.instances) == 0

    def test_space_add_instance(self):
        space = Space('test')
        provider = get_provider('deepseek', 'deepseek-chat')
        instance = Instance(provider)
        space.add_instance(instance)
        assert len(space.instances) == 1
        assert instance in space.instances

    def test_space_add_instances(self):
        space = Space('test')
        provider = get_provider('deepseek', 'deepseek-chat')
        i1 = Instance(provider)
        i2 = Instance(provider)
        space.add_instances([i1, i2])
        assert len(space.instances) == 2

    def test_space_remove_instance(self):
        space = Space('test')
        provider = get_provider('deepseek', 'deepseek-chat')
        instance = Instance(provider)
        space.add_instance(instance)
        space.remove_instance(instance)
        assert len(space.instances) == 0

    def test_space_remove_nonexistent(self):
        space = Space('test')
        provider = get_provider('deepseek', 'deepseek-chat')
        instance = Instance(provider)
        # Should not raise
        space.remove_instance(instance)

    def test_space_clear(self):
        space = Space('test')
        provider = get_provider('deepseek', 'deepseek-chat')
        space.add_instance(Instance(provider))
        space.add_instance(Instance(provider))
        space.clear()
        assert len(space.instances) == 0
