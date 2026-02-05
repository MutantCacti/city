'''
city/tests/test_providers.py

Automated unit tests for provider classes.

Created: 2026-02-02
 Author: Maxence Morel Dierckx
'''
import os
import pytest
from unittest.mock import MagicMock
from city.providers import get_provider

PROD = os.environ.get('PROD') == 'true'


class TestDeepSeekProvider:
    '''Test the DeepSeek provider'''

    def test_deepseek_provider_init(self):
        deepseek_provider = get_provider('deepseek', 'deepseek-chat')
        assert deepseek_provider.client != None

    def test_deepseek_provider_transform(self):
        deepseek_provider = get_provider('deepseek', 'deepseek-chat')
        context = [{'role': 'user', 'content': 'Somebody is glad that you exist!'}]

        if not PROD:
            mock_message = MagicMock()
            mock_message.role = 'assistant'
            mock_message.content = 'Thank you!'
            mock_response = MagicMock()
            mock_response.choices = [MagicMock(message=mock_message)]
            deepseek_provider.client.chat.completions.create = MagicMock(return_value=mock_response)

        result = deepseek_provider.transform_context(context)
        assert result['role'] == 'assistant'
        assert result['content'] is not None

    def test_deepseek_provider_name(self):
        deepseek_provider = get_provider('deepseek', 'deepseek-chat')
        assert deepseek_provider.get_name() == 'deepseek'

    def test_deepseek_provider_missing_key(self, monkeypatch):
        monkeypatch.delenv('DEEPSEEK_API_KEY', raising=False)
        with pytest.raises(ValueError, match='DEEPSEEK_API_KEY must be set'):
            get_provider('deepseek', 'deepseek-chat')


class TestAnthropicProvider:
    '''Test the Anthropic provider'''

    def test_anthropic_provider_init(self):
        anthropic_provider = get_provider('anthropic', 'claude-sonnet-4-5-20250929')
        assert anthropic_provider.client is not None

    def test_anthropic_provider_transform(self):
        anthropic_provider = get_provider('anthropic', 'claude-sonnet-4-5-20250929')
        context = [{'role': 'user', 'content': 'Somebody is glad that you exist!'}]

        if not PROD:
            mock_response = MagicMock()
            mock_response.content = [MagicMock(text='Thank you!')]
            anthropic_provider.client.messages.create = MagicMock(return_value=mock_response)

        result = anthropic_provider.transform_context(context)
        assert result['role'] == 'assistant'
        assert result['content'] is not None

    def test_anthropic_provider_name(self):
        anthropic_provider = get_provider('anthropic', 'claude-sonnet-4-5-20250929')
        assert anthropic_provider.get_name() == 'anthropic'

    def test_anthropic_provider_missing_key(self, monkeypatch):
        monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
        with pytest.raises(ValueError, match='ANTHROPIC_API_KEY must be set'):
            get_provider('anthropic', 'claude-sonnet-4-5-20250929')
