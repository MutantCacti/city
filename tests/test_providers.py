'''
city/tests/test_providers.py

Automated unit tests for provider classes.

Created: 2026-02-02
 Author: Maxence Morel Dierckx
'''
import pytest
from city.providers import get_provider


class TestDeepSeekProvider:
    '''Test the DeepSeek provider'''

    def test_deepseek_provider_init(self):
        deepseek_provider = get_provider('deepseek', 'deepseek-chat')
        assert deepseek_provider.client != None
        assert deepseek_provider.client.websocket_base_url == 'http://api.deepseek.com'

    def test_deepseek_provider_transform(self):
        deepseek_provider = get_provider('deepseek', 'deepseek-chat')
        # TODO
        pass

    def test_deepseek_provider_name(self):
        # TODO
        pass
