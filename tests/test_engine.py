'''
tests/test_engine.py

Tests for the simulation engine.
'''
import pytest
import pytest_asyncio
from unittest.mock import MagicMock

from city.db import (
    init_db, get_session_maker,
    SpaceService, InstanceService, MessageService, ProviderService,
)
from city.engine import run_turn, TurnResult


@pytest_asyncio.fixture
async def db():
    '''Fresh in-memory database per test.'''
    await init_db()
    session_maker = get_session_maker()
    async with session_maker() as session:
        yield session


@pytest_asyncio.fixture
async def setup(db):
    '''Create a provider, instance, and space. Return their IDs.'''
    provider = await ProviderService.create_provider(db, 'deepseek', 'deepseek-chat')
    instance = await InstanceService.create_instance(db, provider.provider_id)
    space = await SpaceService.create_space(db, 'test-space')
    await SpaceService.add_instance_to_space(db, space.space_id, instance.instance_id)

    # Add system prompt to instance context
    sys_msg = await MessageService.create_message(
        db, instance.instance_id, 'system', 'You are a test instance.'
    )
    await InstanceService.add_message_to_instance(db, instance.instance_id, sys_msg.message_id)

    return {
        'provider_id': provider.provider_id,
        'instance_id': instance.instance_id,
        'space_id': space.space_id,
    }


@pytest.mark.asyncio
async def test_run_turn_returns_turn_result(db, setup, monkeypatch):
    '''run_turn returns a TurnResult with response.'''
    instance_id = setup['instance_id']
    space_id = setup['space_id']

    # Post a message to the space
    msg = await MessageService.create_message(db, instance_id, 'user', 'State: A')
    await SpaceService.add_message_to_space(db, space_id, msg.message_id, instance_id)

    # Mock the provider
    mock_provider = MagicMock()
    mock_provider.transform_context.return_value = {'role': 'assistant', 'content': '[WAIT]'}

    import city.engine
    original_get_provider = city.engine.get_provider
    monkeypatch.setattr(city.engine, 'get_provider', lambda name, model, **kw: mock_provider)

    result = await run_turn(db, instance_id, space_id)

    assert isinstance(result, TurnResult)
    assert result.instance_id == instance_id
    assert result.space_id == space_id
    assert result.error is None
    assert result.response is not None
    assert result.response['content'] == '[WAIT]'
    assert len(result.messages_read) == 1
    assert result.messages_read[0]['content'] == 'State: A'


@pytest.mark.asyncio
async def test_run_turn_advances_pointer(db, setup, monkeypatch):
    '''After run_turn, reading again returns no unread messages.'''
    instance_id = setup['instance_id']
    space_id = setup['space_id']

    msg = await MessageService.create_message(db, instance_id, 'user', 'State: B')
    await SpaceService.add_message_to_space(db, space_id, msg.message_id, instance_id)

    mock_provider = MagicMock()
    mock_provider.transform_context.return_value = {'role': 'assistant', 'content': '[ACT]'}

    import city.engine
    monkeypatch.setattr(city.engine, 'get_provider', lambda name, model, **kw: mock_provider)

    await run_turn(db, instance_id, space_id)

    # No unread messages after turn
    unread = await SpaceService.get_unread_messages(db, space_id, instance_id)
    assert len(unread) == 0


@pytest.mark.asyncio
async def test_run_turn_captures_error(db, setup, monkeypatch):
    '''Provider failure is captured in TurnResult.error, not raised.'''
    instance_id = setup['instance_id']
    space_id = setup['space_id']

    msg = await MessageService.create_message(db, instance_id, 'user', 'State: A')
    await SpaceService.add_message_to_space(db, space_id, msg.message_id, instance_id)

    mock_provider = MagicMock()
    mock_provider.transform_context.side_effect = RuntimeError('API timeout')

    import city.engine
    monkeypatch.setattr(city.engine, 'get_provider', lambda name, model, **kw: mock_provider)

    result = await run_turn(db, instance_id, space_id)

    assert result.error == 'API timeout'
    assert result.response is None


@pytest.mark.asyncio
async def test_run_turn_no_unread(db, setup, monkeypatch):
    '''run_turn with no unread messages still calls provider with existing context.'''
    instance_id = setup['instance_id']
    space_id = setup['space_id']

    mock_provider = MagicMock()
    mock_provider.transform_context.return_value = {'role': 'assistant', 'content': 'Hello'}

    import city.engine
    monkeypatch.setattr(city.engine, 'get_provider', lambda name, model, **kw: mock_provider)

    result = await run_turn(db, instance_id, space_id)

    assert result.error is None
    assert len(result.messages_read) == 0
    assert result.response['content'] == 'Hello'
