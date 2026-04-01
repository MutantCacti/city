'''
city/engine.py

Simulation engine. The core turn loop that everything else composes.

Created: 2026-03-22
'''
import asyncio
from dataclasses import dataclass
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from city.db import (
    SpaceService, InstanceService, MessageService,
    Message as DbMessage,
)
from city.providers import get_provider


class CreditError(Exception):
    '''Raised when the API rejects a request due to insufficient credit.'''
    pass


@dataclass
class TurnResult:
    '''The result of a single turn of an instance in a space.'''
    instance_id: int
    space_id: int
    messages_read: list[dict]  # [{'message_id', 'instance_id', 'role', 'content'}]
    response: Optional[dict] = None  # {'message_id', 'role', 'content'}
    error: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0


async def run_turn(db: AsyncSession, instance_id: int, space_id: int, base_url: str = None) -> TurnResult:
    '''
    Run one turn: read unread messages, build context, call provider, post response.

    This is the atomic unit of simulation. The experiment wraps it with
    stimulus + scoring. The full simulation loop wraps it with routing + lifecycle.
    '''
    # 1. Read unread messages from space
    unread = await SpaceService.get_unread_messages(db, space_id, instance_id)
    messages_read = [
        {'message_id': m.message_id, 'instance_id': m.instance_id,
         'role': m.role, 'content': m.content}
        for m in unread
    ]

    # 2. Add unread messages to instance context
    # Messages from other instances are stored as 'user' role to preserve
    # the conversation structure (only this instance's responses are 'assistant')
    for m in unread:
        if m.instance_id != instance_id and m.role == 'assistant':
            copy = await MessageService.create_message(db, instance_id, 'user', m.content)
            await InstanceService.add_message_to_instance(db, instance_id, copy.message_id)
        else:
            await InstanceService.add_message_to_instance(db, instance_id, m.message_id)

    # 3. Get full context and call provider
    instance = await InstanceService.get_instance_by_id(db, instance_id)
    if not instance:
        return TurnResult(instance_id, space_id, messages_read, error='Instance not found')

    provider = get_provider(instance.provider.provider_name, instance.provider.model_name, base_url=base_url)
    context = await InstanceService.get_context(db, instance_id)
    context_dicts = [{'role': m.role, 'content': m.content} for m in context]

    try:
        response_dict = await asyncio.to_thread(provider.transform_context, context_dicts)
    except Exception as e:
        error_str = str(e).lower()
        if 'credit' in error_str or 'balance' in error_str or 'billing' in error_str:
            raise CreditError(str(e)) from e
        return TurnResult(instance_id, space_id, messages_read, error=str(e))

    # 4. Create response message, add to instance context and space
    response_msg = await MessageService.create_message(
        db, instance_id, response_dict['role'], response_dict['content']
    )
    await InstanceService.add_message_to_instance(db, instance_id, response_msg.message_id)
    await SpaceService.add_message_to_space(db, space_id, response_msg.message_id, instance_id)

    # 5. Advance read pointer
    await SpaceService.advance_pointer(db, space_id, instance_id)

    return TurnResult(
        instance_id=instance_id,
        space_id=space_id,
        messages_read=messages_read,
        response={
            'message_id': response_msg.message_id,
            'role': response_msg.role,
            'content': response_msg.content,
        },
        prompt_tokens=response_dict.get('prompt_tokens', 0),
        completion_tokens=response_dict.get('completion_tokens', 0),
    )
