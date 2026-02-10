'''
city/db.py

Service layer for the relational database.

Created: 2026-02-10
 Author: Maxence Morel Dierckx
'''
import asyncio
from sqlalchemy import text, select, Column, Integer, String, Text, ForeignKey, Table
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import relationship, DeclarativeBase


# MARK: Init

_ENGINE = None
_SESSION_MAKER = None


class Base(DeclarativeBase):
    pass


def get_engine():
    '''Get an async engine'''
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = create_async_engine(
            'sqlite+aiosqlite:///:memory:',
            echo=False,
            future=True
        )
    return _ENGINE


def get_session_maker():
    '''Get an async session maker'''
    global _SESSION_MAKER
    if _SESSION_MAKER is None:
        _SESSION_MAKER = async_sessionmaker(
            get_engine(),
            expire_on_commit=False,
            class_=AsyncSession,
        )
    return _SESSION_MAKER


async def get_db() -> AsyncSession:
    '''Dependency for stores'''
    session_maker = get_session_maker()
    async with session_maker() as session:
        yield session


async def init_db() -> None:
    '''Called on startup. Creates schema'''
    async with get_session_maker()() as session:
        conn = await session.connection()
        await conn.run_sync(Base.metadata.create_all)


# MARK: Models

class Space(Base):
    __tablename__ = 'spaces'

    space_id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(256))

    messages = relationship('Message', secondary=space_messages, back_populates='spaces', lazy='selectin')
    instances = relationship('Instance', secondary=space_instances, back_populates='spaces', lazy='selectin')


class Instance(Base):
    __tablename__ = 'instances'

    instance_id = Column(Integer, primary_key=True, autoincrement=True)
    provider_id = Column(Integer, ForeignKey('providers.provider_id'))

    messages = relationship('Message', secondary=instance_messages, back_populates='instances', lazy='selectin')
    provider = relationship('Provider', back_populates='instances', lazy='selectin')


class Message(Base):
    __tablename__ = 'messages'

    message_id = Column(Integer, primary_key=True, autoincrement=True)
    role = Column(String(16), nullable=False)
    content = Column(Text, nullable=False)

    spaces = relationship('Space', secondary=space_messages, back_populates='messages', lazy='selectin')
    instances = relationship('Instance', secondary=
    instance_messages, back_populates='messages', lazy='selectin')


class Provider(Base):
    __tablename__ = 'providers'

    provider_id = Column(Integer, primary_key=True, autoincrement=True)
    provider_name = Column(String(64), nullable=False)
    model_name = Column(String(64), nullable=False)

    instances = relationship('Instance', back_populates='provider', lazy='selectin')
