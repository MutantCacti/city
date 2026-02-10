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
    '''Get an async engine.'''
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = create_async_engine(
            'sqlite+aiosqlite:///:memory:',
            echo=False,
            future=True
        )
    return _ENGINE


def get_session_maker():
    '''Get an async session maker.'''
    global _SESSION_MAKER
    if _SESSION_MAKER is None:
        _SESSION_MAKER = async_sessionmaker(
            get_engine(),
            expire_on_commit=False,
            class_=AsyncSession,
        )
    return _SESSION_MAKER


async def get_db() -> AsyncSession:
    '''Dependency for stores.'''
    session_maker = get_session_maker()
    async with session_maker() as session:
        yield session


async def init_db() -> None:
    '''Called on startup. Creates schema.'''
    async with get_session_maker()() as session:
        conn = await session.connection()
        await conn.run_sync(Base.metadata.create_all)


# MARK: Junctions

space_messages = Table(
    'space_messages',
    Base.metadata,
    Column('space_id', ForeignKey('spaces.space_id'), primary_key=True),
    Column('message_id', ForeignKey('messages.message_id'), primary_key=True)
)

space_instances = Table(
    'space_instances',
    Base.metadata,
    Column('space_id', ForeignKey('spaces.space_id'), primary_key=True),
    Column('instance_id', ForeignKey('instances.instance_id'), primary_key=True),
    Column('message_id', ForeignKey('messages.message_id'), nullable=True)
)

instance_messages = Table(
    'instance_messages',
    Base.metadata,
    Column('instance_id', ForeignKey('instances.instance_id'), primary_key=True),
    Column('message_id', ForeignKey('messages.message_id'), primary_key=True)
)


# MARK: Models

class Space(Base):
    __tablename__ = 'spaces'

    space_id = Column(Integer, primary_key=True, autoincrement=True)
    space_name = Column(String(256))

    messages = relationship('Message', secondary=space_messages, back_populates='spaces', lazy='selectin')
    instances = relationship('Instance', secondary=space_instances, back_populates='spaces', lazy='selectin')


class Instance(Base):
    __tablename__ = 'instances'

    instance_id = Column(Integer, primary_key=True, autoincrement=True)
    provider_id = Column(Integer, ForeignKey('providers.provider_id'), nullable=False)

    messages = relationship('Message', secondary=instance_messages, back_populates='instances', lazy='selectin')
    provider = relationship('Provider', back_populates='instances', lazy='selectin')


class Message(Base):
    __tablename__ = 'messages'

    message_id = Column(Integer, primary_key=True, autoincrement=True)
    instance_id = Column(Integer, ForeignKey('instances.instance_id'))
    role = Column(String(16), nullable=False)
    content = Column(Text, nullable=False)

    spaces = relationship('Space', secondary=space_messages, back_populates='messages', lazy='selectin')
    instances = relationship('Instance', secondary=instance_messages, back_populates='messages', lazy='selectin')


class Provider(Base):
    __tablename__ = 'providers'

    provider_id = Column(Integer, primary_key=True, autoincrement=True)
    provider_name = Column(String(64), nullable=False)
    model_name = Column(String(64), nullable=False)

    instances = relationship('Instance', back_populates='provider', lazy='selectin')


# MARK: Space

class SpaceService:
    '''The service class for Spaces.'''

    @staticmethod
    async def create_space(db: AsyncSession, name: str) -> Space:
        '''Create a Space.'''
        try:
            space = Space(name=name)
            db.add(space)
            await db.commit()
            await db.refresh(space)
            return space
        except Exception:
            await db.rollback()
            raise
    
    
    @staticmethod
    async def get_space_by_id(db: AsyncSession, space_id: int) -> Space | None:
        '''Get a Space by its ID.'''
        space = await db.execute(
            select(Space)
            .where(Space.space_id == space_id)
        )
        return space.scalar_one_or_none()
    

    @staticmethod
    async def get_space_by_name(db: AsyncSession, space_name: str) -> Space | None:
        '''Get a Space by its name.'''
        space = await db.execute(
            select(Space)
            .where(Space.space_name == space_name)
        )
        return space.scalar_one_or_none()

    
    @staticmethod
    async def get_message_by_instance_id(db: AsyncSession, space_id: int, instance_id: int) -> Message | None:
        '''Find a Space and Instance by their IDs and get the last read Message by that Instance.'''
        message = await db.execute(
            select(Message)
            .join(space_instances, space_instances.c.message_id == Message.message_id)
            .where(space_instances.c.space_id == space_id)
            .where(space_instances.c.instance_id == instance_id)
        )
        return message.scalar_one_or_none()
            
    
    @staticmethod
    async def update_space(db: AsyncSession, space_id: int, space_name: str) -> Space | None:
        '''Find a Space by ID and update its name.'''
        try:
            space = await SpaceService.get_space_by_id(db, space_id)
            if not space:
                return None

            space.space_name = space_name
            await db.commit()
            await db.refresh(space)
            return space
        except Exception:
            await db.rollback()
            raise
    

    @staticmethod
    async def delete_space(db: AsyncSession, space_id: int) -> bool:
        '''Find a Space by ID and delete it. Returns True if deleted.'''
        try:
            space = await SpaceService.get_space_by_id(db, space_id)
            if not space:
                return False
            await db.delete(space)
            await db.commit()
            return True
        except Exception:
            await db.rollback()
            raise

    
    @staticmethod
    async def add_message_to_space(db: AsyncSession, space_id: int, message_id: int, instance_id: int) -> bool:
        '''Find a Space and Message by their IDs and add the message to the space. Returns true if added.'''
        try:
            space = await SpaceService.get_space_by_id(db, space_id)
            if not space:
                return False
            message = await MessageService.get_message_by_id(db, message_id)
            if not message:
                return False
            
            space.messages.append(message)
            await db.commit()
            return True
        except Exception:
            await db.rollback()
            raise


    @staticmethod
    async def add_instance_to_space(db: AsyncSession, space_id: int, instance_id: int) -> bool:
        '''Find a Space and Instance by their IDs and add the instance to the space. Returns true if added.'''
        try:
            space = await SpaceService.get_space_by_id(db, space_id)
            if not space:
                return False
            instance = await InstanceService.get_instance_by_id(db, instance_id)
            if not instance:
                return False

            space.instances.append(instance)
            await db.commit()
            return True
        except Exception:
            await db.rollback()
            raise
    

# MARK: Instance

class InstanceService:
    '''The service class for Instances.'''
    
    @staticmethod
    async def create_instance(db: AsyncSession, provider_id: int) -> Instance:
        '''Create an Instance.'''
        try:
            instance = Instance(provider_id=provider_id)
            db.add(instance)
            await db.commit()
            await db.refresh(instance)
            return instance
        except Exception:
            await db.rollback()
            raise
    

    @staticmethod
    async def get_instance_by_id(db: AsyncSession, instance_id: int) -> Instance | None:
        '''Get an Instance by its ID.'''
        instance = await db.execute(select(Instance).where(Instance.instance_id == instance_id))
        return instance.scalar_one_or_none()
    

    @staticmethod
    async def update_instance(db: AsyncSession, instance_id: int, provider_id: int) -> Instance | None:
        '''Find an Instance by ID and update its provider ID.'''
        try:
            instance = await InstanceService.get_instance_by_id(db, instance_id)
            if not instance:
                return None

            instance.provider_id = provider_id
            await db.commit()
            await db.refresh(instance)
            return instance
        except Exception:
            await db.rollback()
            raise
    

    @staticmethod
    async def delete_instance(db: AsyncSession, instance_id: int) -> bool:
        '''Find an Instance by ID and delete it. Returns True if deleted.'''
        try:
            instance = await InstanceService.get_instance_by_id(db, instance_id)
            if not instance:
                return False
            
            await db.delete(instance)
            await db.commit()
            return True
        except Exception:
            await db.rollback()
            raise
    

    @staticmethod
    async def add_message_to_instance(db: AsyncSession, instance_id: int, message_id: int) -> bool:
        '''Find an Instance by ID and add a message to its context. Returns True if added.'''
        try:
            instance = await InstanceService.get_instance_by_id(db, instance_id)
            if not instance:
                return False
            message = await MessageService.get_message_by_id(db, message_id)
            if not message:
                return False

            instance.messages.append(message)
            await db.commit()
            return True
        except Exception:
            await db.rollback()
            raise


# MARK: Message

class MessageService:
    '''The service class for Messages.'''

    @staticmethod
    async def create_message(db: AsyncSession, instance_id: int, role: str, content: str) -> Message:
        '''Create a Message.'''
        try:
            message = Message(instance_id=instance_id, role=role, content=content)
            db.add(message)
            await db.commit()
            await db.refresh(message)
            return message
        except Exception:
            await db.rollback()
            raise
    

    @staticmethod
    async def get_message_by_id(db: AsyncSession, message_id: int) -> Message | None:
        '''Get a Message by its ID.'''
        message = await db.execute(select(Message).where(Message.message_id == message_id))
        return message.scalar_one_or_none()
    

    @staticmethod
    async def update_message(db: AsyncSession, message_id: int, instance_id: int | None, role: str | None, content: str | None) -> Message | None:
        '''Find a Message by ID and update its instance_id, role, or content.'''
        try:
            message = await MessageService.get_message_by_id(db, message_id)
            if not message:
                return None
            
            if instance_id:
                message.instance_id = instance_id
            if role:
                message.role = role
            if content:
                message.content = content
            await db.commit()
            await db.refresh(message)
            return message
        except Exception:
            await db.rollback()
            raise
    

    @staticmethod
    async def delete_message(db: AsyncSession, message_id: int) -> bool:
        '''Find a Message by its ID and delete it. Returns True if deleted.'''
        try:
            message = await MessageService.get_message_by_id(db, message_id)
            if not message:
                return False
            
            await db.delete(message)
            await db.commit()
            return True
        except Exception:
            await db.rollback()
            raise
    

# MARK: Provider

class ProviderService:
    '''The service class for Providers.'''

    @staticmethod
    async def create_provider(db: AsyncSession, provider_name: str, model_name: str) -> Provider:
        '''Create a Provider.'''
        try:
            provider = Provider(provider_name=provider_name, model_name=model_name)
            db.add(provider)
            await db.commit()
            await db.refresh(provider)
            return provider
        except Exception:
            await db.rollback()
            raise
    

    @staticmethod
    async def get_provider_by_id(db: AsyncSession, provider_id: int) -> Provider | None:
        '''Get a Provider by its ID.'''
        provider = await db.execute(select(Provider).where(Provider.provider_id == provider_id))
        return provider.scalar_one_or_none()
    

    @staticmethod
    async def get_provider_by_names(db: AsyncSession, provider_name: str, model_name: str) -> Provider | None:
        '''Get a Provider by its provider and model name.'''
        provider = await db.execute(select(Provider).where(Provider.provider_name == provider_name).where(Provider.model_name == model_name))
        return provider.scalar_one_or_none()
    

    @staticmethod
    async def update_provider(db: AsyncSession, provider_id: int, provider_name: str | None, model_name: str | None) -> Provider | None:
        '''Find a Provider by its ID and update its provider name or model name.'''
        try:
            provider = await ProviderService.get_provider_by_id(db, provider_id)
            if not provider:
                return None
            
            if provider_name:
                provider.provider_name = provider_name
            if model_name:
                provider.model_name = model_name
            await db.commit()
            await db.refresh(provider)
            return provider
        except Exception:
            await db.rollback()
            raise
    

    @staticmethod
    async def delete_provider(db: AsyncSession, provider_id: int) -> bool:
        '''Find a Provider by its ID and delete it. Returns True if deleted.'''
        try:
            provider = await ProviderService.get_provider_by_id(db, provider_id)
            if not provider:
                return False
            
            await db.delete(provider)
            await db.commit()
            return True
        except Exception:
            await db.rollback()
            raise
