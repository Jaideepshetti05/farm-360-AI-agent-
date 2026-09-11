import uuid
import datetime
from sqlalchemy import Column, Integer, String, Float, Boolean, ForeignKey, DateTime, Text, JSON, Index
from sqlalchemy.orm import relationship
from sqlalchemy.types import TypeDecorator
from backend.core.database import Base
from backend.core.security import encryptor

class EncryptedString(TypeDecorator):
    """Symmetrically encrypts a string column before writing, and decrypts it on read."""
    impl = Text  # Use Text to handle longer encrypted strings
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is not None:
            return encryptor.encrypt(str(value))
        return value

    def process_result_value(self, value, dialect):
        if value is not None:
            return encryptor.decrypt(value)
        return value

def generate_uuid():
    return str(uuid.uuid4())

class User(Base):
    __tablename__ = "users"
    id = Column(String, primary_key=True, default=generate_uuid)
    email = Column(EncryptedString, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    deleted_at = Column(DateTime, nullable=True)

    profile = relationship("UserProfile", uselist=False, back_populates="user", cascade="all, delete-orphan")
    sessions = relationship("ChatSession", back_populates="user", cascade="all, delete-orphan")
    notifications = relationship("Notification", back_populates="user", cascade="all, delete-orphan")
    settings = relationship("Setting", uselist=False, back_populates="user", cascade="all, delete-orphan")
    audit_logs = relationship("AuditLog", back_populates="user", cascade="all, delete-orphan")
    farms = relationship("Farm", back_populates="owner", cascade="all, delete-orphan")

class UserProfile(Base):
    __tablename__ = "user_profiles"
    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), unique=True, nullable=False, index=True)
    location = Column(String)
    gps_coordinates = Column(EncryptedString)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    user = relationship("User", back_populates="profile")

class ChatSession(Base):
    __tablename__ = "chat_sessions"
    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    title = Column(String, default="New Conversation")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    deleted_at = Column(DateTime, nullable=True)

    user = relationship("User", back_populates="sessions")
    history = relationship("ConversationHistory", back_populates="session", cascade="all, delete-orphan")
    summary = relationship("MemorySummary", uselist=False, back_populates="session", cascade="all, delete-orphan")
    predictions = relationship("Prediction", back_populates="session")

class ConversationHistory(Base):
    __tablename__ = "conversation_history"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    role = Column(String, nullable=False)  # user / assistant / system
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow, index=True)
    deleted_at = Column(DateTime, nullable=True)

    session = relationship("ChatSession", back_populates="history")

    __table_args__ = (
        Index("idx_history_session_timestamp", "session_id", "timestamp"),
    )

class MemorySummary(Base):
    __tablename__ = "memory_summaries"
    id = Column(String, primary_key=True, default=generate_uuid)
    session_id = Column(String, ForeignKey("chat_sessions.id", ondelete="CASCADE"), unique=True, nullable=False, index=True)
    summary_text = Column(Text, nullable=False)
    last_processed_message_id = Column(Integer, default=0)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    session = relationship("ChatSession", back_populates="summary")

class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(String, primary_key=True, default=generate_uuid)
    session_id = Column(String, ForeignKey("chat_sessions.id", ondelete="SET NULL"), nullable=True, index=True)
    task = Column(String, nullable=False)  # crop_disease / breed / yield / animal_disease
    input_metadata = Column(JSON)
    output_result = Column(JSON)
    image_url = Column(String)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    session = relationship("ChatSession", back_populates="predictions")

class Notification(Base):
    __tablename__ = "notifications"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    title = Column(String, nullable=False)
    message = Column(Text, nullable=False)
    is_read = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    user = relationship("User", back_populates="notifications")

class Setting(Base):
    __tablename__ = "settings"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), unique=True, nullable=False, index=True)
    language = Column(String, default="en")
    notifications_enabled = Column(Boolean, default=True)

    user = relationship("User", back_populates="settings")

class KnowledgeMetadata(Base):
    __tablename__ = "knowledge_metadata"
    id = Column(Integer, primary_key=True, autoincrement=True)
    document_source = Column(String, nullable=False)
    category = Column(String, nullable=False, index=True)
    chunk_metadata = Column(JSON)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class AuditLog(Base):
    __tablename__ = "audit_logs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    action = Column(String, nullable=False)
    action_details = Column(JSON)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

    user = relationship("User", back_populates="audit_logs")

# ── FUTURE MODULE COMPATIBILITY STUBS ──────────────────────────────────────────

class Farm(Base):
    __tablename__ = "farms"
    id = Column(String, primary_key=True, default=generate_uuid)
    owner_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    name = Column(String, nullable=False)
    location = Column(String, nullable=False)
    size_acres = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    owner = relationship("User", back_populates="farms")
    crops = relationship("Crop", back_populates="farm", cascade="all, delete-orphan")
    animals = relationship("Animal", back_populates="farm", cascade="all, delete-orphan")

class Crop(Base):
    __tablename__ = "crops"
    id = Column(String, primary_key=True, default=generate_uuid)
    farm_id = Column(String, ForeignKey("farms.id", ondelete="CASCADE"), nullable=False, index=True)
    name = Column(String, nullable=False)
    variety = Column(String)
    season = Column(String, nullable=False)
    area_allocated = Column(Float)
    sowing_date = Column(DateTime)
    status = Column(String, default="active")

    farm = relationship("Farm", back_populates="crops")

class Animal(Base):
    __tablename__ = "animals"
    id = Column(String, primary_key=True, default=generate_uuid)
    farm_id = Column(String, ForeignKey("farms.id", ondelete="CASCADE"), nullable=False, index=True)
    tag_number = Column(String, nullable=False, unique=True, index=True)
    species = Column(String, nullable=False)
    breed = Column(String)
    age_months = Column(Integer)
    health_status = Column(String, default="healthy")

    farm = relationship("Farm", back_populates="animals")
    milk_records = relationship("MilkCollection", back_populates="animal", cascade="all, delete-orphan")

class MilkCollection(Base):
    __tablename__ = "milk_collection"
    id = Column(String, primary_key=True, default=generate_uuid)
    animal_id = Column(String, ForeignKey("animals.id", ondelete="CASCADE"), nullable=False, index=True)
    date = Column(DateTime, default=datetime.datetime.utcnow, index=True)
    yield_liters = Column(Float, nullable=False)
    fat_content = Column(Float)
    snf_content = Column(Float)

    animal = relationship("Animal", back_populates="milk_records")

class PromptTemplate(Base):
    __tablename__ = "prompt_templates"
    id = Column(String, primary_key=True, default=generate_uuid)
    name = Column(String, unique=True, index=True, nullable=False) # e.g. "general_assistant"
    version = Column(String, nullable=False) # e.g. "1.0.0"
    template_text = Column(Text, nullable=False)
    config = Column(JSON) # e.g. {"temperature": 0.2, "max_tokens": 1000}
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

# Vector DB fallback type decorators
try:
    from pgvector.sqlalchemy import Vector
    HAS_PGVECTOR = True
except ImportError:
    HAS_PGVECTOR = False
    from sqlalchemy.types import TypeDecorator
    import json
    
    class Vector(TypeDecorator):
        impl = Text
        cache_ok = True
        
        def process_bind_param(self, value, dialect):
            if value is not None:
                return json.dumps(value)
            return None
            
        def process_result_value(self, value, dialect):
            if value is not None:
                return json.loads(value)
            return None

class Document(Base):
    __tablename__ = "documents"
    id = Column(String, primary_key=True, default=generate_uuid)
    filename = Column(String, unique=True, index=True, nullable=False)
    source = Column(String, nullable=True) # e.g. "ICAR", "Govt"
    license = Column(String, nullable=True)
    version = Column(String, default="1.0.0")
    language = Column(String, default="en")
    region = Column(String, nullable=True)
    publication_date = Column(DateTime, nullable=True)
    confidence = Column(Float, default=1.0)
    status = Column(String, default="active") # active, archived
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    id = Column(String, primary_key=True, default=generate_uuid)
    document_id = Column(String, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    content = Column(Text, nullable=False)
    embedding = Column(Vector(768), nullable=False) # 768-dim Gemini embeddings
    page_number = Column(Integer, nullable=True)
    chunk_index = Column(Integer, nullable=True)

class MemoryRecord(Base):
    __tablename__ = "memory_records"
    id = Column(String, primary_key=True, default=generate_uuid)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow, index=True)
    category = Column(String, nullable=False, index=True)  # working, short_term, long_term_semantic, episodic
    content = Column(Text, nullable=False)
    importance = Column(Float, default=1.0)
    recency = Column(Float, default=1.0)
    access_frequency = Column(Integer, default=1)
    confidence = Column(Float, default=1.0)
    embedding = Column(Vector(768), nullable=True)
    source = Column(String, nullable=True)
    tags = Column(JSON, default=list)
    expiration_policy = Column(String, nullable=True)
    status = Column(String, default="active", index=True)  # active, archived, compressed, deleted
    feedback_score = Column(Float, default=0.0)
