"""initial_schema

Revision ID: 0001_initial_schema
Revises: 
Create Date: 2026-09-02 08:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '0001_initial_schema'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    is_postgres = bind.dialect.name == "postgresql"

    # Enable pgvector extension on PostgreSQL
    if is_postgres:
        op.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        try:
            from pgvector.sqlalchemy import Vector
            vector_type = Vector(768)
        except ImportError:
            vector_type = sa.Text()
    else:
        vector_type = sa.Text()

    # 1. users
    op.create_table(
        'users',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('email', sa.Text(), nullable=False),
        sa.Column('hashed_password', sa.String(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('deleted_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_users_email'), 'users', ['email'], unique=True)

    # 2. user_profiles
    op.create_table(
        'user_profiles',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('user_id', sa.String(), nullable=False),
        sa.Column('location', sa.String(), nullable=True),
        sa.Column('gps_coordinates', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('user_id')
    )
    op.create_index(op.f('ix_user_profiles_user_id'), 'user_profiles', ['user_id'], unique=True)

    # 3. chat_sessions
    op.create_table(
        'chat_sessions',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('user_id', sa.String(), nullable=False),
        sa.Column('title', sa.String(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.Column('deleted_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_chat_sessions_user_id'), 'chat_sessions', ['user_id'], unique=False)

    # 4. conversation_history
    op.create_table(
        'conversation_history',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('session_id', sa.String(), nullable=False),
        sa.Column('role', sa.String(), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('timestamp', sa.DateTime(), nullable=True),
        sa.Column('deleted_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['session_id'], ['chat_sessions.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_conversation_history_session_id'), 'conversation_history', ['session_id'], unique=False)
    op.create_index(op.f('ix_conversation_history_timestamp'), 'conversation_history', ['timestamp'], unique=False)
    op.create_index('idx_history_session_timestamp', 'conversation_history', ['session_id', 'timestamp'], unique=False)

    # 5. memory_summaries
    op.create_table(
        'memory_summaries',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('session_id', sa.String(), nullable=False),
        sa.Column('summary_text', sa.Text(), nullable=False),
        sa.Column('last_processed_message_id', sa.Integer(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['session_id'], ['chat_sessions.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('session_id')
    )
    op.create_index(op.f('ix_memory_summaries_session_id'), 'memory_summaries', ['session_id'], unique=True)

    # 6. predictions
    op.create_table(
        'predictions',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('session_id', sa.String(), nullable=True),
        sa.Column('task', sa.String(), nullable=False),
        sa.Column('input_metadata', sa.JSON(), nullable=True),
        sa.Column('output_result', sa.JSON(), nullable=True),
        sa.Column('image_url', sa.String(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['session_id'], ['chat_sessions.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_predictions_session_id'), 'predictions', ['session_id'], unique=False)

    # 7. notifications
    op.create_table(
        'notifications',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('user_id', sa.String(), nullable=False),
        sa.Column('title', sa.String(), nullable=False),
        sa.Column('message', sa.Text(), nullable=False),
        sa.Column('is_read', sa.Boolean(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_notifications_user_id'), 'notifications', ['user_id'], unique=False)

    # 8. settings
    op.create_table(
        'settings',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('user_id', sa.String(), nullable=False),
        sa.Column('language', sa.String(), nullable=True),
        sa.Column('notifications_enabled', sa.Boolean(), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('user_id')
    )
    op.create_index(op.f('ix_settings_user_id'), 'settings', ['user_id'], unique=True)

    # 9. knowledge_metadata
    op.create_table(
        'knowledge_metadata',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('document_source', sa.String(), nullable=False),
        sa.Column('category', sa.String(), nullable=False),
        sa.Column('chunk_metadata', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_knowledge_metadata_category'), 'knowledge_metadata', ['category'], unique=False)

    # 10. audit_logs
    op.create_table(
        'audit_logs',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('user_id', sa.String(), nullable=False),
        sa.Column('action', sa.String(), nullable=False),
        sa.Column('action_details', sa.JSON(), nullable=True),
        sa.Column('timestamp', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_audit_logs_user_id'), 'audit_logs', ['user_id'], unique=False)

    # 11. farms
    op.create_table(
        'farms',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('owner_id', sa.String(), nullable=False),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('location', sa.String(), nullable=False),
        sa.Column('size_acres', sa.Float(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['owner_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_farms_owner_id'), 'farms', ['owner_id'], unique=False)

    # 12. crops
    op.create_table(
        'crops',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('farm_id', sa.String(), nullable=False),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('variety', sa.String(), nullable=True),
        sa.Column('season', sa.String(), nullable=False),
        sa.Column('area_allocated', sa.Float(), nullable=True),
        sa.Column('sowing_date', sa.DateTime(), nullable=True),
        sa.Column('status', sa.String(), nullable=True),
        sa.ForeignKeyConstraint(['farm_id'], ['farms.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_crops_farm_id'), 'crops', ['farm_id'], unique=False)

    # 13. animals
    op.create_table(
        'animals',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('farm_id', sa.String(), nullable=False),
        sa.Column('tag_number', sa.String(), nullable=False),
        sa.Column('species', sa.String(), nullable=False),
        sa.Column('breed', sa.String(), nullable=True),
        sa.Column('age_months', sa.Integer(), nullable=True),
        sa.Column('health_status', sa.String(), nullable=True),
        sa.ForeignKeyConstraint(['farm_id'], ['farms.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_animals_farm_id'), 'animals', ['farm_id'], unique=False)
    op.create_index(op.f('ix_animals_tag_number'), 'animals', ['tag_number'], unique=True)

    # 14. milk_collection
    op.create_table(
        'milk_collection',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('animal_id', sa.String(), nullable=False),
        sa.Column('date', sa.DateTime(), nullable=True),
        sa.Column('yield_liters', sa.Float(), nullable=False),
        sa.Column('fat_content', sa.Float(), nullable=True),
        sa.Column('snf_content', sa.Float(), nullable=True),
        sa.ForeignKeyConstraint(['animal_id'], ['animals.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_milk_collection_animal_id'), 'milk_collection', ['animal_id'], unique=False)
    op.create_index(op.f('ix_milk_collection_date'), 'milk_collection', ['date'], unique=False)

    # 15. prompt_templates
    op.create_table(
        'prompt_templates',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('version', sa.String(), nullable=False),
        sa.Column('template_text', sa.Text(), nullable=False),
        sa.Column('config', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_prompt_templates_name'), 'prompt_templates', ['name'], unique=True)

    # 16. documents
    op.create_table(
        'documents',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('filename', sa.String(), nullable=False),
        sa.Column('source', sa.String(), nullable=True),
        sa.Column('license', sa.String(), nullable=True),
        sa.Column('version', sa.String(), nullable=True),
        sa.Column('language', sa.String(), nullable=True),
        sa.Column('region', sa.String(), nullable=True),
        sa.Column('publication_date', sa.DateTime(), nullable=True),
        sa.Column('confidence', sa.Float(), nullable=True),
        sa.Column('status', sa.String(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_documents_filename'), 'documents', ['filename'], unique=True)

    # 17. document_chunks
    op.create_table(
        'document_chunks',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('document_id', sa.String(), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('embedding', vector_type, nullable=False),
        sa.Column('page_number', sa.Integer(), nullable=True),
        sa.Column('chunk_index', sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(['document_id'], ['documents.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_document_chunks_document_id'), 'document_chunks', ['document_id'], unique=False)

    # 18. memory_records
    op.create_table(
        'memory_records',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('timestamp', sa.DateTime(), nullable=True),
        sa.Column('category', sa.String(), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('importance', sa.Float(), nullable=True),
        sa.Column('recency', sa.Float(), nullable=True),
        sa.Column('access_frequency', sa.Integer(), nullable=True),
        sa.Column('confidence', sa.Float(), nullable=True),
        sa.Column('embedding', vector_type, nullable=True),
        sa.Column('source', sa.String(), nullable=True),
        sa.Column('tags', sa.JSON(), nullable=True),
        sa.Column('expiration_policy', sa.String(), nullable=True),
        sa.Column('status', sa.String(), nullable=True),
        sa.Column('feedback_score', sa.Float(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_memory_records_category'), 'memory_records', ['category'], unique=False)
    op.create_index(op.f('ix_memory_records_status'), 'memory_records', ['status'], unique=False)
    op.create_index(op.f('ix_memory_records_timestamp'), 'memory_records', ['timestamp'], unique=False)


def downgrade() -> None:
    # Drop tables in reverse order of foreign key dependencies
    op.drop_index(op.f('ix_memory_records_timestamp'), table_name='memory_records')
    op.drop_index(op.f('ix_memory_records_status'), table_name='memory_records')
    op.drop_index(op.f('ix_memory_records_category'), table_name='memory_records')
    op.drop_table('memory_records')

    op.drop_index(op.f('ix_document_chunks_document_id'), table_name='document_chunks')
    op.drop_table('document_chunks')

    op.drop_index(op.f('ix_documents_filename'), table_name='documents')
    op.drop_table('documents')

    op.drop_index(op.f('ix_prompt_templates_name'), table_name='prompt_templates')
    op.drop_table('prompt_templates')

    op.drop_index(op.f('ix_milk_collection_date'), table_name='milk_collection')
    op.drop_index(op.f('ix_milk_collection_animal_id'), table_name='milk_collection')
    op.drop_table('milk_collection')

    op.drop_index(op.f('ix_animals_tag_number'), table_name='animals')
    op.drop_index(op.f('ix_animals_farm_id'), table_name='animals')
    op.drop_table('animals')

    op.drop_index(op.f('ix_crops_farm_id'), table_name='crops')
    op.drop_table('crops')

    op.drop_index(op.f('ix_farms_owner_id'), table_name='farms')
    op.drop_table('farms')

    op.drop_index(op.f('ix_audit_logs_user_id'), table_name='audit_logs')
    op.drop_table('audit_logs')

    op.drop_index(op.f('ix_knowledge_metadata_category'), table_name='knowledge_metadata')
    op.drop_table('knowledge_metadata')

    op.drop_index(op.f('ix_settings_user_id'), table_name='settings')
    op.drop_table('settings')

    op.drop_index(op.f('ix_notifications_user_id'), table_name='notifications')
    op.drop_table('notifications')

    op.drop_index(op.f('ix_predictions_session_id'), table_name='predictions')
    op.drop_table('predictions')

    op.drop_index(op.f('ix_memory_summaries_session_id'), table_name='memory_summaries')
    op.drop_table('memory_summaries')

    op.drop_index('idx_history_session_timestamp', table_name='conversation_history')
    op.drop_index(op.f('ix_conversation_history_timestamp'), table_name='conversation_history')
    op.drop_index(op.f('ix_conversation_history_session_id'), table_name='conversation_history')
    op.drop_table('conversation_history')

    op.drop_index(op.f('ix_chat_sessions_user_id'), table_name='chat_sessions')
    op.drop_table('chat_sessions')

    op.drop_index(op.f('ix_user_profiles_user_id'), table_name='user_profiles')
    op.drop_table('user_profiles')

    op.drop_index(op.f('ix_users_email'), table_name='users')
    op.drop_table('users')
