"""initial

Revision ID: ecca5942d332
Revises: 
Create Date: 2026-03-28 03:28:19.943488

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'ecca5942d332'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # cameras table
    op.create_table(
        'cameras',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('location', sa.String(), nullable=True),
        sa.Column('rtsp_url', sa.String(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id'),
    )

    # events table
    op.create_table(
        'events',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('camera_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('track_id', sa.BigInteger(), nullable=True),
        sa.Column('timestamp_utc', sa.DateTime(timezone=True), nullable=True),
        sa.Column('frame_number', sa.BigInteger(), nullable=True),
        sa.Column('class_name', sa.String(), nullable=True),
        sa.Column('attributes', sa.JSON(), nullable=True),
        sa.Column('bbox', sa.JSON(), nullable=True),
        sa.Column('confidence', sa.Float(), nullable=True),
        sa.Column('caption', sa.Text(), nullable=True),
        sa.Column('clip_path', sa.String(), nullable=True),
        sa.Column('qdrant_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['camera_id'], ['cameras.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_events_timestamp_utc', 'events', ['timestamp_utc'])
    op.create_index('ix_events_class_name', 'events', ['class_name'])
    op.create_index('ix_events_camera_timestamp', 'events', ['camera_id', 'timestamp_utc'])

    # scene_diffs table
    op.create_table(
        'scene_diffs',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('camera_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('object_label', sa.String(), nullable=True),
        sa.Column('event_type', sa.String(), nullable=True),
        sa.Column('timestamp_utc', sa.DateTime(timezone=True), nullable=True),
        sa.Column('frame_number', sa.BigInteger(), nullable=True),
        sa.Column('baseline_frame_path', sa.String(), nullable=True),
        sa.Column('diff_frame_path', sa.String(), nullable=True),
        sa.Column('clip_path', sa.String(), nullable=True),
        sa.ForeignKeyConstraint(['camera_id'], ['cameras.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_scene_diffs_timestamp_utc', 'scene_diffs', ['timestamp_utc'])


def downgrade() -> None:
    op.drop_table('scene_diffs')
    op.drop_table('events')
    op.drop_table('cameras')
