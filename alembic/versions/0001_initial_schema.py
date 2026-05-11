"""initial schema — users + image_index

Revision ID: 0001_initial
Revises:
Create Date: 2026-05-11

"""
from typing import Sequence, Union

from alembic import op


revision: str = "0001_initial"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id          TEXT PRIMARY KEY,
            email       TEXT UNIQUE NOT NULL,
            name        TEXT,
            created_at  TIMESTAMP DEFAULT NOW()
        );
    """)

    op.execute("""
        CREATE TABLE IF NOT EXISTS image_index (
            id              TEXT PRIMARY KEY,
            user_id         TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            source_id       TEXT,
            embedding_enc   BYTEA,
            caption         TEXT,
            ocr_text        TEXT,
            perceptual_hash TEXT,
            taken_at        TIMESTAMP,
            latitude        DOUBLE PRECISION,
            longitude       DOUBLE PRECISION,
            location_name   TEXT,
            device_model    TEXT,
            width           INTEGER,
            height          INTEGER,
            file_size       BIGINT,
            thumbnail_url   TEXT,
            source_type     TEXT DEFAULT 'upload',
            created_at      TIMESTAMP DEFAULT NOW()
        );
    """)

    op.execute("CREATE INDEX IF NOT EXISTS idx_img_user  ON image_index(user_id);")
    op.execute("CREATE INDEX IF NOT EXISTS idx_img_taken ON image_index(user_id, taken_at);")
    op.execute("CREATE INDEX IF NOT EXISTS idx_img_phash ON image_index(user_id, perceptual_hash);")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS image_index CASCADE;")
    op.execute("DROP TABLE IF EXISTS users CASCADE;")
