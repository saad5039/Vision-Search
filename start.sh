#!/bin/sh
set -e
echo "▶ Running migrations..."
alembic upgrade head
echo "✓ Migrations done. Starting server on port $PORT..."
exec gunicorn app:app --bind "0.0.0.0:$PORT" --workers 1 --threads 4 --timeout 300 --preload
