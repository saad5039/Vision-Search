"""
Vision Search — Local Startup Script
Run with: python run.py
For production (Railway) gunicorn launches app:app directly.
"""
import os, sys

def check_deps():
    # map: display name → actual import name
    pkgs = {
        'flask':        'flask',
        'PyJWT':        'jwt',
        'cryptography': 'cryptography',
        'numpy':        'numpy',
        'psycopg2':     'psycopg2',
    }
    missing = []
    for display, module in pkgs.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(display)
    if missing:
        print(f"❌ Missing packages: {', '.join(missing)}")
        print(f"   Run: pip install -r requirements.txt")
        sys.exit(1)

if __name__ == "__main__":
    check_deps()
    if not os.environ.get("DATABASE_URL"):
        print("❌ DATABASE_URL is not set.")
        print("   Local:    set DATABASE_URL=postgresql://user:pass@localhost:5432/visionsearch")
        print("   Railway:  add the PostgreSQL plugin — it injects DATABASE_URL automatically.")
        sys.exit(1)

    import subprocess
    print("▶ Running database migrations…")
    result = subprocess.run([sys.executable, "-m", "alembic", "upgrade", "head"])
    if result.returncode != 0:
        print("❌ Alembic migration failed — aborting.")
        sys.exit(1)
    print("✓ Migrations up to date.")

    from app import app
    port = int(os.environ.get("PORT", 5000))
    print(f"""
╔══════════════════════════════════════════════╗
║           VISION SEARCH  v1.0.0              ║
║   Privacy-first AI Image Search Engine       ║
╠══════════════════════════════════════════════╣
║  → Open:  http://localhost:{port:<5}             ║
║  → Stack: Flask · PostgreSQL · NumPy · AES   ║
║  → FYP:   FAST NUCES Karachi · Spring 2026   ║
╚══════════════════════════════════════════════╝
    """)
    app.run(debug=False, port=port, threaded=True, host="0.0.0.0")
