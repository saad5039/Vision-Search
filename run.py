"""
Vision Search — Startup Script
Run this file to start the server: python3 run.py
"""
import sys
import os

def check_deps():
    missing = []
    for pkg in ['flask', 'jwt', 'cryptography', 'numpy']:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"❌ Missing packages: {', '.join(missing)}")
        print(f"   Run: pip3 install {' '.join(missing)}")
        sys.exit(1)

if __name__ == "__main__":
    check_deps()
    from app import app, init_db
    init_db()
    port = int(os.environ.get("PORT", 5000))
    print(f"""
╔══════════════════════════════════════════════╗
║           VISION SEARCH  v1.0.0              ║
║   Privacy-first AI Image Search Engine       ║
╠══════════════════════════════════════════════╣
║  → Open:  http://localhost:{port}               ║
║  → Stack: Flask · SQLite · NumPy · AES-256   ║
║  → FYP:   FAST NUCES Karachi · Spring 2026   ║
╚══════════════════════════════════════════════╝
    """)
    app.run(debug=False, port=port, threaded=True, host="0.0.0.0")
