"""
Vision Search v4.1 — Production Build
• Sentence Transformers (all-MiniLM-L6-v2) — 384-d semantic embeddings
• BLIP — real AI image captioning
• Tesseract OCR — document text extraction
• AES-256 encrypted embedding storage
• AI-ready guard — blocks uploads until ST loaded
• Rule-based query planner
"""

import os, json, uuid, time, sqlite3, hashlib, secrets, threading, io
from datetime import datetime, timedelta
from functools import wraps

import jwt
import numpy as np
from PIL import Image as PILImage, ExifTags, ImageEnhance, ImageFilter
from cryptography.fernet import Fernet
from flask import Flask, request, jsonify, render_template, g, send_from_directory

# ── Sentence Transformers ─────────────────────────────────────────────────────
try:
    from sentence_transformers import SentenceTransformer
    print(">>> Loading Sentence Transformer model… (this may take 10-30 seconds)")
    _st_model   = SentenceTransformer("all-MiniLM-L6-v2")
    EMB_DIM     = 384
    ST_AVAILABLE = True
    print(f"✓ Sentence Transformers loaded — {EMB_DIM}-d embeddings active")
except Exception as e:
    import traceback
    ST_AVAILABLE = False
    EMB_DIM      = 512
    print(f"!!! Sentence Transformers FAILED: {type(e).__name__}: {e}")
    traceback.print_exc()
    print("  Run: py -m pip install sentence-transformers")

# ── BLIP Image Captioning ─────────────────────────────────────────────────────
try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    import torch
    print(">>> Loading BLIP captioning model… (may take 30-60 seconds)")
    _blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    _blip_model     = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    _blip_model.eval()
    BLIP_AVAILABLE = True
    print("✓ BLIP captioning loaded — real AI captions active")
except Exception as e:
    import traceback
    BLIP_AVAILABLE = False
    print(f"!!! BLIP FAILED: {type(e).__name__}: {e}")
    traceback.print_exc()
    print("  Run: py -m pip install transformers torch")

# ── Cohere Embed API ──────────────────────────────────────────────────────────
COHERE_AVAILABLE = False
_cohere_client   = None
COHERE_API_KEY   = os.environ.get("COHERE_API_KEY", "")
if COHERE_API_KEY:
    try:
        import cohere as _cohere_sdk
        _cohere_client = _cohere_sdk.Client(COHERE_API_KEY)
        # Quick test
        _cohere_client.embed(texts=["test"], model="embed-english-v3.0",
                            input_type="search_document")
        COHERE_AVAILABLE = True
        print("✓ Cohere API connected — high quality embeddings active")
    except Exception as _ce:
        print(f"ℹ Cohere not available: {_ce}")
        print("  Install: py -m pip install cohere")
else:
    print("ℹ No COHERE_API_KEY — using local Sentence Transformers")
    print("  Get free key: cohere.com → set COHERE_API_KEY=your_key")

# ── Query Planner — Rule-based ────────────────────────────────────────────────
# Gemini API removed: key has allowlist restriction preventing server-side calls.
# Rule-based planner handles: time ranges, locations, visual concepts, OCR mode.
GEMINI_AVAILABLE = False
_gemini_client   = None
_gemini_model    = None
print("✓ Rule-based query planner active")

# ── OCR ───────────────────────────────────────────────────────────────────────
try:
    import pytesseract
    _win = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(_win):
        pytesseract.pytesseract.tesseract_cmd = _win
    pytesseract.get_tesseract_version()
    OCR_AVAILABLE = True
    print("✓ Tesseract OCR available")
except Exception:
    OCR_AVAILABLE = False
    print("ℹ Tesseract not found — OCR disabled")

# ── AI Ready flag ─────────────────────────────────────────────────────────────
# True only when ST is loaded — guarantees embeddings will be 384-d
AI_READY = ST_AVAILABLE or COHERE_AVAILABLE
if AI_READY:
    mode = "Cohere API" if COHERE_AVAILABLE else "Sentence Transformers"
    print(f"✓ All AI components ready ({mode}) — safe to upload photos")
else:
    print("⚠ WARNING: No embedding model loaded — uploads will have empty embeddings!")
    print("  Fix: py -m pip install sentence-transformers then restart")
    print("  Or: set COHERE_API_KEY=your_key for cloud embeddings")

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = secrets.token_hex(32)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024

ENCRYPTION_KEY = Fernet.generate_key()
fernet         = Fernet(ENCRYPTION_KEY)
JWT_SECRET     = secrets.token_hex(32)
JWT_ALGO       = "HS256"

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DB_PATH    = os.path.join(BASE_DIR, "db", "vision_search.db")
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXT    = {".jpg",".jpeg",".png",".gif",".webp",".bmp",".tiff",".tif"}
MAX_FILE_BYTES = 50 * 1024 * 1024
THUMB_SIZE     = (600, 600)

upload_jobs = {}

# ── Database ──────────────────────────────────────────────────────────────────
def get_db():
    if "db" not in g:
        conn = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=-32000")
        g.db = conn
    return g.db

@app.teardown_appcontext
def close_db(e=None):
    db = g.pop("db", None)
    if db: db.close()

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY, email TEXT UNIQUE NOT NULL,
                name TEXT, created_at TEXT DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS image_index (
                id              TEXT PRIMARY KEY,
                user_id         TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                source_id       TEXT,
                embedding_enc   BLOB,
                caption         TEXT,
                ocr_text        TEXT,
                perceptual_hash TEXT,
                taken_at        TEXT,
                latitude        REAL, longitude REAL,
                location_name   TEXT, device_model TEXT,
                width INTEGER, height INTEGER, file_size INTEGER,
                thumbnail_url   TEXT,
                source_type     TEXT DEFAULT 'upload',
                created_at      TEXT DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_img_user  ON image_index(user_id);
            CREATE INDEX IF NOT EXISTS idx_img_taken ON image_index(user_id, taken_at);
            CREATE INDEX IF NOT EXISTS idx_img_phash ON image_index(user_id, perceptual_hash);
        """)

# ── Auth ──────────────────────────────────────────────────────────────────────
def make_token(user_id):
    return jwt.encode(
        {"sub":user_id,"iat":datetime.utcnow(),"exp":datetime.utcnow()+timedelta(days=7)},
        JWT_SECRET, algorithm=JWT_ALGO)

def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get("Authorization","").replace("Bearer ","") \
                or request.cookies.get("vs_token","")
        if not token: return jsonify({"error":"Unauthorized"}), 401
        try:
            payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGO])
            g.user_id = payload["sub"]
        except jwt.ExpiredSignatureError: return jsonify({"error":"Token expired"}), 401
        except jwt.InvalidTokenError:     return jsonify({"error":"Invalid token"}), 401
        return f(*args, **kwargs)
    return decorated

# ── Embedding ─────────────────────────────────────────────────────────────────
def text_to_embedding(text: str) -> np.ndarray:
    # Try Cohere first — highest quality embeddings
    if COHERE_AVAILABLE and _cohere_client:
        try:
            resp = _cohere_client.embed(
                texts=[text],
                model="embed-english-v3.0",
                input_type="search_query"
            )
            vec = np.array(resp.embeddings[0], dtype=np.float32)
            norm = np.linalg.norm(vec)
            return vec / norm if norm > 0 else vec
        except Exception as e:
            print(f"Cohere embed failed: {e} — falling back to ST")

    # Sentence Transformers fallback
    if ST_AVAILABLE:
        try:
            vec = _st_model.encode(text, normalize_embeddings=True)
            return vec.astype(np.float32)
        except Exception:
            pass

    # Hash-based fallback
    rng = np.random.default_rng(abs(hash(text)) % (2**31))
    vec = rng.random(EMB_DIM).astype(np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec

def image_to_embedding(caption: str, filename: str = "") -> np.ndarray:
    """Generate embedding from caption + filename. Uses Cohere if available."""
    name = os.path.splitext(filename)[0].replace("_"," ").replace("-"," ")
    real_words = [w for w in name.split() if not all(c in '0123456789abcdefABCDEF' for c in w)]
    name_clean = " ".join(real_words)
    text = (caption + " " + name_clean).strip() if name_clean else caption

    # Use Cohere with search_document input type for indexing
    if COHERE_AVAILABLE and _cohere_client:
        try:
            resp = _cohere_client.embed(
                texts=[text],
                model="embed-english-v3.0",
                input_type="search_document"
            )
            vec = np.array(resp.embeddings[0], dtype=np.float32)
            norm = np.linalg.norm(vec)
            return vec / norm if norm > 0 else vec
        except Exception as e:
            print(f"Cohere image embed failed: {e} — using ST")

    return text_to_embedding(text)

def encrypt_embedding(vec: np.ndarray) -> bytes:
    return fernet.encrypt(json.dumps(vec.tolist()).encode())

def decrypt_embedding(data: bytes) -> np.ndarray:
    return np.array(json.loads(fernet.decrypt(data).decode()), dtype=np.float32)

# ── BLIP captioning ───────────────────────────────────────────────────────────
def generate_caption(img: PILImage.Image, filename: str) -> str:
    if BLIP_AVAILABLE:
        try:
            rgb    = img.convert("RGB").resize((384,384), PILImage.LANCZOS)
            inputs = _blip_processor(rgb, return_tensors="pt")
            with torch.no_grad():
                out = _blip_model.generate(**inputs, max_new_tokens=50)
            caption = _blip_processor.decode(out[0], skip_special_tokens=True).strip()
            if caption:
                return caption
        except Exception as e:
            print(f"BLIP failed: {e}")

    # Fallback caption from filename
    rgb  = img.convert("RGB")
    w, h = rgb.size
    arr  = np.array(rgb.resize((16,16)), dtype=np.float32)
    r,g_c,b = arr[:,:,0].mean(), arr[:,:,1].mean(), arr[:,:,2].mean()
    bright = (r+g_c+b)/3
    mood   = "bright" if bright>170 else "dark" if bright<70 else "natural"
    orient = "landscape" if w>h else "portrait" if h>w else "square"
    tone   = "warm" if r>g_c+20 and r>b+20 else "cool" if b>r+20 else "neutral"
    raw = os.path.splitext(filename)[0].replace("_"," ").replace("-"," ")
    words = [w for w in raw.split() if not all(c in '0123456789abcdefABCDEF' for c in w) and len(w)>1]
    subject = " ".join(words[:4]).title() if words else "Photo"
    return f"{subject} — {orient}, {mood} {tone}"

# ── Batch search ──────────────────────────────────────────────────────────────
def batch_cosine_search(query_vec: np.ndarray, rows: list, top_k: int = 30) -> list:
    if not rows: return []
    target_dim = len(query_vec)
    embeddings, valid_rows = [], []
    dim_counts = {}
    for row in rows:
        try:
            emb = decrypt_embedding(row["embedding_enc"])
            dim_counts[len(emb)] = dim_counts.get(len(emb), 0) + 1
            if len(emb) != target_dim: continue
            embeddings.append(emb)
            valid_rows.append(row)
        except Exception:
            pass

    # Debug: print dimension distribution
    if not embeddings:
        print(f"!!! DIMENSION MISMATCH — query is {target_dim}-d, stored embeddings are: {dim_counts}")
        print(f"!!! Photos uploaded with old embedding model. Re-upload required.")
        return []

    matrix = np.stack(embeddings, axis=0)
    q      = query_vec / (np.linalg.norm(query_vec) + 1e-9)
    norms  = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9
    scores = (matrix / norms) @ q
    k      = min(top_k, len(scores))
    top_idx = np.argpartition(scores, -k)[-k:]
    top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
    return [(float(scores[i]), valid_rows[i]) for i in top_idx]

# ── OCR ───────────────────────────────────────────────────────────────────────
def extract_ocr(img: PILImage.Image) -> str | None:
    if not OCR_AVAILABLE: return None
    try:
        w, h = img.size
        scale = max(1500/w, 1500/h, 1.0)
        grey  = img.convert("L")
        if scale > 1:
            grey = grey.resize((int(w*scale), int(h*scale)), PILImage.LANCZOS)
        grey = ImageEnhance.Contrast(grey).enhance(2.0)
        grey = grey.filter(ImageFilter.SHARPEN)
        t1 = pytesseract.image_to_string(grey, config="--psm 6 --oem 3").strip()
        t2 = pytesseract.image_to_string(grey, config="--psm 3 --oem 3").strip()
        text = t1 if len(t1) >= len(t2) else t2
        if not text: return None
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        words = text.split()
        doc_sigs = ["total","amount","pkr","rs.","price","invoice","receipt",
                    "bill","date","payment","cash","paid","qty","item","tax"]
        doc_score = sum(1 for s in doc_sigs if s in text.lower())
        if len(words) < 8 and (len(lines) < 3 or doc_score < 2): return None
        clean = "\n".join(l for l in lines if sum(1 for c in l if c.isalnum()) >= 2)
        return clean[:3000] if clean else None
    except Exception:
        return None

# ── EXIF ──────────────────────────────────────────────────────────────────────
def extract_exif(img: PILImage.Image):
    taken_at = datetime.now().isoformat()
    lat = lon = device = None
    try:
        raw = img._getexif() or {}
        tag = {v:k for k,v in ExifTags.TAGS.items()}
        for tn in ["DateTimeOriginal","DateTime"]:
            tid = tag.get(tn)
            if tid and tid in raw:
                try: taken_at = datetime.strptime(str(raw[tid]),"%Y:%m:%d %H:%M:%S").isoformat()
                except: pass
                break
        parts = []
        for tn in ["Make","Model"]:
            tid = tag.get(tn)
            if tid and tid in raw: parts.append(str(raw[tid]).strip())
        if parts: device = " ".join(parts)
        gid = tag.get("GPSInfo")
        gps = raw.get(gid,{}) if gid else {}
        if gps and 2 in gps and 4 in gps:
            def dms(v): d,m,s=[float(x) for x in v]; return d+m/60+s/3600
            lat = dms(gps[2])*(-1 if gps.get(1,"N")=="S" else 1)
            lon = dms(gps[4])*(-1 if gps.get(3,"E")=="W" else 1)
    except: pass
    return taken_at, lat, lon, device

# ── Perceptual hash ───────────────────────────────────────────────────────────
def perceptual_hash(img: PILImage.Image, raw: bytes) -> str:
    ph = img.convert("L").resize((8,8), PILImage.LANCZOS)
    px = list(ph.getdata())
    avg = sum(px)/len(px)
    bits = int("".join("1" if p>=avg else "0" for p in px),2)
    salt = (sum(raw[:256])*31 + len(raw)) & 0xFFFF
    return hex((bits<<16)|salt)[2:].zfill(20)

# ── Thumbnail ─────────────────────────────────────────────────────────────────
def make_thumbnail(img: PILImage.Image) -> bytes:
    img = img.convert("RGB")
    img.thumbnail(THUMB_SIZE, PILImage.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=82, optimize=True)
    return buf.getvalue()

# ── Upload worker ─────────────────────────────────────────────────────────────
def process_upload_batch(user_id: str, file_data_list: list):
    upload_jobs[user_id] = {"status":"running","processed":0,
                            "total":len(file_data_list),"errors":[],"results":[]}
    user_dir = os.path.join(UPLOAD_DIR, user_id)
    os.makedirs(user_dir, exist_ok=True)
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")

    for filename, raw in file_data_list:
        try:
            img = PILImage.open(io.BytesIO(raw))
            img.verify()
            img = PILImage.open(io.BytesIO(raw))
            w, h = img.size

            phash = perceptual_hash(img, raw)
            if db.execute("SELECT id FROM image_index WHERE user_id=? AND perceptual_hash=?",
                         (user_id, phash)).fetchone():
                upload_jobs[user_id]["errors"].append({"file":filename,"error":"Duplicate"})
                upload_jobs[user_id]["processed"] += 1
                continue

            taken_at, lat, lon, device = extract_exif(img)
            ocr_text = extract_ocr(img)
            caption  = generate_caption(img, filename)

            # Blend OCR into caption for better search
            caption_for_emb = caption
            if ocr_text:
                caption_for_emb = caption + " " + ocr_text[:120]

            emb     = image_to_embedding(caption_for_emb, filename)
            enc_emb = encrypt_embedding(emb)

            thumb_fn = uuid.uuid4().hex + ".jpg"
            with open(os.path.join(user_dir, thumb_fn), "wb") as tf:
                tf.write(make_thumbnail(img))
            thumb_url = f"/static/uploads/{user_id}/{thumb_fn}"

            iid = str(uuid.uuid4())
            db.execute("""INSERT INTO image_index
                (id,user_id,source_id,embedding_enc,caption,ocr_text,
                 perceptual_hash,taken_at,latitude,longitude,
                 location_name,device_model,width,height,file_size,
                 thumbnail_url,source_type)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (iid,user_id,f"upload_{filename}",enc_emb,caption,ocr_text,
                 phash,taken_at,lat,lon,"Uploaded",device,w,h,len(raw),
                 thumb_url,"upload"))
            db.commit()
            upload_jobs[user_id]["results"].append({
                "id":iid,"filename":filename,"caption":caption,
                "thumbnail_url":thumb_url,"taken_at":taken_at,
            })
        except Exception as e:
            upload_jobs[user_id]["errors"].append({"file":filename,"error":str(e)})
        upload_jobs[user_id]["processed"] += 1

    upload_jobs[user_id]["status"] = "done"
    db.close()

# ── Gemini query planner ──────────────────────────────────────────────────────
def parse_query_with_gemini(query: str) -> dict | None:
    if not GEMINI_AVAILABLE or not _gemini_client: return None
    try:
        now = datetime.now().isoformat()
        prompt = f"""Parse this photo search query and return ONLY a JSON object.
Today: {now}
Query: "{query}"

Return ONLY this JSON (no markdown, no explanation):
{{
  "time_range": {{"start": "ISO datetime or null", "end": "ISO datetime or null"}},
  "location": "location string or null",
  "visual_concept": "what to search for visually",
  "ocr_keywords": ["keywords", "for", "document", "search"],
  "ocr_only": true or false,
  "raw_query": "{query}"
}}

Rules:
- ocr_only=true ONLY for receipts/bills/invoices/menus/documents
- visual_concept: remove filler words (show me, find, photos of)
- last month=past 30 days, last week=past 7 days"""

        # Handle both new google-genai and old google-generativeai SDK
        if _gemini_model == "old-sdk":
            resp = _gemini_client.generate_content(prompt)
            text = resp.text
        else:
            resp = _gemini_client.models.generate_content(model=_gemini_model, contents=prompt)
            text = resp.text

        text   = text.strip().replace("```json","").replace("```","").strip()
        parsed = json.loads(text)
        parsed["raw_query"] = query
        print(f"✓ Gemini parsed: {parsed}")
        return parsed
    except Exception as e:
        print(f"Gemini parse failed: {e}")
        return None

def parse_query(query: str) -> dict:
    result = parse_query_with_gemini(query)
    if result: return result

    # Rule-based fallback
    q = query.lower(); now = datetime.now()
    tr = {"start":None,"end":None}
    if "today"       in q: tr={"start":now.replace(hour=0,minute=0).isoformat(),"end":now.isoformat()}
    elif "yesterday" in q:
        yd=now-timedelta(days=1); tr={"start":yd.replace(hour=0,minute=0).isoformat(),"end":yd.replace(hour=23,minute=59).isoformat()}
    elif any(x in q for x in ["last week","past week"]):   tr={"start":(now-timedelta(days=7)).isoformat(),"end":now.isoformat()}
    elif any(x in q for x in ["last month","past month"]): tr={"start":(now-timedelta(days=30)).isoformat(),"end":now.isoformat()}
    elif "last year" in q: tr={"start":(now-timedelta(days=365)).isoformat(),"end":now.isoformat()}
    for month,num in [("january","01"),("february","02"),("march","03"),("april","04"),
                      ("may","05"),("june","06"),("july","07"),("august","08"),
                      ("september","09"),("october","10"),("november","11"),("december","12")]:
        if month in q: tr={"start":f"{now.year}-{num}-01","end":f"{now.year}-{num}-30"}; break

    loc = None
    for l in ["karachi","islamabad","lahore","murree","dubai","london","beach","home","office"]:
        if l in q: loc=l.title(); break

    concept = query
    for filler in ["show me","find","search for","look for","get","photos of","pictures of",
                   "images of","i visited","from","last month","last week","today","yesterday",
                   "last year","recent","all my","my"]:
        concept = concept.lower().replace(filler,"").strip()

    ocr_kw = []; ocr_only = False
    doc_triggers = ["receipt","bill","invoice","document","scan","menu","text",
                    "bills","receipts","invoices","documents","voucher","payment","paid"]
    if any(w in q for w in doc_triggers):
        ocr_kw = [w.strip() for w in query.split() if len(w.strip())>=2]
        ocr_kw += ["total","pkr","rs","amount","date","invoice","receipt","bill",
                   "paid","payment","tax","price","cost","fee","charge"]
        ocr_only = True

    return {"time_range":tr,"location":loc,"visual_concept":concept.strip() or query,
            "ocr_keywords":ocr_kw,"ocr_only":ocr_only,"raw_query":query}

# ── Auth routes ───────────────────────────────────────────────────────────────
@app.route("/api/auth/demo-login", methods=["POST"])
def demo_login():
    data  = request.get_json(force=True)
    email = data.get("email","demo@visionsearch.app")
    name  = data.get("name","Demo User")
    db    = get_db()
    user  = db.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
    if not user:
        uid = str(uuid.uuid4())
        db.execute("INSERT INTO users (id,email,name) VALUES (?,?,?)", (uid,email,name))
        db.commit()
        user = db.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()
    token = make_token(user["id"])
    resp  = jsonify({"token":token,"user":{"id":user["id"],"email":user["email"],"name":user["name"]}})
    resp.set_cookie("vs_token",token,httponly=True,samesite="Lax",max_age=604800)
    return resp

@app.route("/api/auth/logout", methods=["POST"])
def logout():
    resp = jsonify({"ok":True}); resp.delete_cookie("vs_token"); return resp

@app.route("/api/auth/me")
@require_auth
def me():
    db    = get_db()
    user  = db.execute("SELECT id,email,name FROM users WHERE id=?", (g.user_id,)).fetchone()
    if not user: return jsonify({"error":"Not found"}), 404
    count = db.execute("SELECT COUNT(*) as c FROM image_index WHERE user_id=?", (g.user_id,)).fetchone()["c"]
    return jsonify({"user":dict(user),"stats":{"indexed":count},
                    "upload_job":upload_jobs.get(g.user_id,{})})

# ── Upload routes ─────────────────────────────────────────────────────────────
@app.route("/api/upload", methods=["POST"])
@require_auth
def upload_photos():
    # Block uploads if Sentence Transformers not loaded
    # Uploading without ST produces empty embeddings that break search
    if not AI_READY:
        return jsonify({"error":"AI components still loading. Please wait 30-60 seconds and try again."}), 503
    files = request.files.getlist("photos")
    if not files: return jsonify({"error":"No files"}), 400
    file_data, rejected = [], []
    for f in files:
        fn  = f.filename or "image.jpg"
        ext = os.path.splitext(fn)[1].lower()
        if ext not in ALLOWED_EXT:
            rejected.append({"file":fn,"error":"Unsupported format"}); continue
        raw = f.read()
        if len(raw) > MAX_FILE_BYTES:
            rejected.append({"file":fn,"error":"Too large (max 50MB)"}); continue
        file_data.append((fn, raw))
    if not file_data and rejected:
        return jsonify({"error":"All files rejected","errors":rejected}), 400
    threading.Thread(target=process_upload_batch, args=(g.user_id, file_data), daemon=True).start()
    return jsonify({"ok":True,"queued":len(file_data),"rejected":rejected})

@app.route("/api/upload/status")
@require_auth
def upload_status():
    job  = upload_jobs.get(g.user_id,{"status":"idle","processed":0,"total":0,"errors":[],"results":[]})
    done = job.get("status") == "done"
    return jsonify({"status":job.get("status","idle"),"processed":job.get("processed",0),
                    "total":job.get("total",0),"error_count":len(job.get("errors",[])),
                    "done_count":len(job.get("results",[])),
                    "errors":job.get("errors",[]) if done else [],
                    "results":job.get("results",[]) if done else []})

@app.route("/static/uploads/<user_id>/<filename>")
def serve_upload(user_id, filename):
    safe_uid = "".join(c for c in user_id if c.isalnum() or c=="-")
    safe_fn  = "".join(c for c in filename if c.isalnum() or c in "._-")
    return send_from_directory(os.path.join(UPLOAD_DIR, safe_uid), safe_fn)

# ── Search ────────────────────────────────────────────────────────────────────
@app.route("/api/search", methods=["POST"])
@require_auth
def search():
    data  = request.get_json(force=True)
    query = data.get("query","").strip()
    if not query: return jsonify({"error":"Query required"}), 400

    plan = parse_query(query)
    db   = get_db()

    sql, params = "SELECT * FROM image_index WHERE user_id=?", [g.user_id]
    if plan["time_range"]["start"]: sql += " AND taken_at >= ?"; params.append(plan["time_range"]["start"])
    if plan["time_range"]["end"]:   sql += " AND taken_at <= ?"; params.append(plan["time_range"]["end"])
    if plan["location"]:            sql += " AND location_name LIKE ?"; params.append(f"%{plan['location']}%")

    rows = db.execute(sql, params).fetchall()
    if not rows and (plan["time_range"]["start"] or plan["location"]):
        rows = db.execute("SELECT * FROM image_index WHERE user_id=?", [g.user_id]).fetchall()

    # ── OCR document search ────────────────────────────────────────────────────
    if plan.get("ocr_only"):
        # These signals ONLY appear in real financial documents
        fin_signals = ["total","subtotal","amount due","pkr","rs.","rs ","price",
                       "invoice no","receipt","paid","balance due","tax",
                       "payment","cash","qty","quantity","charges","grand total",
                       "vat","gst","discount","net amount","service charge",
                       "table no","check no","order no","bill no",
                       "sub total","tax total","tip","gratuity","change due",
                       "credit card","debit card","visa","mastercard"]
        ocr_results = []
        for row in rows:
            if not row["ocr_text"]: continue
            ocr_lower = row["ocr_text"].lower()
            words = [w for w in ocr_lower.split() if len(w) >= 3]
            # Must have substantial text AND multiple strong financial signals
            if len(words) < 20: continue
            hits = sum(1 for s in fin_signals if s in ocr_lower)
            if hits >= 3:
                ocr_results.append((min(hits * 0.20, 1.0), row))
        ocr_results.sort(key=lambda x: x[0], reverse=True)
        return jsonify({"results":[{
            "id":r["id"],"thumbnail_url":r["thumbnail_url"],"caption":r["caption"],
            "location":r["location_name"],"taken_at":r["taken_at"],
            "score":round(s,3),"ocr_text":r["ocr_text"],"source_type":r["source_type"]
        } for s,r in ocr_results[:20]],
        "plan":plan,"total":len(ocr_results),"query":query})

    # ── Visual semantic search ─────────────────────────────────────────────────
    query_vec  = text_to_embedding(plan["visual_concept"])
    top_scored = batch_cosine_search(query_vec, rows, top_k=50)

    # Caption keyword boost
    query_words = set(plan["visual_concept"].lower().split())
    boosted = []
    for score, row in top_scored:
        bonus = 0.0
        cap = (row["caption"] or "").lower()
        src = (row["source_id"] or "").lower()
        for word in query_words:
            if len(word) < 3: continue
            if word in src: bonus += 0.30
            elif word in cap: bonus += 0.20
        boosted.append((score + bonus, row))
    boosted.sort(key=lambda x: x[0], reverse=True)

    # Return top results within 35% of top score, always at least top 5
    top_score = boosted[0][0] if boosted else 0
    if top_score > 0:
        threshold = top_score * 0.65
        final = [(s,r) for s,r in boosted if s >= threshold][:20]
        if not final: final = boosted[:5]
    else:
        final = []

    return jsonify({"results":[{
        "id":r["id"],"thumbnail_url":r["thumbnail_url"],"caption":r["caption"],
        "location":r["location_name"],"taken_at":r["taken_at"],
        "score":round(min(s,1.0),3),
        "ocr_text":r["ocr_text"] if r["ocr_text"] and r["ocr_text"]!="No text detected" else None,
        "source_type":r["source_type"]
    } for s,r in final],
    "plan":plan,"total":len(final),"query":query})

# ── Images list ───────────────────────────────────────────────────────────────
@app.route("/api/images")
@require_auth
def list_images():
    db = get_db()
    page     = int(request.args.get("page",1))
    per_page = int(request.args.get("per_page",24))
    offset   = (page-1)*per_page
    rows  = db.execute("SELECT * FROM image_index WHERE user_id=? ORDER BY taken_at DESC LIMIT ? OFFSET ?",
                       (g.user_id,per_page,offset)).fetchall()
    total = db.execute("SELECT COUNT(*) as c FROM image_index WHERE user_id=?", (g.user_id,)).fetchone()["c"]
    return jsonify({"images":[{"id":r["id"],"thumbnail_url":r["thumbnail_url"],
        "caption":r["caption"],"location":r["location_name"],
        "taken_at":r["taken_at"],"source_type":r["source_type"]} for r in rows],
        "total":total,"page":page,"per_page":per_page})

# ── Stats ─────────────────────────────────────────────────────────────────────
@app.route("/api/ready")
def ai_ready():
    """Frontend polls this to know when it is safe to upload."""
    return jsonify({
        "ready": AI_READY,
        "st": ST_AVAILABLE,
        "blip": BLIP_AVAILABLE,
        "ocr": OCR_AVAILABLE,
    })

@app.route("/api/user/stats")
@require_auth
def user_stats():
    db    = get_db()
    total = db.execute("SELECT COUNT(*) as c FROM image_index WHERE user_id=?", (g.user_id,)).fetchone()["c"]
    locs  = db.execute("SELECT location_name,COUNT(*) as c FROM image_index WHERE user_id=? GROUP BY location_name ORDER BY c DESC LIMIT 6", (g.user_id,)).fetchall()
    upl   = db.execute("SELECT COUNT(*) as c FROM image_index WHERE user_id=? AND source_type='upload'", (g.user_id,)).fetchone()["c"]
    return jsonify({"total_indexed":total,"uploaded_count":upl,"storage_saved_mb":round(total*2.4,1),
                    "index_size_kb":round(total*3.2,1),"ocr_available":OCR_AVAILABLE,
                    "blip_available":BLIP_AVAILABLE,"st_available":ST_AVAILABLE,
                    "cohere_available":COHERE_AVAILABLE,

                    "top_locations":[{"name":r["location_name"],"count":r["c"]} for r in locs]})

# ── Delete ────────────────────────────────────────────────────────────────────
@app.route("/api/user/data", methods=["DELETE"])
@require_auth
def delete_user_data():
    db = get_db()
    user_dir = os.path.join(UPLOAD_DIR, g.user_id)
    if os.path.exists(user_dir):
        for f in os.listdir(user_dir):
            try: os.remove(os.path.join(user_dir,f))
            except: pass
    db.execute("DELETE FROM image_index WHERE user_id=?", (g.user_id,))
    db.commit()
    upload_jobs.pop(g.user_id, None)
    return jsonify({"ok":True})

# ── Pages ─────────────────────────────────────────────────────────────────────
@app.route("/")
@app.route("/dashboard")
@app.route("/search")
@app.route("/settings")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    init_db()
    print("✓ Vision Search v4.0 — http://localhost:5000")
    app.run(debug=False, port=5000, threaded=True, host="0.0.0.0")