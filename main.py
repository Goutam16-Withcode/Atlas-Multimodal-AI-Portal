"""
Atlas Multimodal AI Portal — FastAPI backend (v3.0.0)

Adds, on top of the original service:
  - JWT-based session tokens (HttpOnly + signed, replaces bare-UUID cookie lookups)
  - Per-IP + per-user rate limiting (sliding window, in-memory w/ pluggable backend)
  - Strict upload validation: extension allow-list, magic-byte sniffing, size caps
  - Streaming chat responses over Server-Sent Events (SSE)
  - Thread search, pinning, and soft-delete (with 30-day recovery window)
  - Markdown/PDF thread export
  - Structured, consistent error envelope for every failure path
  - Request-ID propagation + structured logging middleware
  - Centralized settings validation at startup (fails fast on misconfiguration)
  - Password strength enforcement + login lockout after repeated failures
  - CORS locked to configured origins (no wildcard in production)

Run:
    uvicorn main:app --host 127.0.0.1 --port 8000
"""

import os
import re
import io
import uuid
import time
import json
import base64
import hmac
import hashlib
import secrets
import sqlite3
import datetime
import contextlib
from typing import List, Optional, Dict, Any, AsyncGenerator

import requests
from fastapi import (
    FastAPI, HTTPException, Response, Cookie, UploadFile, File,
    Request, Depends, Query, status
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, field_validator

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from graph import chatbot
from logger import get_logger
from config import settings
from asset_export import (
    parse_presentation_markup, parse_poster_markup,
    build_pptx, build_poster_pdf,
)

logger = get_logger("api")

# --------------------------------------------------------------------------- #
# Constants & configuration
# --------------------------------------------------------------------------- #

JWT_SECRET = getattr(settings, "JWT_SECRET", None) or secrets.token_hex(32)
JWT_ALGO = "HS256"
SESSION_TTL_SECONDS = 60 * 60 * 24 * 7          # 7 days
GUEST_TTL_SECONDS = 60 * 60 * 24                # 1 day

MAX_UPLOAD_BYTES = 25 * 1024 * 1024             # 25 MB
ALLOWED_UPLOAD_EXT = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp",
    ".mp3", ".wav", ".m4a", ".ogg",
    ".mp4", ".mov", ".avi", ".webm", ".mkv",
    ".pdf", ".txt", ".csv",
}
# Magic-byte signatures for the file types we actually trust the extension of.
MAGIC_SIGNATURES = {
    b"\xff\xd8\xff": "image",                    # JPEG
    b"\x89PNG\r\n\x1a\n": "image",                # PNG
    b"GIF87a": "image", b"GIF89a": "image",       # GIF
    b"RIFF": "media",                             # WAV/AVI/WEBP container
    b"%PDF-": "file",                             # PDF
    b"\x1a\x45\xdf\xa3": "media",                 # WEBM/MKV (EBML)
    b"ID3": "audio",                              # MP3 w/ ID3 tag
    b"\xff\xfb": "audio", b"\xff\xf3": "audio", b"\xff\xf2": "audio",  # raw MP3
}

MAX_LOGIN_ATTEMPTS = 5
LOGIN_LOCKOUT_SECONDS = 15 * 60

RATE_LIMIT_WINDOW_SECONDS = 60
RATE_LIMIT_MAX_REQUESTS = {
    "/chat": 20,
    "/upload": 15,
    "/login": 10,
    "/register": 5,
}
DEFAULT_RATE_LIMIT = 120

os.makedirs(os.path.join("static", "uploads"), exist_ok=True)

# --------------------------------------------------------------------------- #
# Password hashing
# --------------------------------------------------------------------------- #

PASSWORD_MIN_LEN = 8
PASSWORD_RULES = re.compile(r"^(?=.*[A-Za-z])(?=.*\d).{8,}$")


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    db_hash = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100_000)
    return salt.hex() + ":" + db_hash.hex()


def verify_password(password: str, stored: str) -> bool:
    try:
        salt_hex, hash_hex = stored.split(":")
        salt = bytes.fromhex(salt_hex)
        db_hash = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100_000)
        return secrets.compare_digest(db_hash.hex(), hash_hex)
    except Exception:
        return False


def validate_password_strength(password: str) -> Optional[str]:
    if len(password) < PASSWORD_MIN_LEN:
        return f"Password must be at least {PASSWORD_MIN_LEN} characters."
    if not PASSWORD_RULES.match(password):
        return "Password must contain at least one letter and one number."
    return None


# --------------------------------------------------------------------------- #
# Database
# --------------------------------------------------------------------------- #

def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(settings.SQLITE_DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA cache_size=-64000;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


@contextlib.contextmanager
def db_cursor():
    """Context-managed cursor: commits on success, rolls back on error, always closes."""
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        yield cur
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_metadata_db():
    try:
        with db_cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    username TEXT PRIMARY KEY,
                    password_hash TEXT,
                    created_at TEXT,
                    failed_attempts INTEGER DEFAULT 0,
                    locked_until TEXT
                )
            """)
            # Verify and update old users table structure
            for ddl in (
                "ALTER TABLE users ADD COLUMN failed_attempts INTEGER DEFAULT 0",
                "ALTER TABLE users ADD COLUMN locked_until TEXT",
            ):
                try:
                    cursor.execute(ddl)
                except sqlite3.OperationalError:
                    pass

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    username TEXT,
                    expires_at TEXT,
                    created_at TEXT,
                    user_agent TEXT
                )
            """)
            for ddl in (
                "ALTER TABLE sessions ADD COLUMN created_at TEXT",
                "ALTER TABLE sessions ADD COLUMN user_agent TEXT",
            ):
                try:
                    cursor.execute(ddl)
                except sqlite3.OperationalError:
                    pass

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS thread_metadata (
                    thread_id TEXT PRIMARY KEY,
                    title TEXT,
                    updated_at TEXT,
                    username TEXT,
                    pinned INTEGER DEFAULT 0,
                    deleted_at TEXT
                )
            """)
            for ddl in (
                "ALTER TABLE thread_metadata ADD COLUMN username TEXT",
                "ALTER TABLE thread_metadata ADD COLUMN pinned INTEGER DEFAULT 0",
                "ALTER TABLE thread_metadata ADD COLUMN deleted_at TEXT",
            ):
                try:
                    cursor.execute(ddl)
                except sqlite3.OperationalError:
                    pass

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS long_term_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT,
                    entity TEXT,
                    relation TEXT,
                    target TEXT,
                    created_at TEXT
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT,
                    action TEXT,
                    detail TEXT,
                    ip_address TEXT,
                    created_at TEXT
                )
            """)

            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_username ON sessions(username)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_thread_metadata_username ON thread_metadata(username)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_thread_metadata_deleted ON thread_metadata(deleted_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_long_term_memory_username ON long_term_memory(username)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_long_term_memory_entity ON long_term_memory(entity)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_username ON audit_log(username)")

        logger.info("Metadata database initialized (schema v3).")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise


# --------------------------------------------------------------------------- #
# Custom standard-compliant JWT implementation
# --------------------------------------------------------------------------- #

def base64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b'=').decode('utf-8')


def base64url_decode(data: str) -> bytes:
    padding = '=' * (4 - (len(data) % 4))
    return base64.urlsafe_b64decode(data + padding)


def issue_session_token(username: str, ttl_seconds: int, session_id: str) -> str:
    header = {"alg": JWT_ALGO, "typ": "JWT"}
    now = int(time.time())
    payload = {
        "sub": username,
        "jti": session_id,
        "iat": now,
        "exp": now + ttl_seconds,
    }
    header_json = json.dumps(header, separators=(',', ':')).encode('utf-8')
    payload_json = json.dumps(payload, separators=(',', ':')).encode('utf-8')
    
    header_b64 = base64url_encode(header_json)
    payload_b64 = base64url_encode(payload_json)
    
    signing_input = f"{header_b64}.{payload_b64}".encode('utf-8')
    signature = hmac.new(JWT_SECRET.encode('utf-8'), signing_input, hashlib.sha256).digest()
    signature_b64 = base64url_encode(signature)
    
    return f"{header_b64}.{payload_b64}.{signature_b64}"


def decode_session_token(token: str) -> Optional[Dict[str, Any]]:
    try:
        parts = token.split('.')
        if len(parts) != 3:
            return None
        header_b64, payload_b64, signature_b64 = parts
        
        signing_input = f"{header_b64}.{payload_b64}".encode('utf-8')
        expected_signature = hmac.new(JWT_SECRET.encode('utf-8'), signing_input, hashlib.sha256).digest()
        
        if not secrets.compare_digest(base64url_encode(expected_signature), signature_b64):
            return None
            
        payload_bytes = base64url_decode(payload_b64)
        payload = json.loads(payload_bytes.decode('utf-8'))
        
        if payload.get("exp", 0) < int(time.time()):
            return None
            
        # Verify session is still valid in SQLite DB
        jti = payload.get("jti")
        if not jti:
            return None
        with db_cursor() as cursor:
            cursor.execute("SELECT expires_at FROM sessions WHERE session_id = ?", (jti,))
            row = cursor.fetchone()
            if not row:
                return None
            expires_dt = datetime.datetime.fromisoformat(row[0])
            if datetime.datetime.utcnow() > expires_dt:
                return None

        return payload
    except Exception:
        return None


def create_session(username: str, ttl_seconds: int, user_agent: str = "") -> str:
    session_id = str(uuid.uuid4())
    expires = (datetime.datetime.utcnow() + datetime.timedelta(seconds=ttl_seconds)).isoformat()
    with db_cursor() as cursor:
        cursor.execute(
            "INSERT INTO sessions (session_id, username, expires_at, created_at, user_agent) VALUES (?, ?, ?, ?, ?)",
            (session_id, username, expires, datetime.datetime.utcnow().isoformat(), user_agent),
        )
    token = issue_session_token(username, ttl_seconds, session_id)
    return token


def get_current_user(session_id: Optional[str]) -> str:
    """Validates the JWT session cookie. Raises 401 on any failure."""
    if not session_id:
        raise HTTPException(status_code=401, detail="Not authenticated")
    payload = decode_session_token(session_id)
    if not payload or "sub" not in payload:
        raise HTTPException(status_code=401, detail="Session expired or invalid")
    return payload["sub"]


# --------------------------------------------------------------------------- #
# Pure-Python PDF exporter
# --------------------------------------------------------------------------- #

def generate_simple_pdf(title: str, lines: List[str]) -> bytes:
    """Generates a standard compliant PDF document from plain text lines without dependencies."""
    pdf = bytearray()
    pdf.extend(b"%PDF-1.4\n")
    
    objects = []
    def add_object(obj_bytes: bytes) -> int:
        idx = len(objects) + 1
        objects.append(len(pdf))
        pdf.extend(f"{idx} 0 obj\n".encode('utf-8'))
        pdf.extend(obj_bytes)
        pdf.extend(b"\nendobj\n")
        return idx

    # Obj 1: Catalog, Obj 2: Pages list
    catalog_id = 1
    pages_id = 2
    
    # Wrap text lines to fit A4 page
    content_lines = []
    content_lines.append("BT")
    content_lines.append("/F1 12 Tf")
    content_lines.append("14 TL")
    content_lines.append("50 780 Td")
    
    y = 780
    for line in lines:
        clean_line = line.replace("(", "\\(").replace(")", "\\)")
        words = clean_line.split()
        curr_line = ""
        for word in words:
            if len(curr_line) + len(word) > 75:
                content_lines.append(f"({curr_line}) Tj T*")
                y -= 14
                curr_line = word
            else:
                curr_line = f"{curr_line} {word}".strip()
        if curr_line:
            content_lines.append(f"({curr_line}) Tj T*")
            y -= 14
        
        # Simple spacing between paragraphs/lines
        content_lines.append("() Tj T*")
        y -= 14
            
    content_lines.append("ET")
    content_stream = "\n".join(content_lines).encode('utf-8')
    
    obj1 = b"<< /Type /Catalog /Pages 2 0 R >>"
    obj2 = b"<< /Type /Pages /Kids [ 5 0 R ] /Count 1 >>"
    obj3 = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
    obj4 = f"<< /Length {len(content_stream)} >>\nstream\n".encode('utf-8') + content_stream + b"\nendstream"
    obj5 = b"<< /Type /Page /Parent 2 0 R /MediaBox [ 0 0 595.27 841.89 ] /Contents 4 0 R /Resources << /Font << /F1 3 0 R >> >> >>"
    
    add_object(obj1)
    add_object(obj2)
    add_object(obj3)
    add_object(obj4)
    add_object(obj5)
    
    startxref = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode('utf-8'))
    for offset in objects:
        pdf.extend(f"{offset:010d} 00000 n \n".encode('utf-8'))
        
    pdf.extend(f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{startxref}\n%%EOF\n".encode('utf-8'))
    return bytes(pdf)


# --------------------------------------------------------------------------- #
# Rate limiting (sliding window, in-memory)
# --------------------------------------------------------------------------- #

class RateLimiter:
    def __init__(self):
        self._hits: Dict[str, List[float]] = {}

    def check(self, key: str, limit: int, window: int = RATE_LIMIT_WINDOW_SECONDS) -> bool:
        now = time.time()
        bucket = self._hits.setdefault(key, [])
        cutoff = now - window
        while bucket and bucket[0] < cutoff:
            bucket.pop(0)
        if len(bucket) >= limit:
            return False
        bucket.append(now)
        return True


rate_limiter = RateLimiter()


def enforce_rate_limit(request: Request, identity: str, path: str):
    limit = RATE_LIMIT_MAX_REQUESTS.get(path, DEFAULT_RATE_LIMIT)
    key = f"{identity}:{path}"
    if not rate_limiter.check(key, limit):
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded for {path}. Try again shortly.",
        )


def client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


# --------------------------------------------------------------------------- #
# File extraction helpers
# --------------------------------------------------------------------------- #

def extract_text_from_pdf(filepath: str) -> str:
    if not os.path.exists(filepath):
        return "[Error: PDF file not found]"
    try:
        import pypdf
        reader = pypdf.PdfReader(filepath)
        text = ""
        max_pages = min(len(reader.pages), 15)
        for i in range(max_pages):
            page_text = reader.pages[i].extract_text()
            if page_text:
                text += f"\n--- Page {i + 1} ---\n{page_text}"
        return text.strip() or "[Empty PDF content]"
    except Exception as e:
        logger.error(f"Failed to parse PDF using pypdf: {e}")
        try:
            with open(filepath, "rb") as f:
                content = f.read()
            import string
            printable = set(string.printable.encode("ascii"))
            text_list, current = [], bytearray()
            for b in content:
                if b in printable:
                    current.append(b)
                else:
                    if len(current) > 4:
                        text_list.append(current.decode("ascii", errors="ignore"))
                    current = bytearray()
            fallback_text = " ".join(text_list[:1000])
            return f"[Fallback Extraction] {fallback_text}"
        except Exception as e_fallback:
            return f"[Error extracting PDF text: {str(e_fallback)}]"


def transcribe_audio_file(filepath: str) -> str:
    if not os.path.exists(filepath):
        return "[Error: Audio file not found]"
    try:
        url = "https://api.groq.com/openai/v1/audio/transcriptions"
        headers = {"Authorization": f"Bearer {settings.GROQ_API_KEY}"}
        with open(filepath, "rb") as f:
            files = {"file": (os.path.basename(filepath), f)}
            data = {"model": "whisper-large-v3"}
            response = requests.post(url, headers=headers, files=files, data=data, timeout=60)
        if response.status_code == 200:
            return response.json().get("text", "")
        logger.error(f"Whisper transcription failed: {response.text}")
        return f"[Error transcribing media file: {response.reason}]"
    except Exception as e:
        logger.error(f"Failed to transcribe media: {e}")
        return f"[Error transcribing media: {str(e)}]"


def sniff_file_type(head: bytes) -> Optional[str]:
    for sig, kind in MAGIC_SIGNATURES.items():
        if head.startswith(sig):
            return kind
    return None


def write_audit(username: str, action: str, detail: str = "", ip: str = ""):
    try:
        with db_cursor() as cursor:
            cursor.execute(
                "INSERT INTO audit_log (username, action, detail, ip_address, created_at) VALUES (?, ?, ?, ?, ?)",
                (username, action, detail, ip, datetime.datetime.utcnow().isoformat()),
            )
    except Exception as e:
        logger.warning(f"Audit log write failed: {e}")


# --------------------------------------------------------------------------- #
# App setup
# --------------------------------------------------------------------------- #

app = FastAPI(title="Atlas Multimodal AI Portal", version="3.0.0")

ALLOWED_ORIGINS = getattr(settings, "ALLOWED_ORIGINS", None) or ["http://localhost:8000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    start = time.time()
    try:
        response = await call_next(request)
    except Exception as e:
        logger.error(f"[{request_id}] Unhandled error on {request.url.path}: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": {"code": "internal_error", "message": "Internal server error", "request_id": request_id}},
        )
    duration_ms = round((time.time() - start) * 1000, 1)
    response.headers["x-request-id"] = request_id
    logger.info(f"[{request_id}] {request.method} {request.url.path} -> {response.status_code} ({duration_ms}ms)")
    return response


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": exc.status_code, "message": exc.detail}},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"error": {"code": 422, "message": "Validation failed", "details": exc.errors()}},
    )


# --------------------------------------------------------------------------- #
# Pydantic models
# --------------------------------------------------------------------------- #

class AuthRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=32)
    password: str = Field(..., min_length=1, max_length=256)

    @field_validator("username")
    @classmethod
    def username_charset(cls, v: str) -> str:
        v = v.strip()
        if not re.match(r"^[A-Za-z0-9_.-]+$", v):
            raise ValueError("Username may only contain letters, numbers, underscore, dot, and dash.")
        return v


class AttachmentItem(BaseModel):
    type: str
    url: str
    filename: Optional[str] = None


class ChatRequest(BaseModel):
    thread_id: str = Field(..., min_length=1, max_length=128)
    message: str = Field("", max_length=20_000)
    attachments: Optional[List[AttachmentItem]] = []
    language: Optional[str] = "auto"
    stream: Optional[bool] = False


class ChatResponse(BaseModel):
    thread_id: str
    reply: str
    intent: Optional[str] = None
    needs_escalation: bool = False


class RenameThreadRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)


class ThreadOut(BaseModel):
    thread_id: str
    title: str
    updated_at: str
    pinned: bool = False


# --------------------------------------------------------------------------- #
# Thread helpers
# --------------------------------------------------------------------------- #

def update_thread_title_and_time(thread_id: str, first_message: str = None, username: str = None):
    try:
        with db_cursor() as cursor:
            cursor.execute("SELECT title FROM thread_metadata WHERE thread_id = ? AND deleted_at IS NULL", (thread_id,))
            row = cursor.fetchone()
            now = datetime.datetime.utcnow().isoformat()
            if row:
                cursor.execute("UPDATE thread_metadata SET updated_at = ? WHERE thread_id = ?", (now, thread_id))
            else:
                title = thread_id
                if first_message:
                    clean_msg = first_message.strip()
                    title = clean_msg[:40] + ("..." if len(clean_msg) > 40 else "")
                cursor.execute("""
                    INSERT OR REPLACE INTO thread_metadata (thread_id, title, updated_at, username, pinned, deleted_at)
                    VALUES (?, ?, ?, ?, 0, NULL)
                """, (thread_id, title, now, username))
    except Exception as e:
        logger.error(f"Error updating thread metadata for {thread_id}: {e}")


def get_thread_owner(thread_id: str) -> Optional[str]:
    with db_cursor() as cursor:
        cursor.execute("SELECT username FROM thread_metadata WHERE thread_id = ?", (thread_id,))
        row = cursor.fetchone()
        return row[0] if row else None


def assert_thread_access(thread_id: str, username: str):
    owner = get_thread_owner(thread_id)
    if owner and owner != username:
        raise HTTPException(status_code=403, detail="Forbidden: Thread belongs to another user")


# --------------------------------------------------------------------------- #
# Auth routes
# --------------------------------------------------------------------------- #

@app.get("/me")
def get_me(response: Response, session_id: Optional[str] = Cookie(None)):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    if not session_id:
        return {"authenticated": False}
    payload = decode_session_token(session_id)
    if not payload:
        return {"authenticated": False}
    return {"authenticated": True, "username": payload["sub"]}


@app.post("/register")
def register(req: AuthRequest, response: Response, request: Request):
    enforce_rate_limit(request, client_ip(request), "/register")
    username = req.username.strip()
    password = req.password

    weakness = validate_password_strength(password)
    if weakness:
        raise HTTPException(status_code=400, detail=weakness)

    try:
        with db_cursor() as cursor:
            cursor.execute("SELECT username FROM users WHERE username = ?", (username,))
            if cursor.fetchone():
                raise HTTPException(status_code=400, detail="Username already exists")

            pwd_hash = hash_password(password)
            cursor.execute(
                "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
                (username, pwd_hash, datetime.datetime.utcnow().isoformat()),
            )

        token = create_session(username, SESSION_TTL_SECONDS, request.headers.get("user-agent", ""))
        write_audit(username, "register", ip=client_ip(request))

        response.set_cookie(
            key="session_id", value=token, httponly=True, samesite="lax",
            secure=getattr(settings, "COOKIE_SECURE", False), max_age=SESSION_TTL_SECONDS,
        )
        return {"status": "ok", "username": username}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during registration")


@app.post("/login")
def login(req: AuthRequest, response: Response, request: Request):
    enforce_rate_limit(request, client_ip(request), "/login")
    username = req.username.strip()
    password = req.password

    try:
        with db_cursor() as cursor:
            cursor.execute(
                "SELECT password_hash, failed_attempts, locked_until FROM users WHERE username = ?",
                (username,),
            )
            row = cursor.fetchone()

            if row:
                _, failed_attempts, locked_until = row
                if locked_until:
                    locked_dt = datetime.datetime.fromisoformat(locked_until)
                    if datetime.datetime.utcnow() < locked_dt:
                        raise HTTPException(
                            status_code=423,
                            detail="Account temporarily locked due to repeated failed logins. Try again later.",
                        )

            if not row or not verify_password(password, row[0]):
                if row:
                    new_attempts = (row[1] or 0) + 1
                    lock_value = None
                    if new_attempts >= MAX_LOGIN_ATTEMPTS:
                        lock_value = (
                            datetime.datetime.utcnow() + datetime.timedelta(seconds=LOGIN_LOCKOUT_SECONDS)
                        ).isoformat()
                    cursor.execute(
                        "UPDATE users SET failed_attempts = ?, locked_until = ? WHERE username = ?",
                        (new_attempts, lock_value, username),
                    )
                write_audit(username, "login_failed", ip=client_ip(request))
                raise HTTPException(status_code=401, detail="Invalid username or password")

            cursor.execute(
                "UPDATE users SET failed_attempts = 0, locked_until = NULL WHERE username = ?",
                (username,),
            )

        token = create_session(username, SESSION_TTL_SECONDS, request.headers.get("user-agent", ""))
        write_audit(username, "login", ip=client_ip(request))

        response.set_cookie(
            key="session_id", value=token, httponly=True, samesite="lax",
            secure=getattr(settings, "COOKIE_SECURE", False), max_age=SESSION_TTL_SECONDS,
        )
        return {"status": "ok", "username": username}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during login")


@app.post("/guest")
def login_guest(response: Response, request: Request):
    enforce_rate_limit(request, client_ip(request), "/guest")
    username = f"guest_{uuid.uuid4().hex[:6]}"
    try:
        with db_cursor() as cursor:
            cursor.execute(
                "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
                (username, "guest", datetime.datetime.utcnow().isoformat()),
            )

        token = create_session(username, GUEST_TTL_SECONDS, request.headers.get("user-agent", ""))
        write_audit(username, "guest_login", ip=client_ip(request))

        response.set_cookie(
            key="session_id", value=token, httponly=True, samesite="lax",
            secure=getattr(settings, "COOKIE_SECURE", False), max_age=GUEST_TTL_SECONDS,
        )
        return {"status": "ok", "username": username}
    except Exception as e:
        logger.error(f"Guest session error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during guest sign in")


@app.post("/logout")
def logout(response: Response, session_id: Optional[str] = Cookie(None)):
    if session_id:
        payload = decode_session_token(session_id)
        if payload:
            write_audit(payload["sub"], "logout")
            jti = payload.get("jti")
            if jti:
                try:
                    with db_cursor() as cursor:
                        cursor.execute("DELETE FROM sessions WHERE session_id = ?", (jti,))
                except Exception as e:
                    logger.warning(f"Failed to delete session {jti} on logout: {e}")
    response.delete_cookie(
        "session_id",
        path="/",
        httponly=True,
        samesite="lax",
        secure=settings.COOKIE_SECURE,
    )
    return {"status": "ok"}


# --------------------------------------------------------------------------- #
# Upload route
# --------------------------------------------------------------------------- #

@app.post("/upload")
def upload_file(
    request: Request,
    file: UploadFile = File(...),
    session_id: Optional[str] = Cookie(None),
):
    username = get_current_user(session_id)
    enforce_rate_limit(request, username, "/upload")

    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_UPLOAD_EXT:
        raise HTTPException(status_code=400, detail=f"File type '{ext}' is not allowed.")

    contents = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File exceeds max size of {MAX_UPLOAD_BYTES // (1024*1024)}MB.")
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # Strict magic-byte sniffing validation
    sniffed_type = sniff_file_type(contents[:16])
    if ext in [".png", ".jpg", ".jpeg", ".gif", ".webp"]:
        if sniffed_type not in ["image", "media"]:
            raise HTTPException(status_code=400, detail="Uploaded file contains invalid image magic signatures.")
    elif ext == ".pdf":
        if sniffed_type != "file":
            raise HTTPException(status_code=400, detail="Uploaded file contains invalid PDF magic signatures.")
    elif ext in [".wav", ".avi"]:
        if sniffed_type != "media":
            raise HTTPException(status_code=400, detail="Uploaded file contains invalid container magic signatures.")
    elif ext in [".mp3", ".ogg"]:
        if sniffed_type not in ["audio", "media"]:
            raise HTTPException(status_code=400, detail="Uploaded file contains invalid audio magic signatures.")

    unique_filename = f"{uuid.uuid4().hex}{ext}"
    filepath = os.path.join("static", "uploads", unique_filename)

    try:
        with open(filepath, "wb") as f:
            f.write(contents)

        url_path = f"/static/uploads/{unique_filename}"

        if ext in [".png", ".jpg", ".jpeg", ".gif", ".webp"]:
            file_type = "image"
        elif ext in [".mp3", ".wav", ".m4a", ".ogg"]:
            file_type = "audio"
        elif ext in [".mp4", ".mov", ".avi", ".webm", ".mkv"]:
            file_type = "video"
        else:
            file_type = "file"

        write_audit(username, "upload", detail=file.filename, ip=client_ip(request))
        return {"url": url_path, "filename": file.filename, "type": file_type, "size": len(contents)}
    except Exception as e:
        logger.error(f"Failed to save upload: {e}")
        raise HTTPException(status_code=500, detail="Failed to write file to storage")


# --------------------------------------------------------------------------- #
# Fact extraction (long-term memory)
# --------------------------------------------------------------------------- #

def extract_and_save_facts(username: str, user_message: str):
    if not user_message or len(user_message.strip()) < 10:
        return
    if "[SCADA TELEMETRY WARNING]" in user_message or "SCADA Alarm alert received:" in user_message:
        return

    prompt = [
        SystemMessage(content=(
            "You are a factual entity-relation extractor. Extract permanent facts about the user or equipment "
            "mentioned in this message. Only extract facts that are useful to remember long-term (e.g. user job roles, "
            "plant location/shift, equipment faults, or user preferences). Do not extract conversational noise.\n"
            "Format the output as a clean JSON list of objects with keys: entity, relation, target. If no facts, output an empty list.\n"
            "Do not output any introductory or conversational text, markdown formatting (no code blocks), just raw JSON.\n"
            "Example input: 'I work on line 3 and my pump-2 is leaking'\n"
            "Example output: [{\"entity\": \"user\", \"relation\": \"works_on\", \"target\": \"line 3\"}, {\"entity\": \"pump-2\", \"relation\": \"status\", \"target\": \"leaking\"}]"
        )),
        HumanMessage(content=user_message),
    ]
    try:
        from langchain_groq import ChatGroq
        extractor = ChatGroq(model="llama-3.1-8b-instant", temperature=0.0, api_key=settings.GROQ_API_KEY or None)
        res = extractor.invoke(prompt)
        text = (res.content or "").strip()
        if not text:
            return

        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        if not text:
            return

        try:
            facts = json.loads(text)
        except json.JSONDecodeError:
            logger.warning(f"Fact extraction failed to decode JSON: {text}")
            return

        if isinstance(facts, list) and len(facts) > 0:
            with db_cursor() as cursor:
                for f in facts:
                    entity = str(f.get("entity", "")).strip()
                    relation = str(f.get("relation", "")).strip()
                    target = str(f.get("target", "")).strip()
                    if entity and relation and target:
                        cursor.execute("""
                            INSERT INTO long_term_memory (username, entity, relation, target, created_at)
                            VALUES (?, ?, ?, ?, ?)
                        """, (username, entity, relation, target, datetime.datetime.utcnow().isoformat()))
            logger.info(f"Factual memory: extracted {len(facts)} facts for user {username}")
    except Exception as e:
        logger.warning(f"Fact extraction failed: {e}")


# --------------------------------------------------------------------------- #
# Multimodal message assembly
# --------------------------------------------------------------------------- #

def build_human_message(req: ChatRequest):
    has_image = False
    image_parts = []
    transcriptions = []
    text_content = req.message

    for att in req.attachments:
        local_path = att.url.lstrip("/")
        resolved = os.path.realpath(local_path)
        allowed_root = os.path.realpath("static")
        if not resolved.startswith(allowed_root) or not os.path.exists(resolved):
            continue

        if att.type == "image":
            has_image = True
            with open(resolved, "rb") as img_file:
                b64_string = base64.b64encode(img_file.read()).decode("utf-8")
            image_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64_string}"},
            })
        elif att.type in ["audio", "video"]:
            transcription = transcribe_audio_file(resolved)
            transcriptions.append(f"[Uploaded {att.type} transcription for '{att.filename or 'file'}': '{transcription}']")
        elif att.type == "file" or resolved.lower().endswith(".pdf"):
            pdf_text = extract_text_from_pdf(resolved)
            transcriptions.append(f"[Content of uploaded document '{att.filename or 'document.pdf'}']:\n{pdf_text}")

    if transcriptions:
        text_content += "\n\n" + "\n".join(transcriptions)

    if has_image:
        content_list = [{"type": "text", "text": text_content}] + image_parts
        return HumanMessage(content=content_list)
    return HumanMessage(content=text_content)


# --------------------------------------------------------------------------- #
# Chat routes
# --------------------------------------------------------------------------- #

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, request: Request, session_id: Optional[str] = Cookie(None)):
    username = get_current_user(session_id)
    enforce_rate_limit(request, username, "/chat")

    if not req.message.strip() and not req.attachments:
        raise HTTPException(status_code=400, detail="Message or attachment required")

    update_thread_title_and_time(req.thread_id, req.message, username)
    assert_thread_access(req.thread_id, username)

    msg = build_human_message(req)

    config = {
        "configurable": {
            "thread_id": req.thread_id,
            "username": username,
            "language": req.language or "auto",
        },
        "metadata": {
            "username": username,
            "thread_id": req.thread_id,
            "language": req.language or "auto",
            "interface": "web"
        }
    }
    try:
        result = chatbot.invoke({"messages": [msg]}, config=config)
    except Exception as e:
        logger.error(f"Graph invocation failed: {e}")
        raise HTTPException(status_code=500, detail="Internal chatbot error") from e

    try:
        extract_and_save_facts(username, req.message)
    except Exception as fact_err:
        logger.warning(f"Background fact extraction setup error: {fact_err}")

    reply = result["messages"][-1]
    write_audit(username, "chat", detail=req.thread_id, ip=client_ip(request))

    return ChatResponse(
        thread_id=req.thread_id,
        reply=reply.content,
        intent=result.get("intent"),
        needs_escalation=result.get("needs_escalation", False),
    )


@app.post("/chat/stream")
def chat_stream(req: ChatRequest, request: Request, session_id: Optional[str] = Cookie(None)):
    username = get_current_user(session_id)
    enforce_rate_limit(request, username, "/chat")

    if not req.message.strip() and not req.attachments:
        raise HTTPException(status_code=400, detail="Message or attachment required")

    update_thread_title_and_time(req.thread_id, req.message, username)
    assert_thread_access(req.thread_id, username)

    msg = build_human_message(req)
    config = {
        "configurable": {
            "thread_id": req.thread_id,
            "username": username,
            "language": req.language or "auto",
        },
        "metadata": {
            "username": username,
            "thread_id": req.thread_id,
            "language": req.language or "auto",
            "interface": "web"
        }
    }

    def sse_event(event: str, data: str) -> str:
        safe = data.replace("\n", "\\n")
        return f"event: {event}\ndata: {safe}\n\n"

    def generator():
        try:
            final_state = None
            if hasattr(chatbot, "stream"):
                for chunk in chatbot.stream({"messages": [msg]}, config=config, stream_mode="values"):
                    final_state = chunk
                    msgs = chunk.get("messages") or []
                    if msgs:
                        last = msgs[-1]
                        if isinstance(last, AIMessage) and last.content:
                            yield sse_event("token", last.content)
            if final_state is None:
                final_state = chatbot.invoke({"messages": [msg]}, config=config)

            reply = final_state["messages"][-1]
            payload = {
                "thread_id": req.thread_id,
                "reply": reply.content,
                "intent": final_state.get("intent"),
                "needs_escalation": final_state.get("needs_escalation", False),
            }
            try:
                extract_and_save_facts(username, req.message)
            except Exception as fact_err:
                logger.warning(f"Background fact extraction setup error: {fact_err}")

            yield sse_event("done", json.dumps(payload))
            write_audit(username, "chat_stream", detail=req.thread_id, ip=client_ip(request))
        except Exception as e:
            logger.error(f"Streaming chat failed: {e}")
            yield sse_event("error", json.dumps({"detail": "Internal chatbot error"}))

    return StreamingResponse(generator(), media_type="text/event-stream")


# --------------------------------------------------------------------------- #
# Thread management routes
# --------------------------------------------------------------------------- #

@app.get("/threads")
def list_threads(
    session_id: Optional[str] = Cookie(None),
    q: Optional[str] = Query(None, description="Search threads by title"),
    include_deleted: bool = Query(False),
):
    username = get_current_user(session_id)
    
    # 30-day soft-delete auto-purge window
    thirty_days_ago = (datetime.datetime.utcnow() - datetime.timedelta(days=30)).isoformat()
    try:
        with db_cursor() as cursor:
            cursor.execute("SELECT thread_id FROM thread_metadata WHERE deleted_at IS NOT NULL AND deleted_at < ?", (thirty_days_ago,))
            expired = [r[0] for r in cursor.fetchall()]
            for tid in expired:
                cursor.execute("DELETE FROM checkpoints WHERE thread_id = ?", (tid,))
                cursor.execute("DELETE FROM writes WHERE thread_id = ?", (tid,))
                cursor.execute("DELETE FROM thread_metadata WHERE thread_id = ?", (tid,))
    except Exception as e:
        logger.warning(f"Failed to auto-purge 30-day expired threads: {e}")

    try:
        with db_cursor() as cursor:
            sql = "SELECT thread_id, title, updated_at, pinned, deleted_at FROM thread_metadata WHERE username = ?"
            params: List[Any] = [username]
            if not include_deleted:
                sql += " AND deleted_at IS NULL"
            if q:
                sql += " AND title LIKE ?"
                params.append(f"%{q}%")
            cursor.execute(sql, params)
            rows = cursor.fetchall()

        out = [
            {"thread_id": r[0], "title": r[1], "updated_at": r[2], "pinned": bool(r[3]), "deleted_at": r[4]}
            for r in rows
        ]
        out.sort(key=lambda x: (not x["pinned"], x["updated_at"] or ""), reverse=False)
        out.sort(key=lambda x: x["pinned"], reverse=True)
        return {"threads": out}
    except Exception as e:
        logger.error(f"Failed to list threads for {username}: {e}")
        return {"threads": []}


@app.put("/threads/{thread_id}")
def rename_thread(thread_id: str, req: RenameThreadRequest, session_id: Optional[str] = Cookie(None)):
    username = get_current_user(session_id)
    assert_thread_access(thread_id, username)
    try:
        now = datetime.datetime.utcnow().isoformat()
        with db_cursor() as cursor:
            cursor.execute("""
                INSERT OR REPLACE INTO thread_metadata (thread_id, title, updated_at, username, pinned, deleted_at)
                VALUES (?, ?, ?, ?, COALESCE((SELECT pinned FROM thread_metadata WHERE thread_id = ?), 0), NULL)
            """, (thread_id, req.title, now, username, thread_id))
        return {"status": "ok"}
    except Exception as e:
        logger.error(f"Failed to rename thread {thread_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/threads/{thread_id}/pin")
def pin_thread(thread_id: str, pinned: bool = Query(True), session_id: Optional[str] = Cookie(None)):
    username = get_current_user(session_id)
    assert_thread_access(thread_id, username)
    try:
        with db_cursor() as cursor:
            cursor.execute(
                "UPDATE thread_metadata SET pinned = ? WHERE thread_id = ?",
                (1 if pinned else 0, thread_id),
            )
        return {"status": "ok", "pinned": pinned}
    except Exception as e:
        logger.error(f"Failed to pin thread {thread_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/threads/{thread_id}")
def delete_thread(
    thread_id: str,
    session_id: Optional[str] = Cookie(None),
    hard: bool = Query(False, description="Permanently delete instead of soft-delete"),
):
    username = get_current_user(session_id)
    assert_thread_access(thread_id, username)
    try:
        with db_cursor() as cursor:
            if hard:
                cursor.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
                cursor.execute("DELETE FROM writes WHERE thread_id = ?", (thread_id,))
                cursor.execute("DELETE FROM thread_metadata WHERE thread_id = ?", (thread_id,))
            else:
                cursor.execute(
                    "UPDATE thread_metadata SET deleted_at = ? WHERE thread_id = ?",
                    (datetime.datetime.utcnow().isoformat(), thread_id),
                )
        return {"status": "ok", "hard_deleted": hard}
    except sqlite3.OperationalError:
        with db_cursor() as cursor:
            cursor.execute(
                "UPDATE thread_metadata SET deleted_at = ? WHERE thread_id = ?",
                (datetime.datetime.utcnow().isoformat(), thread_id),
            )
        return {"status": "ok", "hard_deleted": False}
    except Exception as e:
        logger.error(f"Failed to delete thread {thread_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/threads/{thread_id}/restore")
def restore_thread(thread_id: str, session_id: Optional[str] = Cookie(None)):
    username = get_current_user(session_id)
    assert_thread_access(thread_id, username)
    with db_cursor() as cursor:
        cursor.execute("UPDATE thread_metadata SET deleted_at = NULL WHERE thread_id = ?", (thread_id,))
    return {"status": "ok"}


@app.get("/history/{thread_id}")
def history(thread_id: str, session_id: Optional[str] = Cookie(None)):
    username = get_current_user(session_id)
    assert_thread_access(thread_id, username)

    config = {"configurable": {"thread_id": thread_id}}
    state = chatbot.get_state(config)
    if not state or not state.values.get("messages"):
        return {"thread_id": thread_id, "messages": []}

    out = []
    for m in state.values["messages"]:
        if isinstance(m, HumanMessage):
            role = "user"
        elif isinstance(m, AIMessage):
            role = "assistant"
        elif isinstance(m, ToolMessage):
            role = "tool"
        else:
            role = "system"

        content_text = ""
        if isinstance(m.content, list):
            for part in m.content:
                if isinstance(part, dict) and part.get("type") == "text":
                    content_text = part.get("text", "")
                    break
        else:
            content_text = m.content

        out.append({"role": role, "content": content_text})
    return {"thread_id": thread_id, "messages": out}


@app.get("/threads/{thread_id}/export")
def export_thread(
    thread_id: str,
    fmt: str = Query("md", pattern="^(md|txt|pdf)$"),
    session_id: Optional[str] = Cookie(None)
):
    username = get_current_user(session_id)
    assert_thread_access(thread_id, username)

    config = {"configurable": {"thread_id": thread_id}}
    state = chatbot.get_state(config)
    messages = state.values.get("messages", []) if state else []

    lines = [f"Thread Export: {thread_id}", f"Exported: {datetime.datetime.utcnow().isoformat()}Z", ""]
    for m in messages:
        role = "You" if isinstance(m, HumanMessage) else "Atlas" if isinstance(m, AIMessage) else "Tool"
        content_text = m.content if isinstance(m.content, str) else str(m.content)
        lines.append(f"{role}: {content_text}")
        lines.append("")

    if fmt == "pdf":
        pdf_bytes = generate_simple_pdf(thread_id, lines)
        filename = f"{thread_id}.pdf"
        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Cache-Control": "no-store, no-cache, must-revalidate"
            },
        )

    body = "\n".join(lines)
    media_type = "text/markdown" if fmt == "md" else "text/plain"
    filename = f"{thread_id}.{fmt}"
    return StreamingResponse(
        io.BytesIO(body.encode("utf-8")),
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/export/pptx/{thread_id}")
def export_pptx(thread_id: str, request: Request, session_id: Optional[str] = Cookie(None)):
    """Builds a real downloadable .pptx from the most recent <presentation>
    markup found in the thread's assistant messages."""
    username = get_current_user(session_id)
    assert_thread_access(thread_id, username)

    config = {"configurable": {"thread_id": thread_id}}
    state = chatbot.get_state(config)
    messages = state.values.get("messages", []) if state else []

    parsed = None
    for m in reversed(messages):
        if isinstance(m, AIMessage) and isinstance(m.content, str):
            parsed = parse_presentation_markup(m.content)
            if parsed and parsed["slides"]:
                break

    if not parsed or not parsed["slides"]:
        raise HTTPException(status_code=404, detail="No presentation found in this thread yet. Ask Atlas to generate one first.")

    try:
        pptx_path = build_pptx(parsed["title"], parsed["slides"])
    except ImportError:
        raise HTTPException(status_code=500, detail="python-pptx is not installed on the server. Run: pip install python-pptx")
    except Exception as e:
        logger.error(f"pptx export failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to build presentation file.")

    filename = f"{parsed['title'][:40].strip().replace(' ', '_') or 'presentation'}.pptx"
    write_audit(username, "export_pptx", detail=thread_id, ip=client_ip(request))
    return FileResponse(
        pptx_path,
        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        filename=filename,
    )


@app.get("/export/poster-pdf/{thread_id}")
def export_poster_pdf(thread_id: str, request: Request, session_id: Optional[str] = Cookie(None)):
    """Builds a real downloadable poster .pdf from the most recent <poster>
    markup found in the thread's assistant messages."""
    username = get_current_user(session_id)
    assert_thread_access(thread_id, username)

    config = {"configurable": {"thread_id": thread_id}}
    state = chatbot.get_state(config)
    messages = state.values.get("messages", []) if state else []

    parsed = None
    for m in reversed(messages):
        if isinstance(m, AIMessage) and isinstance(m.content, str):
            parsed = parse_poster_markup(m.content)
            if parsed and parsed["sections"]:
                break

    if not parsed or not parsed["sections"]:
        raise HTTPException(status_code=404, detail="No research poster found in this thread yet. Ask Atlas to generate one first.")

    try:
        pdf_path = build_poster_pdf(parsed["title"], parsed["authors"], parsed["domain"], parsed["sections"])
    except Exception as e:
        logger.error(f"poster pdf export failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to build poster PDF.")

    filename = f"{parsed['title'][:40].strip().replace(' ', '_') or 'poster'}.pdf"
    write_audit(username, "export_poster", detail=thread_id, ip=client_ip(request))
    return FileResponse(pdf_path, media_type="application/pdf", filename=filename)


@app.get("/download/asset")
def download_asset(url: str = Query(..., description="Local /static/generated/... asset path"),
                    filename: Optional[str] = Query(None),
                    session_id: Optional[str] = Cookie(None)):
    """Forces a proper Content-Disposition download for any already-generated
    local asset (image/video), instead of the browser just navigating to it."""
    get_current_user(session_id)  # any authenticated user may download

    if not url.startswith("/static/generated/") and not url.startswith("/static/exports/"):
        raise HTTPException(status_code=400, detail="Invalid asset path.")

    local_path = url.lstrip("/")
    if not os.path.exists(local_path):
        raise HTTPException(status_code=404, detail="Asset not found or has expired.")

    return FileResponse(local_path, filename=filename or os.path.basename(local_path))


@app.post("/reset/{thread_id}")
def reset(thread_id: str, session_id: Optional[str] = Cookie(None)):
    username = get_current_user(session_id)
    assert_thread_access(thread_id, username)

    config = {"configurable": {"thread_id": thread_id}}
    chatbot.update_state(config, {"messages": [], "summary": None, "needs_escalation": False})
    write_audit(username, "reset_thread", detail=thread_id)
    return {"thread_id": thread_id, "status": "reset"}


# --------------------------------------------------------------------------- #
# UI & health
# --------------------------------------------------------------------------- #

@app.get("/", response_class=HTMLResponse)
def home():
    html_path = os.path.join("static", "index.html")
    if not os.path.exists(html_path):
        raise HTTPException(status_code=404, detail="static/index.html not found")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/health")
def health_get():
    checks = {"database": "ok"}
    try:
        with db_cursor() as cursor:
            cursor.execute("SELECT 1")
    except Exception as e:
        checks["database"] = f"error: {e}"
    overall = "ok" if all(v == "ok" for v in checks.values()) else "degraded"
    return {"status": overall, "checks": checks, "version": app.version}


@app.head("/health")
def health_head():
    return Response()


@app.get("/readiness")
def readiness():
    return {"status": "ready"}


# --------------------------------------------------------------------------- #
# Startup & Static Mounting
# --------------------------------------------------------------------------- #

@app.on_event("startup")
def startup_validation():
    # Central validation at startup
    if not settings.GROQ_API_KEY and not settings.groq_api_keys:
        logger.error("Startup failure: GROQ_API_KEY must be configured in settings.")
        raise ValueError("GROQ_API_KEY must be configured in settings.")
    if not settings.SQLITE_DB_PATH:
        logger.error("Startup failure: SQLITE_DB_PATH must not be empty.")
        raise ValueError("SQLITE_DB_PATH must be configured.")
    init_metadata_db()

    # Log LangSmith Tracing Status
    if settings.LANGCHAIN_TRACING_V2.lower() == "true":
        if settings.LANGCHAIN_API_KEY:
            logger.info(f"LangSmith tracing enabled (Project: {settings.LANGCHAIN_PROJECT})")
        else:
            logger.warning("LANGCHAIN_TRACING_V2 is set to 'true', but LANGCHAIN_API_KEY is not configured.")
    else:
        logger.info("LangSmith tracing is disabled.")


app.mount("/static", StaticFiles(directory="static"), name="static")
