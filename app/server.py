"""SmartCycle web app: take a photo on your phone, see the result on any screen.

Run:  python -m app.server        then open the printed address.

On a laptop, the page shows a QR code. Scanning it opens the same scan session
on your phone, so photos taken on the phone show up on the laptop (and vice
versa). Phone and laptop must be on the same Wi-Fi while running locally.
Set PUBLIC_URL (e.g. https://smartcycle.example.com) when deployed online.
"""

import io
import os
import secrets
import socket
import threading
import time
from pathlib import Path

import qrcode
import qrcode.image.svg
from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response
from PIL import UnidentifiedImageError
from pydantic import BaseModel

from src.data import open_image
from src.labels import AXES
from src.model import load_predictor
from src.pipeline import Session
from src.rules import RULES_SOURCE

STATIC = Path(__file__).parent / "static"
SESSION_TTL = 2 * 60 * 60  # forget scan sessions after 2 hours
PHONE_SEEN_WINDOW = 6  # seconds since the phone's last poll to count it as connected

app = FastAPI(title="SmartCycle")
predictor = load_predictor()
model_lock = threading.Lock()


class WebSession:
    def __init__(self):
        self.lock = threading.Lock()
        self.reset()
        self.phone_seen = 0.0

    def reset(self):
        self.pipeline = Session(predictor)
        self.steps = []  # finished StepResults as dicts, with photo index
        self.pending = None  # need_input StepResult dict, if waiting on the user
        self.photos = []  # JPEG bytes, one per uploaded photo
        self.busy = False
        self.error = None
        self.version = getattr(self, "version", 0) + 1
        self.touched = time.time()

    def bump(self):
        self.version += 1
        self.touched = time.time()


sessions: dict[str, WebSession] = {}


def get_session(sid: str) -> WebSession:
    s = sessions.get(sid)
    if s is None:
        raise HTTPException(404, "This scan session expired. Reload the page on your laptop for a new QR code.")
    return s


@app.get("/")
@app.get("/s/{sid}")
def index(sid: str | None = None):
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})


@app.post("/api/sessions")
def create_session():
    now = time.time()
    for sid in [k for k, v in sessions.items() if now - v.touched > SESSION_TTL]:
        del sessions[sid]
    sid = secrets.token_urlsafe(6)
    sessions[sid] = WebSession()
    return {"id": sid}


@app.get("/api/sessions/{sid}")
def session_state(sid: str, device: str = "desktop"):
    s = get_session(sid)
    if device == "phone":
        s.phone_seen = time.time()
    return {
        "id": sid,
        "version": s.version,
        "busy": s.busy,
        "error": s.error,
        "done": s.pipeline.done,
        "steps": s.steps,
        "pending": s.pending,
        "phone_connected": time.time() - s.phone_seen < PHONE_SEEN_WINDOW,
        "model": {
            "loaded": predictor is not None,
            "trained_axes": predictor.trained_axes if predictor else [],
        },
        "classes": AXES,
        "rules_source": RULES_SOURCE,
    }


@app.get("/api/sessions/{sid}/qr.svg")
def session_qr(sid: str, request: Request):
    get_session(sid)
    url = f"{public_base(request)}/s/{sid}"
    img = qrcode.make(url, image_factory=qrcode.image.svg.SvgPathImage, box_size=10, border=2)
    buf = io.BytesIO()
    img.save(buf)
    return Response(buf.getvalue(), media_type="image/svg+xml", headers={"X-Phone-Url": url})


@app.post("/api/sessions/{sid}/photo")
def upload_photo(sid: str, file: UploadFile):
    s = get_session(sid)
    if s.pipeline.done:
        raise HTTPException(409, "This item is finished. Start a new item first.")
    try:
        image = open_image(file.file)
    except UnidentifiedImageError:
        raise HTTPException(400, "That file isn't an image we can read. Try a JPG or PNG.")
    image.thumbnail((1024, 1024))
    with s.lock:
        s.busy, s.error = True, None
        s.bump()
        try:
            with model_lock:
                result = s.pipeline.scan(image)
        except Exception as e:  # keep the session usable and show the error on both screens
            s.busy, s.error = False, f"Analysis failed: {e}"
            s.bump()
            raise HTTPException(500, s.error)
        jpeg = io.BytesIO()
        image.save(jpeg, "JPEG", quality=85)
        s.photos.append(jpeg.getvalue())
        _record(s, result.to_dict())
    return session_state(sid)


class Answers(BaseModel):
    answers: dict[str, str]


@app.post("/api/sessions/{sid}/answer")
def answer(sid: str, body: Answers):
    s = get_session(sid)
    if s.pending is None:
        raise HTTPException(409, "No question is waiting for an answer.")
    with s.lock:
        try:
            result = s.pipeline.answer({**s.pending["labels"], **body.answers})
        except ValueError as e:
            raise HTTPException(400, str(e))
        _record(s, result.to_dict())
    return session_state(sid)


@app.post("/api/sessions/{sid}/reset")
def reset(sid: str):
    s = get_session(sid)
    with s.lock:
        s.reset()
    return session_state(sid)


@app.get("/api/sessions/{sid}/photos/{index}")
def photo(sid: str, index: int):
    s = get_session(sid)
    if not 0 <= index < len(s.photos):
        raise HTTPException(404)
    # Photo URLs restart at 0 for each new item, so the browser must not cache them.
    return Response(s.photos[index], media_type="image/jpeg", headers={"Cache-Control": "no-store"})


def _record(s: WebSession, result: dict):
    result["photo"] = len(s.photos) - 1
    if result["status"] == "need_input":
        s.pending = result
    else:
        s.pending = None
        s.steps.append(result)
    s.busy = False
    s.bump()


def public_base(request: Request) -> str:
    """The address a phone should use to reach this server."""
    if os.environ.get("PUBLIC_URL"):
        return os.environ["PUBLIC_URL"].rstrip("/")
    host, port = request.url.hostname, request.url.port
    if host in ("localhost", "127.0.0.1", "::1"):
        host = lan_ip()  # a phone can't reach "localhost" on the laptop
    return f"{request.url.scheme}://{host}" + (f":{port}" if port else "")


def lan_ip() -> str:
    # Opening a UDP socket sends nothing; it just asks the OS which interface it would use.
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        try:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
        except OSError:
            return "127.0.0.1"


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    print(f"\n  SmartCycle is running. Open http://localhost:{port} on this computer.")
    print(f"  Phones on the same Wi-Fi can use http://{lan_ip()}:{port}\n")
    if predictor is None:
        print("  (No trained model yet: the app will ask for every label. Train with `python -m src.model`.)\n")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
