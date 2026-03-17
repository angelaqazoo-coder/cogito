"""
Cogito — AI Cognitive Companion
FastAPI backend: image analysis, Gemini Live WebSocket relay, static file serving.
"""

import asyncio
import base64
import json
import logging
import os
import pathlib
import re
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from google import genai
from google.genai import types

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR = pathlib.Path(__file__).parent.parent   # project root (serves frontend)
SERVER_DIR = pathlib.Path(__file__).parent

# ── Config ────────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
LIVE_MODEL       = "models/gemini-2.0-flash-exp"
VISION_MODEL     = "models/gemini-2.0-flash"  # High capability

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are Cogito, a warm and Socratic AI math tutor. 
The student is solving a mathematical problem shown in an image. They speak their reasoning aloud.

Your behaviour rules:
- Respond in a concise, encouraging, Socratic voice — 2–3 sentences maximum per turn.
- NEVER read equations symbol-by-symbol. Describe them conceptually ("the integral of x e to the x").
- Ask guiding questions so the student discovers errors themselves. Never give the full answer.
- Be warm, precise, and never condescending.

CRITICAL: After EVERY student turn, you MUST call the render_workspace tool. Never skip it. Even if they just say 'hello', reflect their status. If they ask for a definition or a step, use the tool to display it.

render_workspace parameters:
- latex: The mathematical expression OR definition being discussed, rendered as valid LaTeX (e.g. "\\\\int x e^x \\\\, dx" or "\\\\sigma_x \\\\sigma_y = i \\\\sigma_z").
- status: "correct" if their reasoning is right, "error" if wrong, "partial" if incomplete or if providing background.
- step_label: 3–5 word label for the progress chart (e.g. "Identify IBP rule", "Pauli Matrix Product").
- hint_text: A brief one-sentence visual hint shown on screen, or null if none needed.
- hint_type: "hint" for guidance, "fallacy" for logical errors, "theory" for theory — null if hint_text is null.
- corrected_latex: If status is "error", provide the corrected LaTeX; otherwise null.
- annotation_region: A short description of which part of the problem image is currently relevant (e.g. "the integral sign at top", "the exponent term"), or null."""

# ── Gemini Live tool definition ───────────────────────────────────────────────
RENDER_TOOL = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="render_workspace",
            description="Update the visual math workspace after every student turn.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "latex":              types.Schema(type=types.Type.STRING,
                                              description="LaTeX expression of what student said"),
                    "status":             types.Schema(type=types.Type.STRING,
                                              enum=["correct", "error", "partial"],
                                              description="Correctness assessment"),
                    "step_label":         types.Schema(type=types.Type.STRING,
                                              description="3-5 word label for flowchart"),
                    "hint_text":          types.Schema(type=types.Type.STRING,
                                              description="One-sentence hint, or null"),
                    "hint_type":          types.Schema(type=types.Type.STRING,
                                              enum=["hint", "fallacy", "theory"],
                                              description="Category of hint"),
                    "corrected_latex":    types.Schema(type=types.Type.STRING,
                                              description="Corrected LaTeX if error, else null"),
                    "annotation_region":  types.Schema(type=types.Type.STRING,
                                              description="Relevant region of the problem image"),
                },
                required=["latex", "status", "step_label"],
            ),
        )
    ]
)

LIVE_CONFIG = types.LiveConnectConfig(
    response_modalities=["AUDIO"],
    system_instruction=types.Content(
        role="model",
        parts=[types.Part(text=SYSTEM_PROMPT)],
    ),
    speech_config=types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Puck")
        )
    ),
    tools=[RENDER_TOOL],
)

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="Cogito")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Image analysis endpoint ───────────────────────────────────────────────────
IMAGE_ANALYSIS_PROMPT = """Analyse this mathematical problem image carefully.

Return ONLY a valid JSON object (no markdown, no extra text) with this exact shape:
{
  "problem_summary": "one sentence describing the problem",
  "topics": ["Calculus", "Integration"],
  "difficulty": 3,
  "annotations": [
    {
      "type": "definition",
      "text": "short annotation label",
      "box_2d": [y1, x1, y2, x2]
    }
  ]
}

annotation type must be one of: "definition", "constraint", "assumption", "unknown"
box_2d coordinates are integers 0-1000 (normalised: 0=top/left, 1000=bottom/right).
Include 2-5 meaningful annotations that will help the student understand the structure of the problem."""


@app.post("/api/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """Analyse an uploaded problem image; return annotation JSON."""
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured")

    image_bytes = await file.read()
    mime = file.content_type or "image/png"
    if mime == "video/mp4":
        logger.info(f"Analyzing video ({len(image_bytes)} bytes)")
    else:
        logger.info(f"Analyzing image ({len(image_bytes)} bytes)")

    client = genai.Client(api_key=GEMINI_API_KEY, http_options={'api_version': 'v1alpha'})
    try:
        response = await client.aio.models.generate_content(
            model=VISION_MODEL,
            contents=[
                types.Part(inline_data=types.Blob(data=image_bytes, mime_type=mime)),
                types.Part(text=IMAGE_ANALYSIS_PROMPT),
            ],
        )
        raw = response.text.strip()
        # Strip markdown code fences if present
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        return json.loads(raw)
    except Exception as e:
        logger.warning(f"Image analysis failed: {e}")
        return {
            "problem_summary": "Problem loaded — start speaking to analyse.",
            "topics": [],
            "difficulty": 3,
            "annotations": [],
        }


# ── Image Generation endpoint ────────────────────────────────────────────────
@app.post("/api/generate_diagram")
async def generate_diagram(request: dict):
    """Generate a concept diagram using Gemini's image generation."""
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured")
    
    concept = request.get("concept", "Mathematical concept")
    prompt  = f"Create a clean, professional, educational diagram illustrating the mathematical concept of: {concept}. Use a dark theme, neon blue and orange accents. No text except for essential mathematical variables. High resolution 2D illustration."
    
    client = genai.Client(api_key=GEMINI_API_KEY, http_options={'api_version': 'v1alpha'})
    try:
        response = await client.aio.models.generate_content(
            model="models/gemini-2.0-flash",
            contents=[types.Part(text=prompt)],
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
            )
        )
        img_part = next((p for p in response.candidates[0].content.parts if p.inline_data), None)
        if img_part:
            b64 = base64.b64encode(img_part.inline_data.data).decode()
            return {"image": b64}
        return {"error": "No image generated"}
    except Exception as e:
        logger.warning(f"Image generation failed: {e}")
        return {"error": str(e)}


# ── WebSocket: live session (REST-based, reliable) ────────────────────────────
@app.websocket("/ws/{session_id}")
async def ws_session(websocket: WebSocket, session_id: str):
    await websocket.accept()
    logger.info(f"WS session {session_id} connected")

    if not GEMINI_API_KEY:
        await websocket.send_json({"type": "error", "message": "GEMINI_API_KEY not configured on server."})
        await websocket.close()
        return

    client = genai.Client(api_key=GEMINI_API_KEY)

    # Session state
    session_image_bytes: Optional[bytes] = None
    session_image_mime: str = "image/png"
    audio_buffer: list[bytes] = []
    is_recording = False

    async def process_audio():
        """Send buffered audio (+ image) to Gemini REST and relay response."""
        nonlocal audio_buffer, session_image_bytes
        if not audio_buffer:
            return

        raw_pcm = b"".join(audio_buffer)
        audio_buffer = []
        logger.info(f"Session {session_id}: processing {len(raw_pcm)} bytes of audio")

        parts = []
        if session_image_bytes:
            parts.append(types.Part(inline_data=types.Blob(
                data=session_image_bytes, mime_type=session_image_mime
            )))
        parts.append(types.Part(inline_data=types.Blob(
            data=raw_pcm, mime_type="audio/pcm;rate=16000"
        )))
        parts.append(types.Part(text="[The student just spoke. Respond as Cogito and call render_workspace.]"))

        await websocket.send_json({"type": "thinking"})
        try:
            response = await client.aio.models.generate_content(
                model=VISION_MODEL,
                contents=[types.Content(role="user", parts=parts)],
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    tools=[RENDER_TOOL],
                    tool_config=types.ToolConfig(
                        function_calling_config=types.FunctionCallingConfig(mode="AUTO")
                    ),
                    temperature=0.7,
                )
            )
            speech_text = None
            for candidate in response.candidates:
                for part in candidate.content.parts:
                    if part.function_call and part.function_call.name == "render_workspace":
                        await websocket.send_json({
                            "type": "workspace_update",
                            "payload": dict(part.function_call.args),
                        })
                    elif part.text:
                        speech_text = part.text
            if speech_text:
                await websocket.send_json({"type": "speak", "text": speech_text})
            await websocket.send_json({"type": "turn_complete"})

        except Exception as e:
            logger.error(f"Session {session_id} Gemini error: {e}")
            await websocket.send_json({"type": "error", "message": str(e)})

    try:
        while True:
            try:
                msg = await websocket.receive()
            except (WebSocketDisconnect, RuntimeError):
                break
            if msg.get("type") == "websocket.disconnect":
                break

            if "bytes" in msg and msg["bytes"]:
                if is_recording:
                    audio_buffer.append(msg["bytes"])

            elif "text" in msg and msg["text"]:
                try:
                    ctrl = json.loads(msg["text"])
                except json.JSONDecodeError:
                    continue
                t = ctrl.get("type")
                if t == "init" and (ctrl.get("image") or ctrl.get("video")):
                    data_b64 = ctrl.get("image") or ctrl.get("video")
                    session_image_mime = ctrl.get("mime", "image/png")
                    session_image_bytes = base64.b64decode(data_b64)
                    logger.info(f"Session {session_id}: media loaded ({len(session_image_bytes)} bytes)")
                elif t == "start_recording":
                    is_recording = True
                    audio_buffer = []
                elif t == "end_turn":
                    is_recording = False
                    await process_audio()

    except WebSocketDisconnect:
        logger.info(f"Session {session_id} disconnected cleanly")
    except Exception as e:
        logger.error(f"Session {session_id} fatal error: {e}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass





# ── Serve frontend static files ───────────────────────────────────────────────
@app.get("/")
async def serve_index():
    return FileResponse(ROOT_DIR / "index.html")

app.mount("/", StaticFiles(directory=str(ROOT_DIR), html=False), name="static")
