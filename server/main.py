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


# ── WebSocket: live session ───────────────────────────────────────────────────
@app.websocket("/ws/{session_id}")
async def ws_session(websocket: WebSocket, session_id: str):
    await websocket.accept()
    logger.info(f"WS session {session_id} connected")

    if not GEMINI_API_KEY:
        await websocket.send_json({"type": "error", "message": "GEMINI_API_KEY not configured on server."})
        await websocket.close()
        return

    client = genai.Client(api_key=GEMINI_API_KEY, http_options={'api_version': 'v1alpha'})

    try:
        async with client.aio.live.connect(model=LIVE_MODEL, config=LIVE_CONFIG) as gemini:

            async def browser_to_gemini():
                """Read from browser WebSocket → relay to Gemini Live."""
                while True:
                    try:
                        msg = await websocket.receive()
                    except (WebSocketDisconnect, RuntimeError):
                        break

                    if msg.get("type") == "websocket.disconnect":
                        break

                    if "bytes" in msg and msg["bytes"]:
                        await gemini.send(
                            input=types.LiveClientRealtimeInput(
                                media_chunks=[
                                    types.Blob(
                                        data=msg["bytes"],
                                        mime_type="audio/pcm;rate=16000",
                                    )
                                ]
                            )
                        )

                    elif "text" in msg and msg["text"]:
                        try:
                            ctrl = json.loads(msg["text"])
                        except json.JSONDecodeError:
                            continue

                        if ctrl.get("type") == "init" and ctrl.get("image"):
                            img_bytes = base64.b64decode(ctrl["image"])
                            img_mime  = ctrl.get("mime", "image/png")
                            logger.info(f"Session {session_id}: loading image ({len(img_bytes)} bytes)")
                            await gemini.send(
                                input=[
                                    types.Part(inline_data=types.Blob(data=img_bytes, mime_type=img_mime)),
                                    types.Part(text=(
                                        "This is the math problem the student is working on. "
                                        "Acknowledge briefly that you can see it and are ready."
                                    )),
                                ],
                                end_of_turn=True,
                            )
                        elif ctrl.get("type") == "end_turn":
                            await gemini.send(end_of_turn=True)

            async def gemini_to_browser():
                """Read from Gemini Live → relay to browser WebSocket."""
                async for response in gemini.receive():
                    try:
                        if response.server_content and response.server_content.model_turn:
                            for part in response.server_content.model_turn.parts:
                                if part.inline_data and "audio" in part.inline_data.mime_type:
                                    audio_b64 = base64.b64encode(part.inline_data.data).decode()
                                    await websocket.send_json({
                                        "type": "audio",
                                        "data": audio_b64,
                                        "mime": part.inline_data.mime_type,
                                    })

                        if response.tool_call:
                            for fn in response.tool_call.function_calls:
                                if fn.name == "render_workspace":
                                    await websocket.send_json({
                                        "type": "workspace_update",
                                        "payload": dict(fn.args),
                                    })
                                    await gemini.send(
                                        input=types.LiveClientToolResponse(
                                            function_responses=[
                                                types.FunctionResponse(
                                                    name=fn.name,
                                                    id=fn.id,
                                                    response={"result": "Workspace updated."},
                                                )
                                            ]
                                        )
                                    )

                        if response.server_content and response.server_content.turn_complete:
                            await websocket.send_json({"type": "turn_complete"})

                        if response.server_content and response.server_content.interrupted:
                            await websocket.send_json({"type": "interrupted"})

                    except WebSocketDisconnect:
                        break
                    except Exception as e:
                        logger.error(f"Receive loop error: {e}")

            send_task    = asyncio.create_task(browser_to_gemini())
            receive_task = asyncio.create_task(gemini_to_browser())
            done, pending = await asyncio.wait(
                [send_task, receive_task],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()

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
