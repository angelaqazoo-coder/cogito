# 🧠 Cogito — AI Cognitive Companion

> *A Socratic AI tutor that sees your problem, hears you think, and speaks back — no typing needed.*

Cogito is a multimodal AI math tutor that allows you to speak your reasoning aloud while it renders math in LaTeX, provides hints, and tracks your progress.

## Setup

```bash
cd cogito
python3 -m venv .venv
source .venv/bin/activate
pip install -r server/requirements.txt
uvicorn server.main:app --port 8000
```
