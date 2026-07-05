"""
================================================================================
PSYCHE AI SERVICE — single-file version
================================================================================
Everything needed to run this service lives in this one file: the Flask app,
the prompt templates, the fallback logic, and setup/deploy instructions below.
The only other file you need afterward is aiService.js (frontend snippet,
included at the bottom of this docstring for copy-paste into psyche-v37).

--------------------------------------------------------------------------------
SETUP (Windows Command Prompt)
--------------------------------------------------------------------------------
pip install flask groq gunicorn
set GROQ_API_KEY=your_groq_key_here
python psyche_ai_service.py

Test it (in a second terminal, while the server is running):
curl -X POST http://localhost:5000/api/generate -H "Content-Type: application/json" -d "{\"feature\": \"narrative\", \"data\": {\"archetype\": \"The Architect\", \"dimensions\": {\"perception\": 72, \"response\": 55}, \"contradictions\": []}}"

Check http://localhost:5000/health in your browser — confirm
"groq_configured": true before deploying.

--------------------------------------------------------------------------------
DEPLOY TO RENDER
--------------------------------------------------------------------------------
1. Push this file + a requirements.txt (flask, groq, gunicorn) to a new repo.
2. Render -> New Web Service -> connect repo.
3. Build command:  pip install -r requirements.txt
4. Start command:  gunicorn psyche_ai_service:app
5. Env var: GROQ_API_KEY = your key
6. Deploy, then hit https://<your-service>.onrender.com/health to confirm.

Test the fallback path deliberately before wiring into v37: remove
GROQ_API_KEY on Render, redeploy, confirm /api/generate still returns 200
with "source": "fallback" instead of erroring. Put the key back after.

--------------------------------------------------------------------------------
FRONTEND SNIPPET — copy into psyche-v37 as aiService.js
--------------------------------------------------------------------------------
const AI_BASE = "https://psyche-ai.onrender.com";  // your deployed URL

export async function generateAI(feature, data) {
  try {
    const res = await fetch(`${AI_BASE}/api/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ feature, data }),
    });
    if (!res.ok) throw new Error(`AI service responded ${res.status}`);
    return await res.json();
  } catch (err) {
    console.error("generateAI failed:", err);
    return { text: null, source: "error" };
  }
}

// Usage in the report UI:
// const { text } = await generateAI("narrative", {
//   archetype: result.archetype,
//   dimensions: result.dimensions,
//   contradictions: result.contradictions,
// });
// render `text` regardless of source (ai or fallback)

--------------------------------------------------------------------------------
ROLLOUT ORDER — don't skip ahead
--------------------------------------------------------------------------------
1. Ship "narrative" only. Confirm real users get real output + fallback works.
2. Add "cbt_reframe" and "pattern_summary" — same file, new dict entries in
   PROMPTS below, no new deploy architecture.
3. Add Supabase caching only once you see actual repeat-call volume in Render
   logs. cache_key() below already gives you the hash to key on.
================================================================================
"""

from flask import Flask, request, jsonify
from groq import Groq
import os
import hashlib
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("psyche-ai")

app = Flask(__name__)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    logger.warning("GROQ_API_KEY not set — /api/generate will always fall back.")

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

MODEL = "llama-3.3-70b-versatile"

# ---------------------------------------------------------------------------
# Prompt templates — one entry per feature. Add new features here only,
# once step 1 in ROLLOUT ORDER above is confirmed live.
# ---------------------------------------------------------------------------

PROMPTS = {
    "narrative": lambda d: f"""You are writing a personality report narrative for a PRISM assessment.
Archetype: {d.get('archetype', 'Unknown')}
Dimension scores: {json.dumps(d.get('dimensions', {}))}
Contradictions detected: {json.dumps(d.get('contradictions', []))}

Write a 250-300 word narrative in second person ("you"). Editorial, literary tone —
precise, warm, no clinical or diagnostic language. Do not mention percentiles or
statistics directly; translate them into observations about behavior and tendency.
If contradictions exist, address them as nuance, not flaws.""",

    "cbt_reframe": lambda d: f"""A user completed a CBT thought record.
Situation: {d.get('situation', '')}
Automatic thought: {d.get('thought', '')}
Evidence for: {d.get('evidence_for', '')}
Evidence against: {d.get('evidence_against', '')}

Suggest one balanced, realistic reframe in 2-3 sentences. Warm, plain language,
CBT-consistent. Do not diagnose or use clinical labels.""",

    "pattern_summary": lambda d: f"""Braindump entries over time (most recent last):
{json.dumps(d.get('entries', []))}

Summarize the recurring pattern in 3-4 sentences. Offer one gentle, specific
observation about a tendency — not a diagnosis, not generic advice.""",
}

# ---------------------------------------------------------------------------
# Fallback content — used if Groq errors, times out, or is rate-limited.
# ---------------------------------------------------------------------------

def fallback_text(feature, data):
    if feature == "narrative":
        archetype = data.get("archetype", "your")
        return (f"Your {archetype} profile reflects a distinct blend of traits. "
                f"A fuller narrative will appear here shortly — refresh in a moment.")
    if feature == "cbt_reframe":
        return ("Consider: is there another way to view this situation that fits "
                "the evidence you listed equally well?")
    if feature == "pattern_summary":
        return ("We've noticed some recurring themes in your recent entries. "
                "Check back shortly for a fuller pattern summary.")
    return "Content temporarily unavailable."


def cache_key(feature, data):
    """Deterministic key for external caching (e.g. Supabase private_blobs).
    Compute this same key on the frontend/Supabase side before calling this
    service, so repeat views skip the AI call entirely."""
    raw = json.dumps({"feature": feature, "data": data}, sort_keys=True)
    return "ai:" + feature + ":" + hashlib.md5(raw.encode()).hexdigest()


def call_ai(prompt, max_tokens=500, temperature=0.7):
    """Single point of contact with the AI provider. Swap Groq for another
    provider here without touching any route or prompt template."""
    if not groq_client:
        raise RuntimeError("No AI client configured")

    completion = groq_client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=8,
    )
    return completion.choices[0].message.content


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/api/generate", methods=["POST"])
def generate():
    body = request.get_json(silent=True) or {}
    feature = body.get("feature")
    data = body.get("data", {})

    if feature not in PROMPTS:
        return jsonify({"error": f"unknown feature '{feature}'",
                         "valid_features": list(PROMPTS.keys())}), 400

    try:
        prompt = PROMPTS[feature](data)
        text = call_ai(prompt)
        return jsonify({
            "text": text,
            "source": "ai",
            "feature": feature,
            "cache_key": cache_key(feature, data),
        })
    except Exception as e:
        logger.error(f"AI generation failed for feature={feature}: {e}")
        return jsonify({
            "text": fallback_text(feature, data),
            "source": "fallback",
            "feature": feature,
            "cache_key": cache_key(feature, data),
        }), 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "groq_configured": groq_client is not None,
        "features": list(PROMPTS.keys()),
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
