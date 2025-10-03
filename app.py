"""
app.py

Simple Flask app that sends a structured prompt to Salesforce CodeGen (mono)
via the Hugging Face Inference API to generate HTML code from a natural-language description.

Usage:
  - Set environment variable HUGGINGFACE_API_TOKEN to your Hugging Face API token.
  - Optionally change MODEL (default uses codegen-350M-mono).
  - Run: python app.py
  - Open http://127.0.0.1:5000 and try a prompt.

Notes:
  - The app asks the model to return only HTML (no explanation text).
  - For larger/production workloads, use async, streaming, retries, caching, rate-limits, etc.
"""

import os
import textwrap
from flask import Flask, render_template_string, request, redirect, url_for, flash
import requests

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "change_this_secret_in_prod")

# Config: change model if you want a different CodeGen mono model (e.g. codegen25-7b-mono)
MODEL = os.environ.get("CODEGEN_MODEL", "Salesforce/codegen-350M-mono")
HF_API_TOKEN = os.environ.get("HUGGINGFACE_API_TOKEN")

if not HF_API_TOKEN:
    # Keep app running but warn user when they try to generate
    print("Warning: HUGGINGFACE_API_TOKEN not set. API calls will fail until you set this env var.")

HF_API_URL = f"https://api-inference.huggingface.co/models/{MODEL}"
HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"} if HF_API_TOKEN else {}

# Built-in prompt template: force the model to output only HTML, valid structure, minimal CSS, no comments.
PROMPT_TEMPLATE = textwrap.dedent("""
    /* You are a code generation model. Produce only valid HTML code (single file). */
    /* DO NOT add any explanation, commentary, or non-code text. */
    /* Output must start with <!doctype html> or <html> and be ready to save as .html. */
    /* Keep CSS and JS minimal and inline unless the prompt explicitly requests external files. */
    /* Ensure the HTML is accessible (proper headings, alt attributes for images if present). */
    """

    "User request: {user_request}"

    "\n\nHTML_CODE:\n"
)

# Simple UI template
INDEX_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>HTML CodeGen — Salesforce CodeGen (mono)</title>
  <style>
    body { font-family: Inter, Roboto, system-ui, sans-serif; margin: 28px; max-width: 1100px; }
    textarea { width: 100%; height: 120px; font-family: monospace; }
    .output { white-space: pre-wrap; background:#0f1115; color:#e6edf3; padding:16px; border-radius:6px; }
    label { font-weight:600; display:block; margin-bottom:6px; }
    .meta { color:#666; font-size:0.9rem; margin-bottom:12px; }
    .btn { padding:10px 14px; border-radius:6px; border:0; cursor:pointer; background:#0b5cff; color:white; }
    .danger { background:#c93c3c; }
    .small { font-size:0.85rem; color:#666; }
  </style>
</head>
<body>
  <h1>HTML CodeGen (Salesforce CodeGen - mono)</h1>
  <p class="meta">Write a short description of the HTML page you want. The model will return ready-to-save HTML only.</p>

  <form method="post" action="{{ url_for('generate') }}">
    <label for="prompt">Describe the HTML page (e.g. "landing page for a bakery with hero, 3 features, contact form")</label>
    <textarea id="prompt" name="prompt" placeholder="Describe the page..." required>{{ example }}</textarea>
    <div style="margin-top:8px;">
      <button class="btn" type="submit">Generate HTML</button>
      <button class="btn danger" formaction="{{ url_for('clear') }}" formmethod="post" type="submit" style="margin-left:8px;">Clear</button>
    </div>
    <p class="small">Model: {{ model }} · If you see warnings about tokens or truncation, try a shorter prompt.</p>
  </form>

  {% if html_code %}
    <h2>Generated HTML</h2>
    <p class="small">Copy and save to <code>.html</code>. The output is provided verbatim—no extra commentary.</p>
    <div class="output">{{ html_code }}</div>
    <p style="margin-top:8px;"><a href="data:text/html;charset=utf-8,{{ html_code | urlencode }}" download="generated.html">Download HTML file</a></p>
  {% endif %}

  {% with messages = get_flashed_messages() %}
    {% if messages %}
      <ul style="color:#b33;">
        {% for m in messages %}
          <li>{{ m }}</li>
        {% endfor %}
      </ul>
    {% endif %}
  {% endwith %}
</body>
</html>
"""

def call_hf_inference(prompt: str, max_length: int = 800):
    """
    Call Hugging Face Inference API for text generation.
    Returns text on success, raises Exception on failure.
    """
    if not HF_API_TOKEN:
        raise RuntimeError("Hugging Face API token is not set. Please set HUGGINGFACE_API_TOKEN.")

    payload = {
        "inputs": prompt,
        # Some model endpoints accept parameters; the inference API supports a `parameters` block.
        "parameters": {
            "max_new_tokens": max_length,
            "temperature": 0.2,
            "top_p": 0.95,
            "repetition_penalty": 1.05,
            # "return_full_text": False sometimes matters; leaving default behavior
        },
        "options": {"use_cache": True}
    }

    resp = requests.post(HF_API_URL, headers=HEADERS, json=payload, timeout=60)
    if resp.status_code == 200:
        # Usually returns dict with "generated_text" or a list. We try to be flexible.
        try:
            data = resp.json()
        except ValueError:
            raise RuntimeError(f"Invalid JSON response from HF inference API: {resp.text}")

        # Different models / wrappers can return different formats.
        # Common: {"generated_text": "..."}
        if isinstance(data, dict) and "generated_text" in data:
            return data["generated_text"]
        # Or the HF Inference API sometimes returns a list of dicts with 'generated_text'
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict) and "generated_text" in data[0]:
            return data[0]["generated_text"]
        # Or model itself may just return a string
        if isinstance(data, str):
            return data

        # Last fallback: inspect first available string-like entry
        # Try to join token text if present
        for item in (data if isinstance(data, list) else [data]):
            if isinstance(item, dict):
                for v in item.values():
                    if isinstance(v, str) and v.strip():
                        return v
        raise RuntimeError(f"Unexpected response format from HF API: {data}")
    else:
        # bubble up error text for easier debugging
        raise RuntimeError(f"HF inference API error {resp.status_code}: {resp.text}")

@app.route("/", methods=["GET"])
def index():
    example = "Landing page for a small bakery: hero with headline and CTA, 3 feature cards (fresh bread, pastries, coffee), opening hours section, and a simple contact form."
    return render_template_string(INDEX_HTML, model=MODEL, html_code=None, example=example)

@app.route("/generate", methods=["POST"])
def generate():
    user_prompt = request.form.get("prompt", "").strip()
    if not user_prompt:
        flash("Please provide a description of the page you want.")
        return redirect(url_for("index"))

    # Compose the effective prompt for the model
    full_prompt = PROMPT_TEMPLATE.format(user_request=user_prompt)

    try:
        generated = call_hf_inference(full_prompt, max_length=900)
    except Exception as e:
        flash(f"Generation failed: {e}")
        return redirect(url_for("index"))

    # Post-processing: many models may echo prompt; try to extract HTML after "HTML_CODE:" or return whole text
    result = generated
    # If model echoed the prompt, find first occurrence of "<!doctype" or "<html"
    idx = None
    lowered = result.lower()
    candidates = ["<!doctype", "<html", "<!doctype html"]
    for c in candidates:
        pos = lowered.find(c)
        if pos != -1:
            idx = pos
            break
    if idx is not None:
        result = result[idx:]
    # Ensure simple safety: strip leading/trailing whitespace
    result = result.strip()

    return render_template_string(INDEX_HTML, model=MODEL, html_code=result, example=user_prompt)

@app.route("/clear", methods=["POST"])
def clear():
    return redirect(url_for("index"))

if __name__ == "__main__":
    # Development server; for production use gunicorn/uvicorn etc.
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
