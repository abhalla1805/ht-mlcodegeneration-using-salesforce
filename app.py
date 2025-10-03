"""
Streamlit app: AI HTML code generator using Salesforce CodeGen (mono) via Hugging Face Inference API.

Steps to run locally:
  1. pip install -r requirements.txt
  2. streamlit run app.py
  3. Make sure you set your Hugging Face token (see below).

On Streamlit Cloud:
  - In "Settings > Secrets", add:
        HUGGINGFACE_API_TOKEN = "hf_your_api_key_here"
"""

import os
import streamlit as st
import requests
import textwrap

# ---------------- CONFIG ----------------
MODEL = "Salesforce/codegen-350M-mono"
HF_API_URL = f"https://api-inference.huggingface.co/models/{MODEL}"
HF_TOKEN = os.environ.get("HUGGINGFACE_API_TOKEN")
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

PROMPT_TEMPLATE = textwrap.dedent("""
    /* You are a code generation model. Produce only valid HTML code (single file). */
    /* Do not add explanations or comments outside HTML. */
    /* Begin output with <!doctype html> or <html>. */

    User request: {user_request}

    HTML_CODE:
""")

# ---------------- HELPERS ----------------
def call_hf(prompt: str, max_new_tokens: int = 400) -> str:
    if not HF_TOKEN:
        raise RuntimeError("Missing Hugging Face token. Set HUGGINGFACE_API_TOKEN in environment/secrets.")

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": 0.2,
            "top_p": 0.95,
            "repetition_penalty": 1.05,
        },
        "options": {"use_cache": True}
    }

    resp = requests.post(HF_API_URL, headers=HEADERS, json=payload, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"Hugging Face API error {resp.status_code}: {resp.text}")

    data = resp.json()
    if isinstance(data, list) and "generated_text" in data[0]:
        return data[0]["generated_text"]
    elif isinstance(data, dict) and "generated_text" in data:
        return data["generated_text"]
    elif isinstance(data, str):
        return data
    else:
        raise RuntimeError(f"Unexpected response: {data}")

def extract_html(text: str) -> str:
    lowered = text.lower()
    start = lowered.find("<!doctype")
    if start == -1:
        start = lowered.find("<html")
    if start != -1:
        text = text[start:]
    return text.strip()

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="AI HTML Generator", page_icon="💻")

st.title("💻 AI HTML Code Generator")
st.caption(f"Powered by Salesforce CodeGen (mono): {MODEL}")

user_prompt = st.text_area(
    "Describe the HTML page you want",
    "Landing page for a bakery: hero section with title and CTA, 3 feature cards, contact form."
)

max_new_tokens = st.slider("Max new tokens", 100, 800, 400, step=50)

if st.button("Generate HTML"):
    if not user_prompt.strip():
        st.error("Please type a description first.")
    else:
        with st.spinner("Generating HTML..."):
            try:
                prompt = PROMPT_TEMPLATE.format(user_request=user_prompt.strip())
                raw = call_hf(prompt, max_new_tokens=max_new_tokens)
                html_code = extract_html(raw)
                st.subheader("Generated HTML")
                st.code(html_code, language="html")
                st.download_button(
                    "Download HTML file",
                    html_code,
                    file_name="generated.html",
                    mime="text/html"
                )
            except Exception as e:
                st.error(f"Error: {e}")
