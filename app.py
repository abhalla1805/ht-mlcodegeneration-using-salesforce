"""
Streamlit app for HTML code generation using Salesforce CodeGen (mono).
Runs locally with transformers + torch.

Usage:
    streamlit run app.py
"""

import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Pick a smaller model unless you own a data center.
MODEL_NAME = "Salesforce/codegen-350M-mono"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map="auto"
    )
    return tokenizer, model

tokenizer, model = load_model()

# Built-in template to force HTML output
PROMPT_TEMPLATE = """/* You are a code generation model. Produce only valid HTML code (single file). */
/* Do not add explanations or comments outside HTML. */
/* Begin output with <!doctype html> or <html>. */

User request: {user_request}

HTML_CODE:
"""

st.set_page_config(page_title="AI HTML Generator", page_icon="💻", layout="centered")

st.title("AI HTML Code Generator")
st.caption("Powered by Salesforce CodeGen (mono)")

user_input = st.text_area(
    "Describe the HTML page you want",
    "Landing page for a coffee shop with hero section, 3 product highlights, and contact form."
)

max_new_tokens = st.slider("Max new tokens", 100, 1000, 400, step=50)

if st.button("Generate HTML"):
    if not user_input.strip():
        st.error("Please provide a description.")
    else:
        prompt = PROMPT_TEMPLATE.format(user_request=user_input.strip())
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.2,
                top_p=0.95,
                repetition_penalty=1.05,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Try to extract HTML portion only
        text = generated
        lower = text.lower()
        start = lower.find("<!doctype")
        if start == -1:
            start = lower.find("<html")
        if start != -1:
            text = text[start:]
        text = text.strip()

        st.subheader("Generated HTML")
        st.code(text, language="html")

        st.download_button("Download HTML file", text, file_name="generated.html", mime="text/html")
