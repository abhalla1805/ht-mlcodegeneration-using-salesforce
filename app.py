"""
Streamlit app for AI HTML code generation using Salesforce CodeGen (mono).
No OpenAI. Runs locally with Hugging Face Transformers + Torch.

Usage:
    pip install -r requirements.txt
    streamlit run app.py
"""

import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Use the smallest CodeGen model that can run in free-tier / local CPU
MODEL_NAME = "Salesforce/codegen-350M-mono"

@st.cache_resource
def load_model():
    # Load tokenizer + model on CPU
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.float32
    ).to("cpu")
    return tokenizer, model

tokenizer, model = load_model()

PROMPT_TEMPLATE = """You are a code generation model.
Generate only valid HTML code in a single file.
Do not include explanations or comments outside HTML.
The code must begin with <!doctype html> or <html>.
Keep CSS inline unless otherwise requested.

User request: {user_request}

HTML_CODE:
"""

st.set_page_config(page_title="AI HTML Generator", page_icon="💻")
st.title("💻 AI HTML Code Generator")
st.caption("Powered by Salesforce CodeGen (mono) — no OpenAI")

user_input = st.text_area(
    "Describe the HTML page you want:",
    "Landing page for a bakery with hero section, 3 feature highlights, and a contact form."
)

max_new_tokens = st.slider("Max new tokens", 100, 800, 300, step=50)

if st.button("Generate HTML"):
    if not user_input.strip():
        st.error("Please enter a description.")
    else:
        with st.spinner("Generating HTML..."):
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

            # Extract just the HTML part
            text = generated
            lowered = text.lower()
            start = lowered.find("<!doctype")
            if start == -1:
                start = lowered.find("<html")
            if start != -1:
                text = text[start:]
            text = text.strip()

            st.subheader("Generated HTML")
            st.code(text, language="html")

            st.download_button(
                "Download HTML file",
                text,
                file_name="generated.html",
                mime="text/html"
            )
