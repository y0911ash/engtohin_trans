import streamlit as st
from transformers import pipeline
import torch

# Page Config
st.set_page_config(page_title="NLP Sequence Hub", layout="wide", page_icon="🌍")

# Custom Styling
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: #e2e8f0;
    }
    h1 {
        background: -webkit-linear-gradient(45deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
</style>
""", unsafe_allow_html=True)

st.title("🌍 NLP Sequence Hub")
st.markdown("### English to Hindi Translation & Abstractive Summarization")
st.markdown("---")

tab1, tab2 = st.tabs(["🌎 English -> Hindi Translation", "📝 Neural Summarization"])

with tab1:
    st.subheader("English to Hindi Machine Translation")
    trans_input = st.text_area("Source Text (English)", height=150, placeholder="Enter English text to translate...")
    if st.button("Translate", key="trans_btn"):
        if trans_input:
            with st.spinner("🔄 Initializing Translation Pipeline..."):
                    # Using AutoModel and AutoTokenizer for maximum reliability
                    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
                    
                    model_name = "Helsinki-NLP/opus-mt-en-hi"
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                    
                    inputs = tokenizer(trans_input, return_tensors="pt", padding=True)
                    translated_tokens = model.generate(**inputs)
                    res_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
                    
                    st.success("##### Translation Output:")
                    st.info(res_text)
                except Exception as e:
                    st.error(f"Engine Error: {str(e)}")
        else:
            st.warning("Please enter some text to translate.")

with tab2:
    st.subheader("Neural Text Summarization")
    sum_input = st.text_area("Source Article", height=250, placeholder="Paste a long English article here...")
    if st.button("Summarize", key="sum_btn"):
        if sum_input:
            with st.spinner("🧠 Synthesizing Content..."):
                try:
                    summarizer = pipeline("summarization", model="t5-small")
                    summary = summarizer(sum_input, max_length=150, min_length=30, do_sample=False)
                    st.success("##### Summary Output:")
                    st.info(summary[0]['summary_text'])
                except Exception as e:
                    st.error(f"Synthesis Error: {str(e)}")
        else:
            st.warning("Please enter some text to summarize.")

st.markdown("---")
st.caption("Built with PyTorch & Transformers | ATML Lab 6 Solutions")
