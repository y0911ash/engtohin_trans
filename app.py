import streamlit as st
import torch
import torch.nn as nn
from model import EncoderLSTM, DecoderLSTM, Seq2Seq, DecoderAttention, Seq2SeqWithAttention
import os
import re
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Page Config
st.set_page_config(page_title="Researcher MT Hub", layout="wide", page_icon="🔬")

# Robust Tokenizers (Portable-Style)
def simple_tokenizer(text):
    text = text.lower()
    text = re.sub(r"([.,!?\"':;])", r" \1 ", text)
    return text.split()

def hindi_tokenizer(text):
    return text.split()

# CSS Styling
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #020617 0%, #171717 100%); color: #f8fafc; }
    .comparison-box { background: rgba(255, 255, 255, 0.03); padding: 1.5rem; border-radius: 12px; border: 1px solid rgba(255,255,255,0.1); font-size: 1.1rem; }
</style>
""", unsafe_allow_html=True)

# Path Resolution
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_models_and_vocabs():
    try:
        # 1. Load Portable Vocabs (Dict Format)
        en_v_path = os.path.join(BASE_DIR, 'en_vocab_portable.pt')
        hi_v_path = os.path.join(BASE_DIR, 'hi_vocab_portable.pt')
        
        if not os.path.exists(en_v_path) or not os.path.exists(hi_v_path):
            return None, "Vocabulary files (.pt) not found. Please sync from GitHub."
            
        en_vocab = torch.load(en_v_path, map_location='cpu')
        hi_vocab = torch.load(hi_v_path, map_location='cpu')
        
        # 2. Setup Model Architecture
        INPUT_DIM = len(en_vocab['stoi']) + 100 # buffer
        OUTPUT_DIM = len(hi_vocab['itos']) + 100
        HID_DIM, N_LAYERS, device = 256, 1, torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Simple Model
        enc_s = EncoderLSTM(INPUT_DIM, HID_DIM, N_LAYERS).to(device)
        dec_s = DecoderLSTM(OUTPUT_DIM, HID_DIM, N_LAYERS).to(device)
        model_s = Seq2Seq(enc_s, dec_s, device).to(device)
        s_path = os.path.join(BASE_DIR, 'en-hi-simple-model.pt')
        if os.path.exists(s_path):
            model_s.load_state_dict(torch.load(s_path, map_location=device), strict=False)
        
        # Attention Model
        enc_a = EncoderLSTM(INPUT_DIM, HID_DIM, N_LAYERS).to(device)
        dec_a = DecoderAttention(OUTPUT_DIM, HID_DIM, N_LAYERS).to(device)
        model_a = Seq2SeqWithAttention(enc_a, dec_a, device).to(device)
        a_path = os.path.join(BASE_DIR, 'en-hi-attention-model.pt')
        if os.path.exists(a_path):
            model_a.load_state_dict(torch.load(a_path, map_location=device), strict=False)
            
        return {'s': model_s, 'a': model_a, 'env': en_vocab, 'hiv': hi_vocab}, None
    except Exception as e:
        return None, f"Asset Loading Error: {str(e)}"

# Main Sidebar
st.sidebar.title("🔬 Seq2Seq Control")
st.sidebar.info("Model: Lab A.1 Attention Strategy")
st.sidebar.write(f"CUDA: {torch.cuda.is_available()}")

# Tabs
tab1, tab2 = st.tabs(["🌎 SOTA Predictor", "🔬 Researcher Bench"])

with tab1:
    st.subheader("Helsinki-NLP / Opus-MT (Direct Mode)")
    txt = st.text_area("Source Text", key="sota_in", value="Where is the library?")
    if st.button("Generate Cloud MT"):
        with st.spinner("Processing..."):
            name = "Helsinki-NLP/opus-mt-en-hi"
            tokenizer = AutoTokenizer.from_pretrained(name)
            model = AutoModelForSeq2SeqLM.from_pretrained(name)
            inputs = tokenizer(txt, return_tensors="pt", padding=True)
            out = tokenizer.decode(model.generate(**inputs)[0], skip_special_tokens=True)
            st.success(out)

with tab2:
    st.subheader("Architecture Comparison (Simple vs Attention)")
    assets, err = load_models_and_vocabs()
    if err:
        st.warning(err)
    else:
        q = st.text_input("Enter benchmark phrase:", value="i am a student")
        if st.button("Compare Architecture"):
            # Inference Logic
            def infer(m, is_attn, text, data_e, data_h):
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                m.eval()
                tokens = simple_tokenizer(text)
                src = [data_e['stoi'].get('<sos>', 2)] + [data_e['stoi'].get(t, 0) for t in tokens] + [data_e['stoi'].get('<eos>', 3)]
                src = torch.LongTensor(src).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    if is_attn: outs, (h, c) = m.encoder.get_all_outputs(src)
                    else: h, c = m.encoder(src)
                
                trg = [data_e['stoi'].get('<sos>', 2)]
                for _ in range(50):
                    t_in = torch.LongTensor([trg[-1]]).to(device)
                    with torch.no_grad():
                        if is_attn: out, h, c, _ = m.decoder(t_in, h, c, outs)
                        else: out, h, c = m.decoder(t_in, h, c)
                    p = out.argmax(1).item()
                    trg.append(p)
                    if p == data_e['stoi'].get('<eos>', 3): break
                return " ".join([data_h['itos'].get(i, '<unk>') for i in trg][1:-1])

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### 🟦 Simple Seq2Seq")
                st.markdown(f'<div class="comparison-box">{infer(assets["s"], False, q, assets["env"], assets["hiv"])}</div>', unsafe_allow_html=True)
            with c2:
                st.markdown("#### 🟨 Attention Mechanism")
                st.markdown(f'<div class="comparison-box">{infer(assets["a"], True, q, assets["env"], assets["hiv"])}</div>', unsafe_allow_html=True)

st.markdown("---")
st.caption("Deployment Protocol v2.1 | Powered by Transformers 4.38.2 & Portable Vocab")
