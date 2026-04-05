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
    .stAlert { background-color: rgba(255,255,255,0.05) !important; border: none !important; }
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
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        HID_DIM, N_LAYERS = 256, 1
        
        # ==========================================
        # Dynamic Size Detection (Self-Syncing)
        # ==========================================
        s_path = os.path.join(BASE_DIR, 'en-hi-simple-model.pt')
        a_path = os.path.join(BASE_DIR, 'en-hi-attention-model.pt')
        
        # Simple Model Dynamic Load
        sd_s = torch.load(s_path, map_location='cpu')
        input_dim_s = sd_s['encoder.embedding.weight'].shape[0]
        output_dim_s = sd_s['decoder.embedding.weight'].shape[0]
        
        enc_s = EncoderLSTM(input_dim_s, HID_DIM, N_LAYERS).to(device)
        dec_s = DecoderLSTM(output_dim_s, HID_DIM, N_LAYERS).to(device)
        model_s = Seq2Seq(enc_s, dec_s, device).to(device)
        model_s.load_state_dict(sd_s, strict=False)
        
        # Attention Model Dynamic Load
        sd_a = torch.load(a_path, map_location='cpu')
        input_dim_a = sd_a['encoder.embedding.weight'].shape[0]
        output_dim_a = sd_a['decoder.embedding.weight'].shape[0]
        
        enc_a = EncoderLSTM(input_dim_a, HID_DIM, N_LAYERS).to(device)
        dec_a = DecoderAttention(output_dim_a, HID_DIM, N_LAYERS).to(device)
        model_a = Seq2SeqWithAttention(enc_a, dec_a, device).to(device)
        model_a.load_state_dict(sd_a, strict=False)
            
        return {
            's': model_s, 'a': model_a, 
            'env': en_vocab, 'hiv': hi_vocab, 
            'ids': input_dim_s, 'ods': output_dim_s,
            'ida': input_dim_a, 'oda': output_dim_a
        }, None
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
            try:
                name = "Helsinki-NLP/opus-mt-en-hi"
                tokenizer = AutoTokenizer.from_pretrained(name)
                model = AutoModelForSeq2SeqLM.from_pretrained(name)
                inputs = tokenizer(txt, return_tensors="pt", padding=True)
                out = tokenizer.decode(model.generate(**inputs)[0], skip_special_tokens=True)
                st.success(out)
            except Exception as e:
                st.error(f"SOTA Hub Error: {str(e)}")

with tab2:
    st.subheader("Architecture Comparison (Simple vs Attention)")
    assets, err = load_models_and_vocabs()
    if err:
        st.error(err)
    else:
        q = st.text_input("Enter benchmark phrase:", value="the book is on the table")
        if st.button("Compare Architecture"):
            # Robust Inference Logic with Index Boundaries
            def infer(m, is_attn, text, data_e, data_h, max_idx):
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                m.eval()
                tokens = simple_tokenizer(text)
                
                # Boundary check: Ensure we don't use indices larger than the weights
                src = []
                src.append(data_e['stoi'].get('<sos>', 2))
                for t in tokens:
                    idx = data_e['stoi'].get(t, 0)
                    if idx < max_idx: src.append(idx)
                    else: src.append(0) # fallback to <unk> if above boundary
                src.append(data_e['stoi'].get('<eos>', 3))
                
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
                res_s = infer(assets["s"], False, q, assets["env"], assets["hiv"], assets['ids'])
                st.markdown(f'<div class="comparison-box">{res_s}</div>', unsafe_allow_html=True)
            with c2:
                st.markdown("#### 🟨 Attention Mechanism")
                res_a = infer(assets["a"], True, q, assets["env"], assets["hiv"], assets['ida'])
                st.markdown(f'<div class="comparison-box">{res_a}</div>', unsafe_allow_html=True)
            
            st.info(f"✨ Note: Auto-synced Model Vocab (EN: {assets['ida']} | HI: {assets['oda']})")

st.markdown("---")
st.caption("Deployment Protocol v2.5 | Multi-Architecture Auto-Sync Suite")
