import streamlit as st
import torch
import torch.nn as nn
from model import EncoderLSTM, DecoderLSTM, Seq2Seq, DecoderAttention, Seq2SeqWithAttention
import os
import sys
from torchtext.data.utils import get_tokenizer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Page Config
st.set_page_config(page_title="Researcher MT Hub", layout="wide", page_icon="🔬")

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
    .comparison-box {
        background: rgba(255, 255, 255, 0.05);
        padding: 20px;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        min-height: 150px;
    }
</style>
""", unsafe_allow_html=True)

# Path Resolution
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Helper Functions
en_tokenizer = get_tokenizer('basic_english')

@st.cache_resource
def load_assets():
    assets = {}
    try:
        en_vocab_path = os.path.join(BASE_DIR, 'en_vocab.pt')
        hi_vocab_path = os.path.join(BASE_DIR, 'hi_vocab.pt')
        
        if not os.path.exists(en_vocab_path) or not os.path.exists(hi_vocab_path):
            return None, "Vocabulary files (.pt) not found in workspace."
            
        assets['en_vocab'] = torch.load(en_vocab_path, map_location='cpu')
        assets['hi_vocab'] = torch.load(hi_vocab_path, map_location='cpu')
        
        INPUT_DIM = len(assets['en_vocab'])
        OUTPUT_DIM = len(assets['hi_vocab'])
        HID_DIM = 256
        N_LAYERS = 1
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize Simple
        enc1 = EncoderLSTM(INPUT_DIM, HID_DIM, N_LAYERS).to(device)
        dec1 = DecoderLSTM(OUTPUT_DIM, HID_DIM, N_LAYERS).to(device)
        model_s = Seq2Seq(enc1, dec1, device).to(device)
        
        simple_path = os.path.join(BASE_DIR, 'en-hi-simple-model.pt')
        if os.path.exists(simple_path):
            model_s.load_state_dict(torch.load(simple_path, map_location=device))
        assets['model_simple'] = model_s
        
        # Initialize Attention
        enc2 = EncoderLSTM(INPUT_DIM, HID_DIM, N_LAYERS).to(device)
        dec2 = DecoderAttention(OUTPUT_DIM, HID_DIM, N_LAYERS).to(device)
        model_a = Seq2SeqWithAttention(enc2, dec2, device).to(device)
        
        attn_path = os.path.join(BASE_DIR, 'en-hi-attention-model.pt')
        if os.path.exists(attn_path):
            model_a.load_state_dict(torch.load(attn_path, map_location=device))
        assets['model_attn'] = model_a
        
        return assets, None
    except Exception as e:
        return None, f"Asset Loading Error: {str(e)}"

def translate_logic(sentence, model, assets, is_attention=True, max_len=50):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    en_vocab = assets['en_vocab']
    hi_vocab = assets['hi_vocab']
    
    model.eval()
    tokens = en_tokenizer(sentence)
    src_indexes = [en_vocab['<sos>']] + [en_vocab[token] for token in tokens] + [en_vocab['<eos>']]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
    
    with torch.no_grad():
        if is_attention:
            encoder_outputs, (hidden, cell) = model.encoder.get_all_outputs(src_tensor)
        else:
            hidden, cell = model.encoder(src_tensor)
            
    trg_indexes = [hi_vocab['<sos>']]
    for i in range(max_len):
        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
        with torch.no_grad():
            if is_attention:
                output, hidden, cell, _ = model.decoder(trg_tensor, hidden, cell, encoder_outputs)
            else:
                output, hidden, cell = model.decoder(trg_tensor, hidden, cell)
                
        pred_token = output.argmax(1).item()
        trg_indexes.append(pred_token)
        if pred_token == hi_vocab['<eos>']: break
        
    itos = hi_vocab.get_itos()
    res = " ".join([itos[i] for i in trg_indexes][1:-1])
    return res if res else "(Empty Output)"

# Main UI
st.title("🌍 Researcher MT Hub")
st.markdown("### English to Hindi Benchmark Suite")
st.markdown("---")

tab1, tab2 = st.tabs(["🚀 SOTA Translator", "🔬 Researcher Comparison"])

with tab1:
    st.subheader("SOTA MarianMT (Direct Model Inference)")
    sota_input = st.text_area("English Source", key="sota_in", value="How are you?")
    if st.button("SOTA Translate"):
        with st.spinner("🔄 Neural Inference in progress..."):
            try:
                model_name = "Helsinki-NLP/opus-mt-en-hi"
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                inputs = tokenizer(sota_input, return_tensors="pt", padding=True)
                translated_tokens = model.generate(**inputs)
                res = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
                st.success(f"**SOTA Result:** {res}")
            except Exception as e:
                st.error(f"SOTA Hub Error: {str(e)}")

with tab2:
    st.subheader("Lab A.1: Custom Encoder-Decoder Benchmarks")
    assets, error = load_assets()
    
    if error:
        st.warning(f"⚠️ {error} Please run the notebook first.")
    else:
        user_input = st.text_input("Enter sentence to compare focus:", value="the book is on the table")
        if st.button("Run Lab Comparison"):
            col1, col2 = st.columns(2)
            with col1:
                st.info("#### 🟦 Simple Seq2Seq")
                out_s = translate_logic(user_input, assets['model_simple'], assets, is_attention=False)
                st.markdown(f'<div class="comparison-box">{out_s}</div>', unsafe_allow_html=True)
            with col2:
                st.info("#### 🟨 Bahdanau Attention")
                out_a = translate_logic(user_input, assets['model_attn'], assets, is_attention=True)
                st.markdown(f'<div class="comparison-box">{out_a}</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/clouds/100/000000/artificial-intelligence.png")
    st.markdown("### Model Config")
    st.write(f"Environment: {sys.version.split()[0]}")
    st.write(f"CUDA Available: {torch.cuda.is_available()}")
    st.success("Deployment: Researcher Suite v1.2")

st.markdown("---")
st.caption("ATML Lab Solutions | Seq2Seq Research Protocol")
