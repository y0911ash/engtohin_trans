import gradio as gr
import torch
import torch.nn as nn
from model import EncoderLSTM, DecoderLSTM, Seq2Seq, DecoderAttention, Seq2SeqWithAttention
import os
import re
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Environment Config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Portable Tokenizer (regex-based to avoid torchtext)
def simple_tokenizer(text):
    text = text.lower()
    text = re.sub(r"([.,!?\"':;])", r" \1 ", text)
    return text.split()

def load_portable_assets():
    try:
        en_data = torch.load('en_vocab_portable.pt', map_location='cpu')
        hi_data = torch.load('hi_vocab_portable.pt', map_location='cpu')
        
        en_stoi, hi_itos = en_data['stoi'], hi_data['itos']
        
        INPUT_DIM = len(en_stoi) + 100 # Buffer
        OUTPUT_DIM = len(hi_itos) + 100 
        HID_DIM = 256
        N_LAYERS = 1
        
        # Simple Model
        enc_s = EncoderLSTM(INPUT_DIM, HID_DIM, N_LAYERS).to(device)
        dec_s = DecoderLSTM(OUTPUT_DIM, HID_DIM, N_LAYERS).to(device)
        model_s = Seq2Seq(enc_s, dec_s, device).to(device)
        if os.path.exists('en-hi-simple-model.pt'):
            model_s.load_state_dict(torch.load('en-hi-simple-model.pt', map_location=device), strict=False)
            
        # Attention Model
        enc_a = EncoderLSTM(INPUT_DIM, HID_DIM, N_LAYERS).to(device)
        dec_a = DecoderAttention(OUTPUT_DIM, HID_DIM, N_LAYERS).to(device)
        model_a = Seq2SeqWithAttention(enc_a, dec_a, device).to(device)
        if os.path.exists('en-hi-attention-model.pt'):
            model_a.load_state_dict(torch.load('en-hi-attention-model.pt', map_location=device), strict=False)
            
        return model_s, model_a, en_data, hi_data
    except Exception as e:
        return None, None, str(e), None

def run_inference(sentence, model, en_data, hi_data, is_attention=True):
    model.eval()
    stoi = en_data['stoi']
    itos = hi_data['itos']
    
    tokens = simple_tokenizer(sentence)
    src_indexes = [stoi.get('<sos>', 2)] + [stoi.get(token, stoi.get('<unk>', 0)) for token in tokens] + [stoi.get('<eos>', 3)]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
    
    with torch.no_grad():
        if is_attention:
            encoder_outputs, (hidden, cell) = model.encoder.get_all_outputs(src_tensor)
        else:
            hidden, cell = model.encoder(src_tensor)
            
    trg_indexes = [stoi.get('<sos>', 2)]
    for i in range(50):
        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
        with torch.no_grad():
            if is_attention:
                output, hidden, cell, _ = model.decoder(trg_tensor, hidden, cell, encoder_outputs)
            else:
                output, hidden, cell = model.decoder(trg_tensor, hidden, cell)
        
        pred_token = output.argmax(1).item()
        trg_indexes.append(pred_token)
        if pred_token == stoi.get('<eos>', 3): break
        
    return " ".join([itos.get(i, '<unk>') for i in trg_indexes][1:-1])

# SOTA Fallback
sota_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-hi")
sota_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-hi")

def benchmark(text):
    ms, ma, en_d, hi_d = load_portable_assets()
    if isinstance(en_d, str): return f"Error: {en_d}", "", ""
    
    # SOTA
    inputs = sota_tokenizer(text, return_tensors="pt")
    sota_out = sota_tokenizer.decode(sota_model.generate(**inputs)[0], skip_special_tokens=True)
    
    # Custom Simple
    simple_out = run_inference(text, ms, en_d, hi_d, is_attention=False)
    
    # Custom Attention
    attn_out = run_inference(text, ma, en_d, hi_d, is_attention=True)
    
    return sota_out, simple_out, attn_out

# Gradio Interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🔬 Research MT Hub (Public Link)")
    gr.Markdown("Compare SOTA vs. Custom Lab A.1 Models side-by-side.")
    
    with gr.Row():
        input_text = gr.Textbox(label="English Input", value="the book is on the table")
    
    btn = gr.Button("Compare Models", variant="primary")
    
    with gr.Row():
        box1 = gr.Textbox(label="🚀 SOTA Result (MarianMT)")
        box2 = gr.Textbox(label="🟦 Simple Seq2Seq (Lab 6)")
        box3 = gr.Textbox(label="🟨 Bahdanau Attention (Lab A.1)")
        
    btn.click(benchmark, inputs=input_text, outputs=[box1, box2, box3])
    gr.Markdown("---")
    gr.Markdown("Built for ATML Lab A.1 | Enc-Dec with Attention Strategy")

if __name__ == "__main__":
    demo.launch(share=True)
