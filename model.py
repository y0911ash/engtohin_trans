import torch
import torch.nn as nn
import torch.optim as optim
import random

# ==========================================
# ATML Lab 6 & A.1: Encoder-Decoder Models
# English to Hindi Machine Translation
# ==========================================

class EncoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.2):
        super(EncoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Embedding layer: converts word indexes to dense vectors
        self.embedding = nn.Embedding(input_size, hidden_size)
        
        # LSTM layer
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (batch_size, sequence_length)
        embedded = self.dropout(self.embedding(x))
        
        # outputs shape: (batch_size, seq_len, hidden_size)
        # hidden, cell shape: (num_layers, batch_size, hidden_size)
        outputs, (hidden, cell) = self.lstm(embedded)
        
        # Context vectors to be passed to Decoder
        return hidden, cell

    def get_all_outputs(self, x):
        embedded = self.dropout(self.embedding(x))
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, (hidden, cell)


class DecoderLSTM(nn.Module):
    def __init__(self, output_size, hidden_size, num_layers=1, dropout=0.2):
        super(DecoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden, cell):
        # x is a single word index (batch_size, 1)
        x = x.unsqueeze(1)
        
        embedded = self.dropout(self.embedding(x))
        
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        
        # Prediction
        prediction = self.fc(output.squeeze(1))
        
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        assert encoder.hidden_size == decoder.hidden_size, "Hidden sizes must match"
        assert encoder.num_layers == decoder.num_layers, "Num layers must match"

    def forward(self, source, target, teacher_forcing_ratio=0.5):
        batch_size = source.shape[0]
        trg_len = target.shape[1]
        trg_vocab_size = self.decoder.output_size
        
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        
        hidden, cell = self.encoder(source)
        x = target[:, 0]
        
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(x, hidden, cell)
            outputs[:, t, :] = output
            best_guess = output.argmax(1)
            x = target[:, t] if random.random() < teacher_forcing_ratio else best_guess
            
        return outputs

# ==========================================
# ATML Lab A.1: Attention Mechanism
# ==========================================

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden: (num_layers, batch_size, hidden_size)
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        
        # Use last layer hidden state
        hidden = hidden[-1].unsqueeze(1).repeat(1, src_len, 1)
        
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        
        return torch.softmax(attention, dim=1)


class DecoderAttention(nn.Module):
    def __init__(self, output_size, hidden_size, num_layers=1, dropout=0.2):
        super(DecoderAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.attention = Attention(hidden_size)
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size * 2, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden, cell, encoder_outputs):
        x = x.unsqueeze(1)
        embedded = self.dropout(self.embedding(x))
        
        a = self.attention(hidden, encoder_outputs)
        a = a.unsqueeze(1)
        
        weighted = torch.bmm(a, encoder_outputs)
        lstm_input = torch.cat((embedded, weighted), dim=2)
        
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        prediction = self.fc(output.squeeze(1))
        
        return prediction, hidden, cell, a.squeeze(1)


class Seq2SeqWithAttention(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2SeqWithAttention, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, source, target, teacher_forcing_ratio=0.5):
        batch_size = source.shape[0]
        trg_len = target.shape[1]
        trg_vocab_size = self.decoder.output_size
        
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        
        encoder_outputs, (hidden, cell) = self.encoder.get_all_outputs(source)
        x = target[:, 0]
        
        for t in range(1, trg_len):
            output, hidden, cell, _ = self.decoder(x, hidden, cell, encoder_outputs)
            outputs[:, t, :] = output
            best_guess = output.argmax(1)
            x = target[:, t] if random.random() < teacher_forcing_ratio else best_guess
            
        return outputs

if __name__ == "__main__":
    print("Initializing ATML Lab Models...")
    SRC_VOCAB_SIZE, TRG_VOCAB_SIZE, HIDDEN_SIZE, NUM_LAYERS = 5000, 6000, 256, 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Test Simple Model
    enc_simple = EncoderLSTM(SRC_VOCAB_SIZE, HIDDEN_SIZE, NUM_LAYERS).to(device)
    dec_simple = DecoderLSTM(TRG_VOCAB_SIZE, HIDDEN_SIZE, NUM_LAYERS).to(device)
    model_simple = Seq2Seq(enc_simple, dec_simple, device).to(device)
    print("Simple Model Initialized.")

    # Test Attention Model
    enc_attn = EncoderLSTM(SRC_VOCAB_SIZE, HIDDEN_SIZE, NUM_LAYERS).to(device)
    dec_attn = DecoderAttention(TRG_VOCAB_SIZE, HIDDEN_SIZE, NUM_LAYERS).to(device)
    model_attn = Seq2SeqWithAttention(enc_attn, dec_attn, device).to(device)
    print("Attention Model Initialized.")
    
    dummy_src = torch.randint(0, SRC_VOCAB_SIZE, (32, 20)).to(device)
    dummy_trg = torch.randint(0, TRG_VOCAB_SIZE, (32, 20)).to(device)
    
    out_s = model_simple(dummy_src, dummy_trg)
    out_a = model_attn(dummy_src, dummy_trg)
    print(f"Simple Output: {out_s.shape} | Attention Output: {out_a.shape}")
