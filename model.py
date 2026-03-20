import torch
import torch.nn as nn
import torch.optim as optim
import random

# ==========================================
# ATML Lab 6: Encoder-Decoder Architecture
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
        # source shape: (batch_size, src_len)
        # target shape: (batch_size, trg_len)
        
        batch_size = source.shape[0]
        trg_len = target.shape[1]
        trg_vocab_size = self.decoder.output_size
        
        # Tensor to store decoder outputs
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        
        # Last hidden & cell states of the encoder are used as initial states for decoder
        hidden, cell = self.encoder(source)
        
        # First input to the decoder is the <SOS> token (Assuming index 1 is <SOS>)
        x = target[:, 0]
        
        for t in range(1, trg_len):
            # Pass previous word, hidden and cell state to decoder
            output, hidden, cell = self.decoder(x, hidden, cell)
            
            outputs[:, t, :] = output
            
            # Get best word prediction
            best_guess = output.argmax(1)
            
            # Teacher forcing: Use actual next word or predicted next word
            x = target[:, t] if random.random() < teacher_forcing_ratio else best_guess
            
        return outputs

# ==========================================
# Example Usage & Setup for ATML Lab 6
# ==========================================
if __name__ == "__main__":
    print("Initializing ATML Lab 6 Encoder-Decoder Model...")
    
    # Hyperparameters (Example)
    SRC_VOCAB_SIZE = 5000  # English Vocabulary
    TRG_VOCAB_SIZE = 6000  # Hindi Vocabulary
    HIDDEN_SIZE = 256
    NUM_LAYERS = 2
    DROPOUT = 0.5
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize Networks
    enc = EncoderLSTM(SRC_VOCAB_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(device)
    dec = DecoderLSTM(TRG_VOCAB_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(device)
    model = Seq2Seq(enc, dec, device).to(device)

    print(model)
    
    # Dummy pass to verify dimensions
    dummy_src = torch.randint(0, SRC_VOCAB_SIZE, (32, 20)).to(device) # Batch 32, Seq length 20
    dummy_trg = torch.randint(0, TRG_VOCAB_SIZE, (32, 20)).to(device)
    
    out = model(dummy_src, dummy_trg)
    print(f"Output shape for dummy pass (batch, seq_len, vocab): {out.shape}")
    print("\nModel code generated successfully. Ready to be integrated with HuggingFace opus_books dataset!")
