# this is for tactron 2 model to train a model from text to mel spectrom 

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import pandas as pd
import numpy as np
import subprocess
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from torch import optim

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
n_mels = 80  # Mel-spectrogram bins
batch_size = 32
learning_rate = 0.001
epochs = 10
teacher_forcing_ratio = 0.5
speaker_embedding_dim = 64  # Speaker embedding size
text_embedding_dim = 512  # Character embedding dimension

# Define character-level vocabulary
VOCAB = "abcdefghijklmnopqrstuvwxyz'.,!? -"
char_to_idx = {ch: idx for idx, ch in enumerate(VOCAB, start=1)}  # 1-based index
char_to_idx["<pad>"] = 0  # Padding token

# Function to convert text to sequence
def text_to_sequence(text):
    return [char_to_idx.get(ch, 0) for ch in text.lower()]  # Default to 0 for unknown chars

# Dataset class
class TextToSpeechDataset(Dataset):
    def __init__(self, csv_path, audio_dir):
        self.data = pd.read_csv(csv_path).fillna("")
        self.audio_dir = audio_dir
        
        # Extract unique speaker representations
        self.speakers = {}
        speaker_id = 0
        for _, row in self.data.iterrows():
            speaker_key = (row['gender'], row['age'], row['accent'])
            if speaker_key not in self.speakers:
                self.speakers[speaker_key] = speaker_id
                speaker_id += 1

    def convert_mp3_to_wav(self, mp3_path):
        """ Convert MP3 to WAV and return the new path """
        wav_path = mp3_path.replace(".mp3", ".wav")
        if not os.path.exists(wav_path):
            subprocess.run(["ffmpeg", "-i", mp3_path, wav_path, "-y"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return wav_path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = row['text']
        text_sequence = torch.tensor(text_to_sequence(text), dtype=torch.long)  # Convert text to indices
        
        speaker_key = (row['gender'], row['age'], row['accent'])
        speaker_id = self.speakers[speaker_key]
        audio_path = os.path.join(self.audio_dir, row['filename'])

        if audio_path.endswith(".mp3"):
            audio_path = self.convert_mp3_to_wav(audio_path)

        waveform, sample_rate = torchaudio.load(audio_path)
        mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels, n_fft=1024, hop_length=256)
        mel_spectrogram = mel_transform(waveform).squeeze(0).T

        return text_sequence, mel_spectrogram, torch.tensor(speaker_id, dtype=torch.long)

# Collate function
def collate_fn(batch):
    text_sequences, mel_spectrograms, speaker_ids = zip(*batch)
    
    # Pad text sequences
    text_sequences = pad_sequence(text_sequences, batch_first=True, padding_value=0)
    
    # Pad mel spectrograms
    mel_spectrograms = pad_sequence(mel_spectrograms, batch_first=True, padding_value=0.0)
    
    speaker_ids = torch.tensor(speaker_ids, dtype=torch.long)
    return text_sequences, mel_spectrograms, speaker_ids

# Tacotron 2 Model with Speaker Embeddings
class Tacotron2(nn.Module):
    def __init__(self, num_speakers, vocab_size=len(VOCAB) + 1):  # +1 for padding index
        super(Tacotron2, self).__init__()
        
        # Text embedding layer
        self.text_embedding = nn.Embedding(vocab_size, text_embedding_dim, padding_idx=0)
        
        # Speaker embedding layer
        self.speaker_embedding = nn.Embedding(num_speakers, speaker_embedding_dim)
        
        # Encoder: Takes text embeddings + speaker embeddings
        self.encoder = nn.LSTM(text_embedding_dim + speaker_embedding_dim, 256, batch_first=True, bidirectional=True)
        
        # Decoder
        self.decoder = nn.LSTM(n_mels + 512, 256, batch_first=True)
        self.fc = nn.Linear(256, n_mels)

    def forward(self, text_seq, mel_spectrogram, speaker_id):
        # Convert text to embeddings
        text_embed = self.text_embedding(text_seq)
        
        # Speaker embedding
        speaker_embed = self.speaker_embedding(speaker_id).unsqueeze(1).repeat(1, text_seq.shape[1], 1)
        
        # Combine text embeddings with speaker embeddings
        encoder_input = torch.cat([text_embed, speaker_embed], dim=-1)
        encoder_outputs, _ = self.encoder(encoder_input)

        outputs = []
        decoder_input = torch.cat([mel_spectrogram[:, 0, :].unsqueeze(1), encoder_outputs[:, 0, :].unsqueeze(1)], dim=-1)
        max_len = min(mel_spectrogram.shape[1], encoder_outputs.shape[1])

        for t in range(1, max_len):
            output, _ = self.decoder(decoder_input, None)
            output = self.fc(output)
            outputs.append(output)

            next_input = mel_spectrogram[:, t, :].unsqueeze(1) if np.random.rand() < teacher_forcing_ratio else output
            decoder_input = torch.cat([next_input, encoder_outputs[:, t, :].unsqueeze(1)], dim=-1)

        return torch.cat(outputs, dim=1)

# Training function
def train_tacotron2(model, dataloader, epochs=10, lr=0.001, save_path="tacotron2_multispeaker.pth"):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for text_seq, mel_spectrogram, speaker_id in tqdm(dataloader):
            text_seq, mel_spectrogram, speaker_id = text_seq.to(device), mel_spectrogram.to(device), speaker_id.to(device)
            optimizer.zero_grad()
            output = model(text_seq, mel_spectrogram, speaker_id)
            loss = criterion(output, mel_spectrogram[:, :output.shape[1], :])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

# Load dataset
data_path = r"C:\Users\Aditya Pokharna\Documents\AAI_WORKPLACE\TEXT_TO_AUDIO\dataset1\train.csv"
audio_path = r"C:\Users\Aditya Pokharna\Documents\AAI_WORKPLACE\TEXT_TO_AUDIO\dataset1"
dataset = TextToSpeechDataset(data_path, audio_path)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# Initialize and train model
num_speakers = len(dataset.speakers)
model = Tacotron2(num_speakers).to(device)
train_tacotron2(model, dataloader, epochs=epochs, lr=learning_rate)
