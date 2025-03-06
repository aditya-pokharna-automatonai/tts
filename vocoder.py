# // this is vocoder using glow for converting mel spectom to wave form or audio file 


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# import matplotlib.pyplot as plt
# import soundfile as sf

# class Invertible1x1Conv(nn.Module):
#     """Stable Invertible 1x1 Convolution with LU Decomposition"""
#     def __init__(self, channels):
#         super().__init__()
#         P = torch.eye(channels)
#         W, _ = torch.linalg.qr(torch.randn(channels, channels), mode='reduced')
#         self.W = nn.Parameter(W)
#         self.P = nn.Parameter(P, requires_grad=False)  # Fixed permutation

#     def forward(self, x):
#         return torch.matmul(self.W, x)

#     def reverse(self, x):
#         W_inv = torch.inverse(self.W)  # More stable than direct inverse
#         return torch.matmul(W_inv, x)


# class CouplingLayer(nn.Module):
#     """Affine Coupling Layer with Speaker & Accent Conditioning"""
#     def __init__(self, channels, cond_dim=128):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Conv1d(channels // 2 + cond_dim, channels // 2, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv1d(channels // 2, channels, kernel_size=3, padding=1)
#         )

#     def forward(self, x, cond):
#         x_a, x_b = x.chunk(2, dim=1)  # Split into two parts
#         cond = cond.unsqueeze(-1).expand(-1, -1, x_a.shape[2])  # Expand cond for time dimension
#         out = self.net(torch.cat([x_a, cond], dim=1))  # Conditioned transformation
#         log_s, t = out.chunk(2, dim=1)
#         s = torch.exp(log_s)
#         x_b = x_b * s + t
#         return torch.cat([x_a, x_b], dim=1)

#     def reverse(self, x, cond):
#         x_a, x_b = x.chunk(2, dim=1)
#         cond = cond.unsqueeze(-1).expand(-1, -1, x_a.shape[2])
#         out = self.net(torch.cat([x_a, cond], dim=1))
#         log_s, t = out.chunk(2, dim=1)
#         s = torch.exp(log_s)
#         x_b = (x_b - t) / s
#         return torch.cat([x_a, x_b], dim=1)


# class GlowVocoder(nn.Module):
#     """Glow-Based Vocoder with Conditioning"""
#     def __init__(self, channels=256, num_flows=6, cond_dim=128):
#         super().__init__()
#         self.flows = nn.ModuleList()
#         for _ in range(num_flows):
#             self.flows.append(Invertible1x1Conv(channels))
#             self.flows.append(CouplingLayer(channels, cond_dim))

#     def forward(self, x, cond):
#         for flow in self.flows:
#             if isinstance(flow, CouplingLayer):
#                 x = flow(x, cond)
#             else:
#                 x = flow(x)
#         return x

#     def reverse(self, x, cond):
#         for flow in reversed(self.flows):
#             if isinstance(flow, CouplingLayer):
#                 x = flow.reverse(x, cond)
#             else:
#                 x = flow.reverse(x)
#         return x


# def visualize_waveform(waveform):
#     """Visualizes the waveform using Matplotlib."""
#     waveform = waveform.detach().squeeze().cpu().numpy()
#     plt.figure(figsize=(10, 4))
#     plt.plot(waveform)
#     plt.title("Generated Waveform")
#     plt.xlabel("Time")
#     plt.ylabel("Amplitude")
#     plt.grid()
#     plt.show()


# def save_waveform(waveform, filename="generated.wav", sample_rate=22050):
#     """Save waveform as a .wav file."""
#     waveform = waveform / (waveform.max() + 1e-6)  # Normalize
#     waveform = waveform.squeeze().detach().cpu().numpy()
#     sf.write(filename, waveform, sample_rate)
#     print(f"Waveform saved as {filename}")


# # Example Usage
# mel_spectrogram = torch.randn(1, 256, 200)  # (batch_size, channels, time)
# condition_vector = torch.randn(1, 128)  # Speaker/accent embedding

# vocoder = GlowVocoder()
# waveform = vocoder.reverse(mel_spectrogram, condition_vector)  # Convert mel to waveform

# # Visualize and save
# visualize_waveform(waveform)
# save_waveform(waveform)


















# _________________________________________________________________________________________________________________________________________________________________________________________



import os
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import subprocess
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define Dataset
class VocoderDataset(Dataset):
    def __init__(self, csv_path, audio_dir, n_mels=80):
        import pandas as pd
        self.data = pd.read_csv(csv_path).fillna("")
        self.audio_dir = audio_dir
        self.n_mels = n_mels

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
        audio_path = os.path.join(self.audio_dir, row['filename'])
        
        if audio_path.endswith(".mp3"):
            audio_path = self.convert_mp3_to_wav(audio_path)

        waveform, sample_rate = torchaudio.load(audio_path)
        mel_transform = T.MelSpectrogram(sample_rate=sample_rate, n_mels=self.n_mels, n_fft=1024, hop_length=256)
        mel_spectrogram = mel_transform(waveform).squeeze(0).T
        
        return mel_spectrogram, waveform.squeeze(0)  # (time, n_mels), (time)

# Collate function
def collate_fn(batch):
    mel_spectrograms, waveforms = zip(*batch)
    
    # Pad mel spectrograms
    mel_spectrograms = pad_sequence(mel_spectrograms, batch_first=True, padding_value=0.0)
    
    # Pad waveforms
    waveforms = pad_sequence(waveforms, batch_first=True, padding_value=0.0)
    
    return mel_spectrograms.to(device), waveforms.to(device)

# Simple Glow-based Vocoder (Placeholder Model)
class GlowVocoder(nn.Module):
    def __init__(self, n_mels=80):
        super(GlowVocoder, self).__init__()
        self.fc = nn.Linear(n_mels, 1)  # Dummy layer, replace with actual Glow-based implementation
    
    def forward(self, mel_spectrogram):
        return self.fc(mel_spectrogram).squeeze(-1)  # (batch, time)

# Training function
def train_glow(model, dataloader, epochs=25, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for mel_spectrogram, waveform in dataloader:
            optimizer.zero_grad()
            output = model(mel_spectrogram)
            loss = criterion(output, waveform[:, :output.shape[1]])  # Trim target to match output
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}")
    
    torch.save(model.state_dict(), "glow_vocoder.pth")
    print("Model saved as glow_vocoder.pth")

# Load dataset
data_path = r"C:\Users\Aditya Pokharna\Documents\AAI_WORKPLACE\TEXT_TO_AUDIO\dataset1\train.csv"
audio_path = r"C:\Users\Aditya Pokharna\Documents\AAI_WORKPLACE\TEXT_TO_AUDIO\dataset1"
dataset = VocoderDataset(data_path, audio_path)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

# Initialize and train model
model = GlowVocoder().to(device)
train_glow(model, dataloader, epochs=25, lr=0.001)
