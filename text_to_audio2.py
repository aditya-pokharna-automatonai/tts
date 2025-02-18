# from TTS.api import TTS

# # Initialize the TTS object with the VITS model
# model_name = "tts_models/en/ljspeech/glow-tts"
# tts = TTS(model_name)

# # Define the text you want to convert to speech
# text = "hello my name is anurag i am fool"

# # Generate and save the audio
# tts.tts_to_file(text=text, file_path="output.wav")


# _____________________________________________________________________________________________________________________________________________________________________________
# from TTS.api import TTS

# # Initialize the TTS model
# tts = TTS("tts_models/multilingual/multi-dataset/your_tts")

# # Define text
# text = "hell my name is anurag i am fool"

# # Select a speaker (choose from the available ones printed earlier)
# speaker = "female-en-5"  # Example: Change this based on available speakers

# # Set the language (needed for multilingual models)
# language = "en"  # Change this based on the required language

# # Generate speech and save to file
# tts.tts_to_file(text=text, speaker=speaker, language=language, file_path="output.wav")







# _____________________________________________________________________________________________________________________________________________________________________________

# from TTS.api import TTS

# # Initialize the TTS model
# tts = TTS("tts_models/multilingual/multi-dataset/your_tts")

# # Display available speakers and languages
# print("Available speakers:", tts.speakers)
# print("Available languages:", tts.languages)

# # Get user input for speaker
# speaker = input("\nEnter the speaker name from the list above: ").strip()

# # Get user input for language
# language = input("\nEnter the language code (e.g., 'en' for English, 'pt' for Portuguese): ").strip()

# # Get user input for text
# text = input("\nEnter the text to convert to speech: ").strip()

# # Generate speech and save to file
# tts.tts_to_file(text=text, speaker=speaker, language=language, file_path="output.wav")

# print("\nAudio saved as 'output.wav'! 🎵")



# _____________________________________________________________________________________________________________________________________________________________________________




# from TTS.api import TTS

# # Initialize the VITS model
# tts = TTS("tts_models/multilingual/multi-dataset/your_tts")

# # Check if multi-speaker
# if tts.speakers:
#     print("Available speakers:", tts.speakers)
#     speaker = input("\nEnter the speaker name from the list above: ").strip()
# else:
#     speaker = None  # Single-speaker model

# language = input("\nEnter the language code (e.g., 'en' for English, 'pt' for Portuguese): ").strip()

# # Get user input for text
# text = input("\nEnter the text to convert to speech: ").strip()

# # Generate speech and save to file
# tts.tts_to_file(text=text, speaker=speaker, language=language, file_path="output.wav")

# print("\nAudio saved as 'output.wav'! 🎵")









# import os
# import json
# import torch
# import torchaudio
# import numpy as np
# import librosa
# import soundfile as sf
# import pandas as pd
# from tqdm import tqdm
# from TTS.tts.models.tacotron2 import Tacotron2
# from TTS.tts.models.vits import VITS
# from TTS.tts.utils.synthesis import synthesis
# from TTS.tts.utils.text.tokenizer import Tokenizer

# # ========================
# # CONFIGURATION
# # ========================
# CONFIG = {
#     "dataset_path": r"C:\Users\Aditya Pokharna\Documents\AAI_WORKPLACE\TEXT_TO_AUDIO\dataset\cv-valid-train2",
#     "metadata_file": r"C:\Users\Aditya Pokharna\Documents\AAI_WORKPLACE\TEXT_TO_AUDIO\dataset\cv-valid-train.csv",
#     "output_dir": r"C:\Users\Aditya Pokharna\Documents\AAI_WORKPLACE\TEXT_TO_AUDIO\output",
#     "sample_rate": 22050,
#     "batch_size": 16,
#     "num_epochs": 10,
#     "device": "cuda" if torch.cuda.is_available() else "cpu"
# }

# # ========================
# # LOAD DATASET
# # ========================
# def load_dataset(metadata_path):
#     df = pd.read_csv(metadata_path, sep="|", header=None, names=["filename", "text", "unused"])
#     df.drop(columns=["unused"], inplace=True)
#     return df

# def preprocess_audio(file_path, target_sr=CONFIG["sample_rate"]):
#     waveform, sr = librosa.load(file_path, sr=target_sr)
#     return torch.tensor(waveform, dtype=torch.float32)

# # ========================
# # INITIALIZE MODELS
# # ========================
# def create_models():
#     tacotron2 = Tacotron2()
#     tacotron2.to(CONFIG["device"])

#     vits = VITS()
#     vits.to(CONFIG["device"])

#     return tacotron2, vits

# # ========================
# # TRAINING LOOP
# # ========================
# def train(tacotron2, vits, dataset):
#     optimizer_tacotron = torch.optim.Adam(tacotron2.parameters(), lr=0.001)
#     optimizer_vits = torch.optim.Adam(vits.parameters(), lr=0.001)
#     criterion = torch.nn.MSELoss()
    
#     tokenizer = Tokenizer()
    
#     for epoch in range(CONFIG["num_epochs"]):
#         total_loss_tacotron = 0
#         total_loss_vits = 0
        
#         for i, row in tqdm(dataset.iterrows(), total=len(dataset)):
#             file_path = os.path.join(CONFIG["dataset_path"], row["filename"])
#             audio = preprocess_audio(file_path)
#             text = row["text"]
            
#             optimizer_tacotron.zero_grad()
#             optimizer_vits.zero_grad()
            
#             # Convert text to phonemes (Tacotron 2 expects phonemes)
#             phonemes = tokenizer.text_to_ids(text)

#             # Convert to tensors
#             phonemes = torch.tensor(phonemes).unsqueeze(0).to(CONFIG["device"])
#             audio = audio.unsqueeze(0).to(CONFIG["device"])

#             # Step 1: Tacotron 2 (Text → Mel Spectrogram)
#             mel_pred = tacotron2(phonemes)
#             loss_tacotron = criterion(mel_pred, audio)
#             loss_tacotron.backward()
#             optimizer_tacotron.step()
#             total_loss_tacotron += loss_tacotron.item()

#             # Step 2: VITS (Mel Spectrogram → Audio)
#             audio_pred = vits(mel_pred)
#             loss_vits = criterion(audio_pred, audio)
#             loss_vits.backward()
#             optimizer_vits.step()
#             total_loss_vits += loss_vits.item()
        
#         print(f"Epoch {epoch+1}/{CONFIG['num_epochs']}, Tacotron Loss: {total_loss_tacotron/len(dataset)}, VITS Loss: {total_loss_vits/len(dataset)}")
        
#         torch.save(tacotron2.state_dict(), os.path.join(CONFIG["output_dir"], f"tacotron2_epoch_{epoch+1}.pth"))
#         torch.save(vits.state_dict(), os.path.join(CONFIG["output_dir"], f"vits_epoch_{epoch+1}.pth"))

# # ========================
# # TEXT-TO-AUDIO INFERENCE
# # ========================
# def text_to_audio(tacotron2, vits, text, output_file="output.wav"):
#     tacotron2.eval()
#     vits.eval()
    
#     tokenizer = Tokenizer()
#     phonemes = tokenizer.text_to_ids(text)
#     phonemes = torch.tensor(phonemes).unsqueeze(0).to(CONFIG["device"])

#     with torch.no_grad():
#         # Step 1: Generate Mel Spectrogram
#         mel_spec = tacotron2(phonemes)

#         # Step 2: Convert Mel Spectrogram to Audio
#         audio = vits(mel_spec)
#         audio = audio.cpu().numpy()
        
#         # Save output
#         sf.write(output_file, audio, CONFIG["sample_rate"])
#         print(f"Generated audio saved at: {output_file}")

# # ========================
# # RUN TRAINING
# # ========================
# if __name__ == "__main__":
#     os.makedirs(CONFIG["output_dir"], exist_ok=True)
    
#     dataset = load_dataset(CONFIG["metadata_file"])
#     tacotron2, vits = create_models()
    
#     train(tacotron2, vits, dataset)
    
#     text_to_audio(tacotron2, vits, "Hello, this is a test synthesis!")
















# import os
# import json
# from pydub import AudioSegment
# from TTS.tts.configs.glow_tts_config import GlowTTSConfig
# from TTS.tts.configs.shared_configs import BaseDatasetConfig
# from TTS.tts.datasets import TTSDataset
# from TTS.tts.models.glow_tts import GlowTTS
# from TTS.tts.utils.text.tokenizer import TTSTokenizer
# from TTS.utils.audio import AudioProcessor
# from trainer import Trainer, TrainerArgs


# # Convert MP3 to WAV function with absolute paths
# def convert_mp3_to_wav(mp3_path, wav_path):
#     if not os.path.exists(mp3_path):
#         raise FileNotFoundError(f"MP3 file not found: {mp3_path}")
    
#     audio = AudioSegment.from_mp3(mp3_path)
#     audio.export(wav_path, format="wav")
#     return wav_path


# # Custom dataset loader function for your JSON dataset
# def load_custom_samples(json_file_path, base_path, split_ratio=0.9):
#     with open(json_file_path, 'r') as f:
#         data = [json.loads(line.strip()) for line in f.readlines()]
    
#     # Split data into train and eval based on split ratio
#     split_index = int(len(data) * split_ratio)
#     train_samples = []
#     eval_samples = []

#     for item in data[:split_index]:
#         mp3_file = item["audio"]
#         mp3_path = os.path.join(base_path, mp3_file)  # Full path to MP3
#         wav_file = mp3_file.replace(".mp3", ".wav")  # Replace extension to .wav
#         wav_path = os.path.join(base_path, wav_file)  # Full path to WAV
#         try:
#             wav_file = convert_mp3_to_wav(mp3_path, wav_path)
#             train_samples.append({
#                 "text": item["text"],
#                 "audio": wav_file,
#                 "accent": item["accent"],
#                 "gender": item["gender"]
#             })
#         except FileNotFoundError as e:
#             print(f"Skipping {mp3_path}: {e}")

#     for item in data[split_index:]:
#         mp3_file = item["audio"]
#         mp3_path = os.path.join(base_path, mp3_file)  # Full path to MP3
#         wav_file = mp3_file.replace(".mp3", ".wav")  # Replace extension to .wav
#         wav_path = os.path.join(base_path, wav_file)  # Full path to WAV
#         try:
#             wav_file = convert_mp3_to_wav(mp3_path, wav_path)
#             eval_samples.append({
#                 "text": item["text"],
#                 "audio": wav_file,
#                 "accent": item["accent"],
#                 "gender": item["gender"]
#             })
#         except FileNotFoundError as e:
#             print(f"Skipping {mp3_path}: {e}")

#     return train_samples, eval_samples


# # Base path for dataset and metadata
# base_path = r"C:\Users\Aditya Pokharna\Documents\AAI_WORKPLACE\TEXT_TO_AUDIO\dataset"

# # Paths for dataset and metadata
# dataset_path = os.path.join(base_path, "dataset", "cv-valid-train")
# metadata_path = os.path.join(base_path, "metadata.jsonl")
# output_path = os.path.join(base_path, "output")


# # Dataset configuration
# dataset_config = BaseDatasetConfig(
#     formatter="custom",  # You can add your own formatter if needed
#     meta_file_train="metadata.jsonl",  # Path to your JSON metadata file
#     path=dataset_path  # Path to the dataset directory
# )


# # Initialize the model configuration
# config = GlowTTSConfig(
#     batch_size=32,
#     eval_batch_size=16,
#     num_loader_workers=4,
#     num_eval_loader_workers=4,
#     run_eval=True,
#     test_delay_epochs=-1,
#     epochs=1,
#     text_cleaner="phoneme_cleaners",
#     use_phonemes=True,
#     phoneme_language="en-us",
#     phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
#     print_step=25,
#     print_eval=False,
#     mixed_precision=True,
#     output_path=output_path,
#     datasets=[dataset_config],
# )


# # Initialize the Audio Processor and Tokenizer
# ap = AudioProcessor.init_from_config(config)
# tokenizer, config = TTSTokenizer.init_from_config(config)


# # Load the custom dataset (train and eval samples)
# train_samples, eval_samples = load_custom_samples(metadata_path, base_path)


# # Initialize the GlowTTS model
# model = GlowTTS(config, ap, tokenizer, speaker_manager=None)


# # Initialize the Trainer
# trainer = Trainer(
#     TrainerArgs(), config, output_path, model=model, train_samples=train_samples, eval_samples=eval_samples
# )


# # Start training
# trainer.fit()




# import os
# import json
# from pydub import AudioSegment
# from TTS.tts.configs.glow_tts_config import GlowTTSConfig
# from TTS.tts.configs.shared_configs import BaseDatasetConfig
# from TTS.tts.datasets import TTSDataset
# from TTS.tts.models.glow_tts import GlowTTS
# from TTS.tts.utils.text.tokenizer import TTSTokenizer
# from TTS.utils.audio import AudioProcessor
# from trainer import Trainer, TrainerArgs
# from torch.utils.data import DataLoader


# # Convert MP3 to WAV function with absolute paths (to ensure it's done once)
# def convert_mp3_to_wav(mp3_path, wav_path):
#     if not os.path.exists(mp3_path):
#         raise FileNotFoundError(f"MP3 file not found: {mp3_path}")
    
#     audio = AudioSegment.from_mp3(mp3_path)
#     audio.export(wav_path, format="wav")
#     return wav_path


# # Custom dataset loader function for your JSON dataset
# def load_custom_samples(json_file_path, base_path, split_ratio=0.9):
#     with open(json_file_path, 'r') as f:
#         data = [json.loads(line.strip()) for line in f.readlines()]
    
#     print(f"Total samples found in metadata: {len(data)}")  # Debug print

#     split_index = int(len(data) * split_ratio)
#     train_samples = []
#     eval_samples = []

#     for idx, item in enumerate(data[:split_index]):
#         mp3_file = item["audio"]
#         mp3_path = os.path.join(base_path, mp3_file)
#         wav_file = mp3_file.replace(".mp3", ".wav")
#         wav_path = os.path.join(base_path, wav_file)
        
#         print(f"Processing train sample {idx+1}/{split_index}: {mp3_path}")  # Debug print
#         try:
#             wav_file = convert_mp3_to_wav(mp3_path, wav_path)
#             train_samples.append({
#                 "text": item["text"],
#                 "audio": wav_file,
#                 "accent": item["accent"],
#                 "gender": item["gender"]
#             })
#         except FileNotFoundError as e:
#             print(f"Skipping {mp3_path}: {e}")

#     for idx, item in enumerate(data[split_index:]):
#         mp3_file = item["audio"]
#         mp3_path = os.path.join(base_path, mp3_file)
#         wav_file = mp3_file.replace(".mp3", ".wav")
#         wav_path = os.path.join(base_path, wav_file)
        
#         print(f"Processing eval sample {idx+1}/{len(data)-split_index}: {mp3_path}")  # Debug print
#         try:
#             wav_file = convert_mp3_to_wav(mp3_path, wav_path)
#             eval_samples.append({
#                 "text": item["text"],
#                 "audio": wav_file,
#                 "accent": item["accent"],
#                 "gender": item["gender"]
#             })
#         except FileNotFoundError as e:
#             print(f"Skipping {mp3_path}: {e}")

#     print(f"Total Train Samples: {len(train_samples)}")
#     print(f"Total Eval Samples: {len(eval_samples)}")

#     return train_samples, eval_samples



# # Base path for dataset and metadata
# base_path = r"C:\Users\Aditya Pokharna\Documents\AAI_WORKPLACE\TEXT_TO_AUDIO\dataset"

# # Paths for dataset and metadata
# dataset_path = os.path.join(base_path, "dataset", "cv-valid-train")
# metadata_path = os.path.join(base_path, "metadata.jsonl")
# output_path = os.path.join(base_path, "output")


# # Dataset configuration
# dataset_config = BaseDatasetConfig(
#     formatter="custom",  # You can add your own formatter if needed
#     meta_file_train="metadata.jsonl",  # Path to your JSON metadata file
#     path=dataset_path  # Path to the dataset directory
# )


# # Initialize the model configuration
# config = GlowTTSConfig(
#     batch_size=128,  # Increased batch size for faster training
#     eval_batch_size=16,
#     num_loader_workers=16,  # Increased number of workers for loading
#     num_eval_loader_workers=4,
#     run_eval=True,
#     test_delay_epochs=-1,
#     epochs=5,  # Training for 5 epochs
#     text_cleaner="phoneme_cleaners",
#     use_phonemes=True,
#     phoneme_language="en-us",
#     phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
#     print_step=25,
#     print_eval=False,
#     mixed_precision=True,  # Enabling mixed precision
#     output_path=output_path,
#     datasets=[dataset_config],
# )


# # Initialize the Audio Processor and Tokenizer
# ap = AudioProcessor.init_from_config(config)
# tokenizer, config = TTSTokenizer.init_from_config(config)


# # Load the custom dataset (train and eval samples)
# train_samples, eval_samples = load_custom_samples(metadata_path, base_path)


# # Initialize the GlowTTS model
# model = GlowTTS(config, ap, tokenizer, speaker_manager=None)


# # Initialize the Trainer
# trainer_args = TrainerArgs(
#     batch_size=config.batch_size,
#     epochs=config.epochs,
#     mixed_precision=config.mixed_precision
# )
# trainer = Trainer(
#     trainer_args, config, output_path, model=model, train_samples=train_samples, eval_samples=eval_samples
# )


# # Create DataLoader for training and evaluation with optimal configurations
# train_loader = DataLoader(train_samples, batch_size=config.batch_size, shuffle=True, num_workers=config.num_loader_workers)
# eval_loader = DataLoader(eval_samples, batch_size=config.eval_batch_size, shuffle=False, num_workers=config.num_eval_loader_workers)


# # Start training
# trainer.fit(train_loader, eval_loader)














import os
import pandas as pd
from trainer import Trainer, TrainerArgs
from TTS.tts.configs.glow_tts_config import GlowTTSConfig
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.glow_tts import GlowTTS
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

# ----------------------------
# 1. Convert Your Metadata
# ----------------------------

# Load dataset
metadata_path = r"C:\Users\Aditya Pokharna\Documents\AAI_WORKPLACE\TEXT_TO_AUDIO\dataset\cv-valid-train.csv"
df = pd.read_csv(metadata_path)

# Select only the necessary columns
df = df[["filename", "text"]]

# Convert MP3 to WAV (if applicable)
df["filename"] = df["filename"].str.replace(".mp3", ".wav", regex=False)

# Ensure the metadata is saved in the correct dataset folder
dataset_path = r"C:\Users\Aditya Pokharna\Documents\AAI_WORKPLACE\TEXT_TO_AUDIO\dataset\cv-valid-train2"
metadata_ljspeech = os.path.join(dataset_path, "metadata_ljspeech.csv")

# Save in LJSpeech format
df["speaker_id"] = "1"  # Add a dummy speaker ID
df.to_csv(metadata_ljspeech, sep="|", index=False, header=False)


print(f"✅ Metadata saved at: {metadata_ljspeech}")

# ----------------------------
# 2. Training Setup
# ----------------------------

# Define dataset configuration
dataset_config = BaseDatasetConfig(
    formatter="ljspeech",  
    meta_file_train=metadata_ljspeech,  # Correct metadata path
    path=os.path.abspath(dataset_path)  # Path where WAV files are stored
)

# Define model training configuration
config = GlowTTSConfig(
    batch_size=32,  
    eval_batch_size=16,
    num_loader_workers=4,
    num_eval_loader_workers=4,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=1000,
    text_cleaner="phoneme_cleaners",
    use_phonemes=True,
    phoneme_language="en-us",
    phoneme_cache_path="phoneme_cache",
    print_step=25,
    print_eval=False,
    mixed_precision=True,  
    output_path=os.getcwd(),  
    datasets=[dataset_config],
)

# Initialize the Audio Processor
ap = AudioProcessor.init_from_config(config)

# Initialize Tokenizer
tokenizer, config = TTSTokenizer.init_from_config(config)

# Load dataset samples
train_samples, eval_samples = load_tts_samples(
    dataset_config,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
)

# Initialize the model
model = GlowTTS(config, ap, tokenizer, speaker_manager=None)

# Initialize the Trainer
trainer = Trainer(
    TrainerArgs(), config, config.output_path, model=model, train_samples=train_samples, eval_samples=eval_samples
)

# Start Training
trainer.fit()














# import os
# import torch
# import torchaudio
# import pandas as pd
# from torch.utils.data import Dataset, DataLoader
# from torch import nn, optim
# import torch.nn.functional as F
# from torch.nn.utils.rnn import pad_sequence

# class TextToSpeechDataset(Dataset):
#     def __init__(self, csv_path, audio_dir):
#         self.data = pd.read_csv(csv_path)
#         self.audio_dir = audio_dir
#         self.char_to_idx = {char: idx for idx, char in enumerate("abcdefghijklmnopqrstuvwxyz0123456789 .,!?")}
    
#     def __len__(self):
#         return len(self.data)
    
#     def text_to_sequence(self, text):
#         return [self.char_to_idx.get(char, 0) for char in text.lower()]
    
#     def __getitem__(self, idx):
#         row = self.data.iloc[idx]
#         text = row['text']
#         audio_path = os.path.join(self.audio_dir, os.path.basename(row['filename']))

#         if not os.path.exists(audio_path):
#             print(f"Warning: File not found {audio_path}, skipping.")
#             return None
        
#         try:
#             waveform, _ = torchaudio.load(audio_path)  
#             text_seq = torch.tensor(self.text_to_sequence(text), dtype=torch.long)
#             return text_seq, waveform.squeeze(0)  # Ensure 1D waveform
#         except Exception as e:
#             print(f"Error loading {audio_path}: {e}")
#             return None

# def collate_fn(batch):
#     batch = [item for item in batch if item is not None]
#     if len(batch) == 0:
#         return None

#     text_seqs, audio_tensors = zip(*batch)
#     text_seqs = pad_sequence(text_seqs, batch_first=True, padding_value=0)
#     audio_tensors = pad_sequence(audio_tensors, batch_first=True, padding_value=0.0)

#     return text_seqs, audio_tensors

# class TransformerTTS(nn.Module):
#     def __init__(self, vocab_size, hidden_dim, num_heads, num_layers):
#         super(TransformerTTS, self).__init__()
#         self.embedding = nn.Embedding(vocab_size, hidden_dim)
#         encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True)
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
#         self.fc = nn.Linear(hidden_dim, 1)  # Predicts one audio sample per timestep
    
#     def forward(self, x, target_length):
#         x = self.embedding(x)  
#         x = self.transformer(x)
#         x = self.fc(x).squeeze(-1)  # (batch, seq_len)

#         # Resample output to match target waveform length
#         x = F.interpolate(x.unsqueeze(1), size=target_length, mode='linear', align_corners=False).squeeze(1)
#         return x

# def train_tts(model, dataloader, epochs=10, lr=0.001):
#     criterion = nn.MSELoss()
#     optimizer = optim.Adam(model.parameters(), lr=lr)
    
#     for epoch in range(epochs):
#         total_loss = 0
#         batch_count = 0
#         for batch in dataloader:
#             if batch is None:
#                 continue

#             text_seq, waveform = batch
#             text_seq = text_seq.to(torch.long)

#             optimizer.zero_grad()
#             target_length = waveform.shape[-1]
#             output = model(text_seq, target_length)  # Ensure output matches waveform length

#             loss = criterion(output, waveform)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
#             batch_count += 1
        
#         avg_loss = total_loss / (batch_count if batch_count > 0 else 1)
#         print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

# # Paths to data
# data_path = r"C:\Users\Aditya Pokharna\Documents\AAI_WORKPLACE\TEXT_TO_AUDIO\dataset\cv-valid-train.csv"
# audio_path = r"C:\Users\Aditya Pokharna\Documents\AAI_WORKPLACE\TEXT_TO_AUDIO\dataset\cv-valid-train"

# dataset = TextToSpeechDataset(data_path, audio_path)
# dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

# # Model definition
# vocab_size = 40  
# model = TransformerTTS(vocab_size, hidden_dim=256, num_heads=4, num_layers=3)

# train_tts(model, dataloader)


























# import os
# import torch
# import torchaudio
# import pandas as pd
# from torch.utils.data import Dataset, DataLoader
# from torch import nn, optim
# import torch.nn.functional as F
# from torch.nn.utils.rnn import pad_sequence
# from tqdm import tqdm  # For progress display

# torchaudio.set_audio_backend("ffmpeg")

# class TextToSpeechDataset(Dataset):
#     def __init__(self, csv_path, audio_dir):
#         self.data = pd.read_csv(csv_path)
#         self.audio_dir = audio_dir
#         self.char_to_idx = {char: idx for idx, char in enumerate("abcdefghijklmnopqrstuvwxyz0123456789 .,!?")}
    
#     def __len__(self):
#         return len(self.data)
    
#     def text_to_sequence(self, text):
#         return [self.char_to_idx.get(char, 0) for char in text.lower()]
    
#     def __getitem__(self, idx):
#         row = self.data.iloc[idx]
#         text = row['text']
#         audio_path = os.path.join(self.audio_dir, os.path.basename(row['filename']))

#         waveform, _ = torchaudio.load(audio_path)  # Load audio
#         waveform = waveform.squeeze(0)  # Ensure shape is [audio_len]

#         return torch.tensor(self.text_to_sequence(text), dtype=torch.long), waveform



# # Custom collate function
# def collate_fn(batch):
#     text_seqs, audio_tensors = zip(*batch)
#     text_seqs = pad_sequence([torch.tensor(seq) for seq in text_seqs], batch_first=True, padding_value=0)
#     audio_tensors = pad_sequence(audio_tensors, batch_first=True, padding_value=0.0)  
#     return text_seqs, audio_tensors

# class TransformerTTS(nn.Module):
#     def __init__(self, vocab_size, hidden_dim, num_heads, num_layers, max_audio_len):
#         super(TransformerTTS, self).__init__()
#         self.embedding = nn.Embedding(vocab_size, hidden_dim)
#         encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
#         self.fc = nn.Linear(hidden_dim, max_audio_len)  # Output directly to audio length

#     def forward(self, x):
#         x = self.embedding(x).permute(1, 0, 2)  # (batch, seq_len, hidden)
#         x = self.transformer(x)

#         x = self.fc(x).permute(1, 0, 2)  # (batch, seq_len, max_audio_len)

#         # 🔹 Reduce seq_len dimension (mean pooling)
#         x = x.mean(dim=1)  # Now shape is (batch, max_audio_len)

#         return x



# def train_tts(model, dataloader, epochs=1, lr=0.001):
#     criterion = nn.MSELoss()
#     optimizer = optim.Adam(model.parameters(), lr=lr)
    
#     for epoch in range(epochs):
#         total_loss = 0
#         batch_count = 0
#         for batch_idx, (text_seq, waveform) in enumerate(dataloader):
#             optimizer.zero_grad()
            
#             output = model(text_seq)  # Output shape: (batch, max_audio_len)

#             # 🔹 Ensure output matches waveform shape
#             output_resized = F.interpolate(output.unsqueeze(1), size=waveform.shape[1], mode='linear', align_corners=False).squeeze(1)

#             loss = criterion(output_resized, waveform)
#             loss.backward()
#             optimizer.step()
            
#             total_loss += loss.item()
#             batch_count += 1
            
#             # 🔹 Progress tracking
#             progress = (batch_idx + 1) / len(dataloader) * 100
#             print(f"Epoch {epoch+1}/{epochs}, Progress: {progress:.2f}% - Loss: {loss.item():.4f}", end="\r")

#         avg_loss = total_loss / batch_count
#         print(f"\nEpoch {epoch+1} completed. Avg Loss: {avg_loss:.4f}")

#     torch.save(model.state_dict(), save_path)
#     print(f"✅ Model saved at {save_path}")

# # Paths to data
# data_path = r"C:\Users\Aditya Pokharna\Documents\AAI_WORKPLACE\TEXT_TO_AUDIO\dataset\cv-valid-train.csv"
# audio_path = r"C:\Users\Aditya Pokharna\Documents\AAI_WORKPLACE\TEXT_TO_AUDIO\dataset\cv-valid-train"
# save_path= r"C:\Users\Aditya Pokharna\Documents\AAI_WORKPLACE\TEXT_TO_AUDIO\output"

# dataset = TextToSpeechDataset(data_path, audio_path)
# dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

# # Model definition
# # Define max_audio_len dynamically
# max_audio_len = 84672  # Replace with the longest waveform length in your dataset

# # Model definition
# vocab_size = 40  # Assuming 40 distinct characters
# model = TransformerTTS(vocab_size, hidden_dim=256, num_heads=4, num_layers=3, max_audio_len=max_audio_len)

# # Training
# train_tts(model, dataloader)











# import os
# import torch
# import torchaudio
# import pandas as pd
# from torch.utils.data import Dataset, DataLoader
# from torch import nn, optim
# import torch.nn.functional as F
# from torch.nn.utils.rnn import pad_sequence
# from tqdm import tqdm

# torchaudio.set_audio_backend("ffmpeg")

# # Check for GPU availability
# if torch.cuda.is_available():
#     device = torch.device("cuda")
#     print("Using GPU:", torch.cuda.get_device_name(0))
# else:
#     device = torch.device("cpu")
#     print("Using CPU.")


# class TextToSpeechDataset(Dataset):
#     def __init__(self, csv_path, audio_dir):
#         self.data = pd.read_csv(csv_path)
#         self.audio_dir = audio_dir
#         self.char_to_idx = {char: idx for idx, char in enumerate("abcdefghijklmnopqrstuvwxyz0123456789 .,!?")}

#     def __len__(self):
#         return len(self.data)

#     def text_to_sequence(self, text):
#         return [self.char_to_idx.get(char, 0) for char in text.lower()]

#     def __getitem__(self, idx):
#         row = self.data.iloc[idx]
#         text = row['text']
#         audio_path = os.path.join(self.audio_dir, os.path.basename(row['filename']))

#         waveform, _ = torchaudio.load(audio_path)
#         waveform = waveform.squeeze(0)  # Ensure shape is [audio_len]

#         return torch.tensor(self.text_to_sequence(text), dtype=torch.long), waveform


# def collate_fn(batch):
#     text_seqs, audio_tensors = zip(*batch)
#     text_seqs = pad_sequence([torch.tensor(seq) for seq in text_seqs], batch_first=True, padding_value=0)
#     audio_tensors = pad_sequence(audio_tensors, batch_first=True, padding_value=0.0)
#     return text_seqs, audio_tensors


# class TransformerTTS(nn.Module):
#     def __init__(self, vocab_size, hidden_dim, num_heads, num_layers, max_audio_len):
#         super(TransformerTTS, self).__init__()
#         self.embedding = nn.Embedding(vocab_size, hidden_dim)
#         encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
#         self.fc = nn.Linear(hidden_dim, max_audio_len)

#     def forward(self, x):
#         x = self.embedding(x).permute(1, 0, 2)  # (seq_len, batch, hidden)
#         x = self.transformer(x)
#         x = self.fc(x).permute(1, 0, 2)  # (batch, seq_len, max_audio_len)
#         x = x.mean(dim=1)  # (batch, max_audio_len)
#         return x


# def train_tts(model, dataloader, epochs=1, lr=0.001):
#     criterion = nn.MSELoss()
#     optimizer = optim.Adam(model.parameters(), lr=lr)

#     for epoch in range(epochs):
#         total_loss = 0
#         batch_count = 0
#         for batch_idx, (text_seq, waveform) in enumerate(dataloader):
#             text_seq = text_seq.to(device).long()  # Move to GPU and ensure correct type
#             waveform = waveform.to(device).float()  # Move to GPU and ensure correct type

#             optimizer.zero_grad()
#             output = model(text_seq)

#             output_resized = F.interpolate(output.unsqueeze(1), size=waveform.shape[1], mode='linear', align_corners=False).squeeze(1)

#             loss = criterion(output_resized, waveform)
#             loss.backward()
#             optimizer.step()

#             total_loss += loss.item()
#             batch_count += 1

#             progress = (batch_idx + 1) / len(dataloader) * 100
#             print(f"Epoch {epoch+1}/{epochs}, Progress: {progress:.2f}% - Loss: {loss.item():.4f}", end="\r")

#         avg_loss = total_loss / batch_count
#         print(f"\nEpoch {epoch+1} completed. Avg Loss: {avg_loss:.4f}")


# # Paths to data (replace with your actual paths)
# data_path = r"C:\Users\Aditya Pokharna\Documents\AAI_WORKPLACE\TEXT_TO_AUDIO\dataset\cv-valid-train.csv"
# audio_path = r"C:\Users\Aditya Pokharna\Documents\AAI_WORKPLACE\TEXT_TO_AUDIO\dataset\cv-valid-train"

# dataset = TextToSpeechDataset(data_path, audio_path)

# # Determine max_audio_len dynamically:
# max_audio_len = 0
# for _, waveform in dataset:  # Iterate through the dataset
#     max_audio_len = max(max_audio_len, waveform.shape[0])
# print(f"Max Audio Length: {max_audio_len}")

# dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn, pin_memory=True)  # Add pin_memory


# vocab_size = 40
# model = TransformerTTS(vocab_size, hidden_dim=256, num_heads=4, num_layers=3, max_audio_len=max_audio_len).to(device)

# train_tts(model, dataloader, epochs=5, lr=0.0001)  # Example: increase epochs, reduced learning rate