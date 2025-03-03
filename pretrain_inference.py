from TTS.api import TTS

# Load a pre-trained Tacotron 2 model
tts = TTS("tts_models/en/ljspeech/tacotron2-DDC")

# Generate speech and save to a file
tts.tts_to_file(text="Okay, working with metadata in a CSV file alongside MP3 audio files requires a few key adjustments to your data loading and preprocessing pipeline Here's a breakdown of the steps and considerations", file_path="12output.wav")

