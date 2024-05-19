import os
import librosa
import numpy as np
import soundfile as sf

def audio_to_spectrogram(audio_path, n_fft=2048, hop_length=512):
    
    y, sr = librosa.load(audio_path, sr=None)
    spectrogram = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    
    return spectrogram, sr, y

def spectrogram_to_audio(spectrogram, sr, y):
    
    audio_reconstructed = librosa.istft(spectrogram, length=len(y), dtype=np.float32)
    
    return audio_reconstructed

def apply_time_mask(spectrogram, max_length=40, num_masks=1):

    masked_spectrogram = np.copy(spectrogram)
    len_spectro = masked_spectrogram.shape[1]
    
    # Apply time mask
    for i in range(num_masks):
        mask_length = np.random.randint(0, max_length)
        start_p = np.random.randint(0, len_spectro - max_length)  
        mask_end = start_p + mask_length  
        masked_spectrogram[:, start_p:mask_end] = 0
        
    return masked_spectrogram



audio_dir = 'data/dcase2023t2/dev_data/raw/fan/train'

for filename in os.listdir(audio_dir):
    if filename.endswith('.wav'):
        audio_path = os.path.join(audio_dir, filename)
        
        spectrogram, sr, y = audio_to_spectrogram(audio_path)
        masked_spectrogram = apply_time_mask(spectrogram)
        reconstructed_audio = spectrogram_to_audio(masked_spectrogram, sr, y)
        
        
        output_audio_path = os.path.join(audio_dir, 'masked_' + filename )
        sf.write(output_audio_path, reconstructed_audio, sr)

print("Time masking applied to all audio files in the directory.")
