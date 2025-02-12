from django.shortcuts import render

def audio_preprocessing(request):
    audio_files = request.session.get('uploaded_audio_files', [])

    preprocessing_steps = [
        "Convert Format (MP3 to WAV)",
        "Trim Audio (Start/End Silence Removal)",
        "Normalize Volume",
        "Extract Features (MFCC, Spectrogram)",
        "Noise Reduction"
    ]

    return render(request, "preprocessing/audio_preprocessing.html", {"audio_files": audio_files, "preprocessing_steps": preprocessing_steps})


import os
import librosa
import numpy as np
import pandas as pd

def extract_audio_features(input_folder, segment_duration=5):
    """
    Extracts features from 5-second segments of all audio files in the given folder.

    Parameters:
        input_folder (str): Path to the folder containing audio files.
        segment_duration (int): Duration (in seconds) for each segment. Default is 5 seconds.

    Returns:
        pd.DataFrame: DataFrame containing extracted audio features for each segment.
    """
    feature_list = []

    # Loop through all files in the folder
    for filename in os.listdir(input_folder):
        if filename.endswith(('.wav', '.mp3', '.flac')):  # Check for audio files
            file_path = os.path.join(input_folder, filename)

            try:
                # Load full audio file
                y, sr = librosa.load(file_path, sr=None)  # Load with original sampling rate
                total_duration = librosa.get_duration(y=y, sr=sr)

                # Compute number of segments
                num_segments = int(total_duration // segment_duration)

                for i in range(num_segments):
                    start_sample = i * segment_duration * sr
                    end_sample = start_sample + (segment_duration * sr)
                    segment = y[int(start_sample):int(end_sample)]

                    if len(segment) < segment_duration * sr:
                        continue  # Skip incomplete last segment

                    # Extract features for the segment
                    mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13).mean(axis=1)
                    spectral_centroid = librosa.feature.spectral_centroid(y=segment, sr=sr).mean()
                    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=segment, sr=sr).mean()
                    rolloff = librosa.feature.spectral_rolloff(y=segment, sr=sr).mean()
                    zero_crossing_rate = librosa.feature.zero_crossing_rate(segment).mean()
                    chroma = librosa.feature.chroma_stft(y=segment, sr=sr).mean()
                    rms = librosa.feature.rms(y=segment).mean()

                    # Append features to list
                    feature_list.append([
                        filename, i+1, sr, spectral_centroid, spectral_bandwidth, rolloff, 
                        zero_crossing_rate, chroma, rms, *mfcc
                    ])

            except Exception as e:
                print(f"Error processing {filename}: {e}")

    # Define column names
    columns = ["Filename", "Segment", "Sample Rate", "Spectral Centroid", "Spectral Bandwidth",
               "Rolloff", "Zero Crossing Rate", "Chroma", "RMS"] + [f"MFCC_{i}" for i in range(1, 14)]

    # Create and return DataFrame
    df = pd.DataFrame(feature_list, columns=columns)
    return df
