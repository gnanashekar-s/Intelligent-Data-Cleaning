from django.shortcuts import render
from django.conf import settings
import os
import librosa
import numpy as np
import pandas as pd
from django.core.paginator import Paginator
import matplotlib.pyplot as plt
import noisereduce as nr
import soundfile as sf
from scipy.signal import butter, lfilter

import matplotlib
matplotlib.use("Agg")  # ✅ Force Matplotlib to use a non-GUI backend
import matplotlib.pyplot as plt

def paginate(request):
    page_number = request.GET.get("page", 1)  
    df = pd.read_json(request.session.get("csv_data"))
    paginator = Paginator(df.to_dict(orient="records"), 3)  
    page_obj = paginator.get_page(page_number)
    return page_obj


def visualize_features(original_file_path,processed_file_path, features):
    """
    Generate feature plots for both original and processed audio.
    Returns two lists of URLs: original and processed plots.
    """
    import librosa.display
    import matplotlib.pyplot as plt

    print("Original:",processed_file_path)
    print("Processed:",original_file_path)
    y_original, sr = librosa.load(original_file_path, sr=None)
    y_processed, _ = librosa.load(processed_file_path, sr=None)

    original_plot_urls = []
    processed_plot_urls = []

    for feature in features:
        plt.figure(figsize=(8, 4))

        # Original Audio Plot
        if feature == "waveform":
            librosa.display.waveshow(y_original, sr=sr)
            plt.title("Waveform (Original)")
        elif feature == "spectrogram":
            S = librosa.amplitude_to_db(np.abs(librosa.stft(y_original)), ref=np.max)
            librosa.display.specshow(S, sr=sr, x_axis="time", y_axis="log")
            plt.title("Spectrogram (Original)")
        elif feature == "mfcc":
            mfccs = librosa.feature.mfcc(y=y_original, sr=sr, n_mfcc=13)
            librosa.display.specshow(mfccs, sr=sr, x_axis="time")
            plt.title("MFCC (Original)")
        elif feature == "chroma":
            chroma = librosa.feature.chroma_stft(y=y_original, sr=sr)
            librosa.display.specshow(chroma, sr=sr, x_axis="time")
            plt.title("Chroma (Original)")

        original_plot_path = os.path.join(settings.MEDIA_ROOT, f"{feature}_original.png")
        plt.savefig(original_plot_path)
        plt.close()
        original_plot_urls.append(settings.MEDIA_URL + f"{feature}_original.png")

        # Processed Audio Plot
        plt.figure(figsize=(8, 4))
        if feature == "waveform":
            librosa.display.waveshow(y_processed, sr=sr)
            plt.title("Waveform (Processed)")
        elif feature == "spectrogram":
            S = librosa.amplitude_to_db(np.abs(librosa.stft(y_processed)), ref=np.max)
            librosa.display.specshow(S, sr=sr, x_axis="time", y_axis="log")
            plt.title("Spectrogram (Processed)")
        elif feature == "mfcc":
            mfccs = librosa.feature.mfcc(y=y_processed, sr=sr, n_mfcc=13)
            librosa.display.specshow(mfccs, sr=sr, x_axis="time")
            plt.title("MFCC (Processed)")
        elif feature == "chroma":
            chroma = librosa.feature.chroma_stft(y=y_processed, sr=sr)
            librosa.display.specshow(chroma, sr=sr, x_axis="time")
            plt.title("Chroma (Processed)")

        processed_plot_path = os.path.join(settings.MEDIA_ROOT, f"{feature}_processed.png")
        plt.savefig(processed_plot_path)
        plt.close()
        processed_plot_urls.append(settings.MEDIA_URL + f"{feature}_processed.png")

    return original_plot_urls, processed_plot_urls


def audio_preprocessing(request):
    audio_files = request.session.get('uploaded_audio_files', [])
    page_obj = paginate(request)
    request.session
    if request.method == "POST":
        action = request.POST.get("action")
        if action == "upload_audio":
            if request.FILES.get("audio_input"):
                audio_file = request.FILES["audio_input"]
                audio_dir = os.path.join(settings.MEDIA_ROOT, "audio")
                os.makedirs(audio_dir, exist_ok=True)

                # Save the file
                audio_path = os.path.join(audio_dir, audio_file.name)
                with open(audio_path, "wb+") as destination:
                    for chunk in audio_file.chunks():
                        destination.write(chunk)
                
                request.session["current_audio_file_path"] = f"/media/audio/{audio_file.name}"
                audio_info = analyze_audio(os.path.join(settings.MEDIA_ROOT, "audio", os.path.basename(request.session.get("current_audio_file_path", ""))))
                return render(request, 'preprocessing/audio_preprocessing.html',{"audio_file":{"url":request.session.get("current_audio_file_path")},"page_obj":page_obj,"audio_info":audio_info})
    
        elif action == "add_to_dataset":
            print("Adding")
            current_audio_df  = extract_audio_features_from_file(file_path=request.session.get("current_audio_file_path"))
            existing_data = pd.read_json(request.session.get("csv_data",[]))

            updated_data = pd.concat([existing_data, current_audio_df], ignore_index=True)

            request.session["csv_data"] = updated_data.to_json(orient="records")
            
            return render(request, 'preprocessing/audio_preprocessing.html',{"audio_file":{"url":request.session.get("current_audio_file_path")},"page_obj":page_obj})

        elif action == "visualize_features":
            original_file_path = os.path.join(settings.MEDIA_ROOT, "audio", os.path.basename(request.session.get("current_audio_file_path")))
            features = request.POST.getlist("features")
            print(features)

            # Retrieve the latest processed audio path
            processed_file_path = request.session.get("latest_processed_audio", original_file_path)

            original_plot_urls, processed_plot_urls = visualize_features(original_file_path, processed_file_path, features)

            return render(request, "preprocessing/audio_preprocessing.html", {
                "audio_file": {"url": request.session.get("current_audio_file_path")},
                "processed_audio_file": {"url": settings.MEDIA_URL +"/audio/"+ os.path.basename(processed_file_path)},
                "original_plot_urls": original_plot_urls,
                "processed_plot_urls": processed_plot_urls,
                "page_obj": paginate(request)
            })
        
        if action == "apply_preprocessing":
            original_file_path = os.path.join(settings.MEDIA_ROOT, "audio", os.path.basename(request.session.get("current_audio_file_path")))

            y, sr = librosa.load(original_file_path, sr=None)

            # Apply preprocessing (filters, noise reduction, etc.)
            steps = request.POST.getlist("preprocessing_steps")
            params = request.POST
            y_processed = apply_preprocessing(y, sr, steps, params)

            # Save the processed file
            if original_file_path.endswith(".wav"):
                processed_file_path = original_file_path.replace(".wav", "_processed.wav")
            if original_file_path.endswith(".mp3"):
                processed_file_path = original_file_path.replace(".mp3","_processed.mp3")
            sf.write(processed_file_path, y_processed, sr)
            print("PPPP:",processed_file_path)

            # Store the latest processed file path in session
            request.session["latest_processed_audio"] = processed_file_path

            return render(request, 'preprocessing/audio_preprocessing.html', {
                "audio_file": {"url": request.session.get("current_audio_file_path")},
                "processed_audio_file": {"url": settings.MEDIA_URL+'/audio/' + os.path.basename(processed_file_path)},
                "page_obj": paginate(request)
            })

        """if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return render(request, "preprocessing/dataset_table.html", {"page_obj": page_obj})
"""
    return render(request, 'preprocessing/audio_preprocessing.html',{"audio_file":{"url":request.session.get("current_audio_file_path")},"page_obj":page_obj})

def butter_filter(y, sr, cutoff, filter_type="low"):
    nyquist = 0.5 * sr
    normal_cutoff = cutoff / nyquist
    b, a = butter(4, normal_cutoff, btype=filter_type, analog=False)
    return lfilter(b, a, y)

def apply_preprocessing(y, sr, steps, params):
    if "noise_reduction" in steps:
        noise_strength = float(params.get("noise_strength", 0.1))
        y = nr.reduce_noise(y=y, sr=sr, prop_decrease=noise_strength)
    
    if "low_pass" in steps:
        cutoff = float(params.get("low_pass_cutoff", 1000))
        y = butter_filter(y = y, sr = sr, cutoff=cutoff, filter_type="low")

    if "high_pass" in steps:
        cutoff = float(params.get("high_pass_cutoff", 500))
        y = butter_filter(y = y, sr = sr, cutoff=cutoff, filter_type="high")

    if "pitch_shift" in steps:
        semitones = int(params.get("pitch_steps", 0))
        y = librosa.effects.pitch_shift(y = y, sr = sr, n_steps = semitones)

    if "time_stretch" in steps:
        factor = float(params.get("stretch_factor", 1.0))
        y = librosa.effects.time_stretch(y = y, rate = factor)

    return y


def analyze_audio(file_path):
    y, sr = librosa.load(file_path, sr=None)
    
    # Compute features
    duration = librosa.get_duration(y=y, sr=sr)
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))
    rms_energy = np.mean(librosa.feature.rms(y=y))
    tempo_array, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(tempo_array) if tempo_array.size > 0 else 0.0
    
    return {
        "duration": round(duration, 2),
        "sample_rate": sr,
        "freq_range": round(spectral_rolloff, 2),
        "spectral_bandwidth": round(spectral_bandwidth, 2),
        "zero_crossing_rate": round(zero_crossing_rate, 4),
        "rms_energy": round(rms_energy, 4),
        "tempo": round(tempo, 2)
    }


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


def extract_audio_features_from_file(file_path, segment_duration=5):

    """
    Extracts features from 5-second segments of an audio file.
    
    Parameters:
        file_path (str): Path to the audio file.
        segment_duration (int): Duration (in seconds) for each segment. Default is 5 seconds.

    Returns:
        pd.DataFrame: Extracted features for each segment.
    """
    feature_list = []
    filename = os.path.basename(file_path)
    file_path = os.path.join(settings.MEDIA_ROOT, "audio", filename)

    try:
        # Load the full audio file
        y, sr = librosa.load(file_path, sr=None)
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
        return pd.DataFrame()  # Return empty DataFrame on failure

    # Define column names
    columns = ["Filename", "Segment", "Sample Rate", "Spectral Centroid", "Spectral Bandwidth",
               "Rolloff", "Zero Crossing Rate", "Chroma", "RMS"] + [f"MFCC_{i}" for i in range(1, 14)]

    return pd.DataFrame(feature_list, columns=columns)


