from django.shortcuts import render
from django.conf import settings
import os
import librosa
import numpy as np
import pandas as pd
from django.core.paginator import Paginator
import matplotlib.pyplot as plt

def paginate(request):
    page_number = request.GET.get("page", 1)  
    df = pd.read_json(request.session.get("csv_data"))
    paginator = Paginator(df.to_dict(orient="records"), 3)  
    page_obj = paginator.get_page(page_number)
    return page_obj

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

                return render(request, 'preprocessing/audio_preprocessing.html',{"audio_file":{"url":request.session.get("current_audio_file_path")},"page_obj":page_obj})
    
        elif action == "add_to_dataset":
            print("Adding")
            current_audio_df  = extract_audio_features_from_file(file_path=request.session.get("current_audio_file_path"))
            existing_data = pd.read_json(request.session.get("csv_data",[]))

            updated_data = pd.concat([existing_data, current_audio_df], ignore_index=True)

            request.session["csv_data"] = updated_data.to_json(orient="records")
            
            return render(request, 'preprocessing/audio_preprocessing.html',{"audio_file":{"url":request.session.get("current_audio_file_path")},"page_obj":page_obj})

        elif action == "vizualiize_features":
            features = request.POST.getlist("features") # Get selected features
            file_path = request.session.get("current_audio_file_path")
            filename = os.path.basename(file_path)
            file_path = os.path.join(settings.MEDIA_ROOT, "audio", filename)
            y, sr = librosa.load(file_path)

            plot_urls = []  # Store generated plot URLs

            for feature in features:
                plt.figure(figsize=(8, 4))

                if feature == "waveform":
                    librosa.display.waveshow(y, sr=sr)
                    plt.title("Waveform")
                
                elif feature == "spectrogram":
                    S = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
                    librosa.display.specshow(S, sr=sr, x_axis="time", y_axis="log")
                    plt.title("Spectrogram")
                
                elif feature == "mfcc":
                    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                    librosa.display.specshow(mfccs, sr=sr, x_axis="time")
                    plt.title("MFCC")
                
                elif feature == "chroma":
                    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
                    librosa.display.specshow(chroma, sr=sr, x_axis="time")
                    plt.title("Chroma")

                # Save each plot separately
                plot_path = os.path.join(settings.MEDIA_ROOT, f"{feature}.png")
                plt.savefig(plot_path)
                plt.close()

                # Append URL for display in the template
                plot_urls.append(settings.MEDIA_URL + f"{feature}.png")

            return render(request, "preprocessing/audio_preprocessing.html", {"audio_file":{"url":request.session.get("current_audio_file_path")},"page_obj":page_obj,"plot_urls": plot_urls})

        """if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return render(request, "preprocessing/dataset_table.html", {"page_obj": page_obj})
"""
    return render(request, 'preprocessing/audio_preprocessing.html',{"audio_file":{"url":request.session.get("current_audio_file_path")},"page_obj":page_obj})
        



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


