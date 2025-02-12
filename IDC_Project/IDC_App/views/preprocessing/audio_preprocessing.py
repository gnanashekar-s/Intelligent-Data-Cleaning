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
