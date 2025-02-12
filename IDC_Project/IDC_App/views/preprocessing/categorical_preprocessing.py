from django.shortcuts import render

def categorical_preprocessing(request):
    text_files = request.session.get('uploaded_text_files', [])

    preprocessing_steps = [
        "Remove Stopwords",
        "Lowercase Conversion",
        "Lemmatization",
        "Tokenization",
        "Remove Special Characters"
    ]

    return render(request, "preprocessing/categorical_preprocessing.html", {"text_files": text_files, "preprocessing_steps": preprocessing_steps})
