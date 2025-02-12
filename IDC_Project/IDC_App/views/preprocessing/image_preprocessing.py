from django.shortcuts import render

def image_preprocessing(request):
    image_files = request.session.get('uploaded_image_files', [])

    preprocessing_steps = [
        "Resize",
        "Convert to Grayscale",
        "Apply Filters",
        "Noise Reduction",
        "Edge Detection"
    ]

    return render(request, "preprocessing/image_preprocessing.html", {"image_files": image_files, "preprocessing_steps": preprocessing_steps})
