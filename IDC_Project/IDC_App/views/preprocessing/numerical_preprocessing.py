from django.shortcuts import render

def numerical_preprocessing(request):
    numerical_data = request.session.get('uploaded_numerical_data', [])

    preprocessing_steps = [
        "Handle Missing Values",
        "Normalization",
        "Standardization",
        "Outlier Removal",
        "Feature Scaling"
    ]

    return render(request, "preprocessing/numerical_preprocessing.html", {"numerical_data": numerical_data, "preprocessing_steps": preprocessing_steps})
