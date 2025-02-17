from django.shortcuts import render,redirect
from django.core.paginator import Paginator
import pandas as pd
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from io import StringIO
import os
from django.core.files.storage import FileSystemStorage
from .preprocessing.audio_preprocessing import extract_audio_features

def add_to_session(request, key, value):
    """Helper function to add data to session."""
    request.session[key] = value
    request.session.modified = True  


def get_from_session(request, key, default=None):
    """Helper function to retrieve data from session."""
    return request.session.get(key, default)


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Get project root

MEDIA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'media')
AUDIO_FOLDER_PATH = os.path.join(BASE_DIR, "media", "audio")

import csv
from django.http import HttpResponse

import json

def download_csv(request):
    csv_data = get_from_session(request, "csv_data")

    if csv_data:
        dataset = json.loads(csv_data)  # Convert JSON string back to list of dictionaries
        
        if isinstance(dataset, str):  # Ensure dataset is properly loaded
            dataset = json.loads(dataset)

        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="output.csv"'
        writer = csv.writer(response)

        # Ensure dataset is a list of dictionaries before accessing keys()
        if dataset and isinstance(dataset[0], dict):
            writer.writerow(dataset[0].keys())  # Write column headers
            for row in dataset:
                writer.writerow(row.values())  # Write data rows
        else:
            return HttpResponse("Invalid dataset format", status=400)

        return response

    return HttpResponse("No CSV data available", status=400)



@csrf_exempt
def UploadAndView(request):
    if request.method == "POST":
        form_type = request.POST.get("form_type")

        if form_type == "uploadform":
            dataset_type = request.POST.get("dataset_type")
            add_to_session(request, "dataset_type", dataset_type)
            uploaded_files = []  # Store uploaded file paths

            if dataset_type == "audio":
                files = request.FILES.getlist("folder_upload")  # Get all files in the folder
                
                if not files:
                    return redirect("upload")

                audio_dir = os.path.join("media", "audio")
                os.makedirs(audio_dir, exist_ok=True)  # Ensure directory exists
                
                for file in files:
                    file_path = os.path.join(audio_dir, file.name)
                    with open(file_path, "wb+") as destination:
                        for chunk in file.chunks():
                            destination.write(chunk)
                    uploaded_files.append(file_path)
                audio_df = extract_audio_features(audio_dir)
                add_to_session(request, "columns", audio_df.columns.tolist())
                add_to_session(request, "csv_data", audio_df.to_json(orient="records"))
                
            
            elif "file_upload" in request.FILES:
                file = request.FILES["file_upload"] # Get single file upload
                
                if not file:
                    return redirect("upload")

                file_name, file_extension = os.path.splitext(file.name)
                file_extension = file_extension.lower()

                # Save file
                fs = FileSystemStorage(location="media/uploads/")
                file_path = fs.save(file.name, file)
                file_full_path = os.path.join("media/uploads/", file_path)

                uploaded_files.append(file_full_path)

                # Process CSV or Excel files
                if file_extension in [".csv", ".xlsx"]:
                    print(file)
                    df = pd.read_csv(file_full_path) if file_extension == ".csv" else pd.read_excel(file)
                    add_to_session(request, "columns", df.columns.tolist())
                    add_to_session(request, "csv_data", df.to_json(orient="records"))
                    add_to_session(request, "dataset_type", dataset_type)

                elif file_extension in [".wav", ".mp3"]:
                    add_to_session(request, "audio_file", file_full_path)  # Store audio path

            # Store uploaded file paths in session
            add_to_session(request, "uploaded_files", uploaded_files)
                
    
    csv_data = request.session.get("csv_data", None)
    df = pd.read_json(StringIO(csv_data)) if csv_data else pd.DataFrame()   
    filter_type = request.POST.get("filter_type", "all")

    selected_columns = request.POST.getlist("required_col")  
    print("Selected Colums:",selected_columns)

    if selected_columns:
        df = df[selected_columns]  


    if not df.empty:
        if filter_type == "integers":
            df = df.select_dtypes(include=[int, float])  
        elif filter_type == "strings":
            df = df.select_dtypes(include=["object"])

    page_number = request.GET.get("page", 1)  
    paginator = Paginator(df.to_dict(orient="records"), 10)  
    page_obj = paginator.get_page(page_number)

    return render(request, "upload.html", {
        "page_obj": page_obj,
        "columns": request.session.get("columns", []),
        "selected_columns": selected_columns,  
    })



def preprocessing_redirect(request):
    dataset_type = get_from_session(request,key="dataset_type")
    print(dataset_type)

    if dataset_type == "audio":
        return redirect('audio_preprocessing')
    elif dataset_type == "image":
        return redirect('image_preprocessing')
    elif dataset_type == "categorical":
        return redirect('categorical_preprocessing')
    elif dataset_type == "numeric":
        return redirect('numerical_preprocessing')

    return redirect('upload')