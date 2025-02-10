from django.shortcuts import render

# Create your views here.
import pandas as pd
from django.http import JsonResponse
from io import StringIO

def UploadAndView(request):
    if request.method == "POST" and request.FILES.get("csv_file"):
        csv_file = request.FILES["csv_file"]
        df = pd.read_csv(csv_file)  # Read CSV into Pandas DataFrame
        
        # Store dataset in session (convert DataFrame to JSON)
        request.session["csv_data"] = df.to_json()

    # Retrieve dataset from session
    csv_data = request.session.get("csv_data", None)
    df = pd.read_json(StringIO(csv_data)) if csv_data else pd.DataFrame()

    # Filtering Logic
    filter_type = request.GET.get("filter_type", "all")
    column_name = request.GET.get("column_name", "")

    if not df.empty:
        if filter_type == "integers":
            df = df.select_dtypes(include=["int", "float"])  # Keep only numeric columns

        elif filter_type == "strings":
            df = df.select_dtypes(include=["object"])  # Keep only string columns

        elif filter_type == "column" and column_name in df.columns:
            df = df[[column_name]]  # Keep only the specified column

    return render(request, "upload.html", {"headers": df.columns.tolist(), "data": df.values.tolist()})

def home(request):
    return render(request, 'home.html')