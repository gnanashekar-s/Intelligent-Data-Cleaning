from django.shortcuts import render
from django.core.paginator import Paginator
import pandas as pd
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from io import StringIO
@csrf_exempt
def UploadAndView(request):
    if request.method == "POST" and request.FILES.get("csv_file"):
        csv_file = request.FILES["csv_file"]
        df = pd.read_csv(csv_file)
        request.session["columns"] = df.columns.tolist()
        
        request.session["csv_data"] = df.to_json(orient="records")  # Use records format

    csv_data = request.session.get("csv_data", None)
    df = pd.read_json(StringIO(csv_data)) if csv_data else pd.DataFrame()   
    filter_type = request.POST.get("filter_type", "all")

    selected_columns = request.POST.getlist("required_col")  # Get list of selected columns
    print(selected_columns)

    if selected_columns:
        df = df[selected_columns]  # Filter DataFFilter DataFrame by selected columns


    if not df.empty:
        if filter_type == "integers":
            df = df.select_dtypes(include=[int, float])  
        elif filter_type == "strings":
            df = df.select_dtypes(include=["object"])

    page_number = request.GET.get("page", 1)  # Get current page number
    paginator = Paginator(df.to_dict(orient="records"), 10)  # Show 10 rows per page
    page_obj = paginator.get_page(page_number)

    return render(request, "upload.html", {
        "page_obj": page_obj,
        "columns": request.session.get("columns", []),
        "selected_columns": selected_columns,  # Pass selected columns to template
    })