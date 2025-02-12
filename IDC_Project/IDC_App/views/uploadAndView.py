from django.shortcuts import render,redirect
from django.core.paginator import Paginator
import pandas as pd
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from io import StringIO

def add_to_session(request, key, value):
    """Helper function to add data to session."""
    request.session[key] = value
    request.session.modified = True  # Ensures session updates immediately


def get_from_session(request, key, default=None):
    """Helper function to retrieve data from session."""
    return request.session.get(key, default)



@csrf_exempt
def UploadAndView(request):
    if request.method == "POST" and request.FILES.get("csv_file"):
        form_type = request.POST.get("form_type")
        if form_type == "uploadform":
            print(request.POST.get("dataset_type"))  #[numeric,categorical,image,audio]
        
        csv_file = request.FILES["csv_file"]
        df = pd.read_csv(csv_file)
        add_to_session(request,key="columns",value=df.columns.tolist())
        add_to_session(request,key="csv_data",value=df.to_json(orient="records"))
        add_to_session(request,key="dataset_type", value = request.POST.get('dataset_type'))

    
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



def preprocessing_redirect(request):
    dataset_type = get_from_session(request,key="dataset_type")
    print(dataset_type)

    if dataset_type == "audio":
        return redirect('audio_preprocessing')
    elif dataset_type == "image":
        return redirect('image_preprocessing')
    elif dataset_type == "categorical":
        return redirect('categorical_preprocessing')
    elif dataset_type == "numerical":
        return redirect('numerical_preprocessing')

    return redirect('upload')