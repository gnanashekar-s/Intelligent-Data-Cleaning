import pandas as pd
import json
from django.shortcuts import render, redirect
from django.core.paginator import Paginator
from django.core.files.storage import FileSystemStorage
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Upload dataset view
def upload_dataset(request):
    if request.method == 'POST' and request.FILES.get('dataset'):
        file = request.FILES['dataset']
        fs = FileSystemStorage()
        filename = fs.save(file.name, file)
        file_path = fs.path(filename)

        # Read CSV and store in session
        df = pd.read_csv(file_path)
        request.session['csv_data'] = df.to_json()
        request.session['filename'] = file.name  

        return redirect('preprocess')  

    return render(request, 'dataset.html')

# Numerical preprocessing view
def numerical_preprocessing(request):
    dataset_json = request.session.get('csv_data')
    df = pd.DataFrame(json.loads(dataset_json)) if dataset_json else None
    description = None
    dataset_preview = None

    if df is not None:
        # Generate dataset description
        description = {
            col: {
                "mean": round(df[col].mean(), 2) if df[col].dtype != 'object' else "N/A",
                "median": round(df[col].median(), 2) if df[col].dtype != 'object' else "N/A",
                "mode": round(df[col].mode().iloc[0], 2) if not df[col].mode().empty and df[col].dtype != 'object' else "N/A",
            }
            for col in df.columns
        }

        if request.method == "POST":
            action = request.POST.get("action")

            if action == "handle_missing":
                print("Miss")
                df.fillna(0, inplace=True)

            elif action == "normalize":
                print("NORM")
                numeric_cols = df.select_dtypes(include=['number']).columns
                df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].min()) / (df[numeric_cols].max() - df[numeric_cols].min())

            elif action == "standardize":
                print("Stand")
                numeric_cols = df.select_dtypes(include=['number']).columns
                df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std()

            elif action == "remove_outliers":
                print("RemoveOuT")
                numeric_cols = df.select_dtypes(include=['number']).columns
                for col in numeric_cols:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    # Corrected condition: Keep rows that are NOT outliers
                    df = df[~((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR)))]

            elif action == "scale_features":
                print("Scale")
                numeric_cols = df.select_dtypes(include=['number']).columns
                scaler = MinMaxScaler()
                df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

            # Save the updated dataset to the session
            request.session['csv_data'] = df.to_json()
            return redirect('preprocess')

        # Implement pagination (show 10 rows per page)
        paginator = Paginator(df.to_dict(orient="records"), 10)
        page_number = request.GET.get("page", 1)
        dataset_preview = paginator.get_page(page_number)

    return render(request, 'preprocessing/numerical_preprocessing.html', {
        'description': description,
        'dataset': dataset_preview
    })