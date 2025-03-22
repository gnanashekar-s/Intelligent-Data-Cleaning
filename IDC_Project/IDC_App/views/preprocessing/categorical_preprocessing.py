from django.shortcuts import render
from django.core.paginator import Paginator
import pandas as pd
import numpy as np
from io import StringIO
from sklearn.preprocessing import LabelEncoder
from scipy import stats

def categorical_preprocessing(request):
    context = {}
    HISTORY_LIMIT = 5
    
    # Initialize session data
    if 'csv_data' not in request.session:
        df = pd.DataFrame({
            'Category': ['A', 'B', np.nan, 'C', 'B', 'A', 'N/A', 'C', 'B', 'A'],
            'Price': [100, 200, np.nan, 300, 'N/A', 150, 200, 250, 300, 350],
            'Size': ['Small', 'Medium', 'Large', np.nan, 'Small', 'Medium', 'Large', 'N/A', 'Small', 'Medium']
        })
        request.session['csv_data'] = df.to_json(orient='records')
        request.session['history'] = []
    
    # Load data
    df = pd.read_json(StringIO(request.session['csv_data']))
    
    # Handle Undo
    if 'undo' in request.POST and request.session['history']:
        request.session['csv_data'] = request.session['history'].pop()
        request.session.modified = True
        df = pd.read_json(StringIO(request.session['csv_data']))
    
    if request.method == 'POST' and 'undo' not in request.POST:
        try:
            # Save history
            history = request.session.get('history', [])
            history.append(request.session['csv_data'])
            request.session['history'] = history[-HISTORY_LIMIT:]
            
            df = pd.read_json(StringIO(request.session['csv_data']))
            invalid_columns = []

            # Handle null values
            if 'handle_nulls' in request.POST:
                columns = request.POST.getlist('columns')
                method = request.POST.get('null_method')
                custom_value = request.POST.get('custom_value', '')

                for col in columns:
                    if method == 'mean':
                        if pd.api.types.is_numeric_dtype(df[col]):
                            df[col].fillna(df[col].mean(), inplace=True)
                        else:
                            invalid_columns.append(col)
                    elif method == 'median':
                        if pd.api.types.is_numeric_dtype(df[col]):
                            df[col].fillna(df[col].median(), inplace=True)
                        else:
                            invalid_columns.append(col)
                    elif method == 'mode':
                        df[col].fillna(df[col].mode()[0], inplace=True)
                    elif method == 'custom':
                        df[col].fillna(custom_value, inplace=True)
                    elif method == 'remove':
                        df.dropna(subset=[col], inplace=True)
            if 'remove_outliers' in request.POST:
                columns = request.POST.getlist('outlier_columns')
                method = request.POST.get('outlier_method')
                
                for col in columns:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        if method == 'zscore':
                            # Using Z-score method to detect outliers
                            z_scores = np.abs(stats.zscore(df[col].dropna()))
                            df = df[(z_scores < 3)]  # Keep only rows where Z-score is less than 3
                        elif method == 'iqr':
                            # Using IQR method to detect outliers
                            Q1 = df[col].quantile(0.25)
                            Q3 = df[col].quantile(0.75)
                            IQR = Q3 - Q1
                            df = df[(df[col] >= (Q1 - 1.5 * IQR)) & (df[col] <= (Q3 + 1.5 * IQR))]
            
            # Handle encoding
            if 'encode' in request.POST:
                encode_columns = request.POST.getlist('encode_columns')
                method = request.POST.get('encode_method')

                for col in encode_columns:
                    if method == 'label':
                        le = LabelEncoder()
                        df[col] = le.fit_transform(df[col].astype(str))
                    elif method == 'onehot':
                        df = pd.get_dummies(df, columns=[col], prefix=[col])

             # Handle value removal
            if 'remove_values' in request.POST:
                remove_value = request.POST.get('remove_value', '').strip()
                columns = request.POST.getlist('remove_columns')
                exact_match = 'exact_match' in request.POST  # New checkbox

                mask = pd.Series(False, index=df.index)
                
                for col in columns:
                    if pd.api.types.is_string_dtype(df[col]):
                        # Handle text columns with exact or partial matches
                        if exact_match:
                            col_mask = (df[col].astype(str).str.strip() == remove_value.strip())
                        else:
                            col_mask = df[col].astype(str).str.contains(remove_value.strip(), regex=False)
                    else:
                        # Handle numeric columns
                        try:
                            numeric_value = float(remove_value)
                            col_mask = (df[col] == numeric_value)
                        except ValueError:
                            col_mask = (df[col].astype(str) == remove_value)
                    
                    # Handle NaN separately
                    if remove_value.lower() == 'nan':
                        col_mask = df[col].isna()
                    
                    mask |= col_mask

                df = df[~mask]
            # Update session data
            request.session['csv_data'] = df.to_json(orient='records')
            request.session.modified = True

        except Exception as e:
            context['error'] = str(e)
            context['invalid_columns'] = invalid_columns
            if request.session['history']:
                request.session['csv_data'] = request.session['history'].pop()

    # Prepare context
    page_number = request.GET.get('page', 1)
    paginator = Paginator(df.values.tolist(), 20)
    page_obj = paginator.get_page(page_number)
    
    context.update({
        'columns': df.columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
        'columns_with_nulls': [(col, df[col].isna().sum()) for col in df.columns],
        'page_data': page_obj.object_list,
        'current_page': page_obj.number,
        'prev_page': page_obj.previous_page_number() if page_obj.has_previous() else None,
        'next_page': page_obj.next_page_number() if page_obj.has_next() else None,
        'total_pages': paginator.num_pages,
        'history_count': len(request.session.get('history', []))
    })

    return render(request, 'preprocessing/categorical_preprocessing.html', context)