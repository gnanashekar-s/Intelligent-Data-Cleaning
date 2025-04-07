from django.shortcuts import render
from django.core.paginator import Paginator
import pandas as pd
import numpy as np
from io import StringIO
from sklearn.preprocessing import LabelEncoder
from scipy import stats
import matplotlib.pyplot as plt
import io
import base64
import seaborn as sns
def categorical_preprocessing(request):
    context = {}
    HISTORY_LIMIT = 5
    outlier_plot = None
    
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
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='ignore')  # Keep strings if not convertible

    # Handle Undo
    if 'undo' in request.POST and request.session['history']:
        request.session['csv_data'] = request.session['history'].pop()
        request.session.modified = True
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

                
            # Outlier Handling
            action = request.POST.get('action', '')
            if action in ['visualize_outliers', 'remove_outliers']:
                method = request.POST.get('outlier_method')
                selected_columns = request.POST.getlist('selected_columns_outliers')
                for col in selected_columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

                selected_columns = [col for col in selected_columns if col in df.select_dtypes(include=[np.number]).columns]

                if selected_columns:
                  if action == 'visualize_outliers':                    fig, axes = plt.subplots(len(selected_columns), 1, figsize=(6, 4 * len(selected_columns)))
                    if len(selected_columns) == 1:
                        axes = [axes]
                    for ax, col in zip(axes, selected_columns):
                        sns.boxplot(data=df, x=col, ax=ax)
                        ax.set_title(f'Boxplot ({method.upper()}) - {col}')
                    buf = io.BytesIO()
                    plt.tight_layout()
                    plt.savefig(buf, format='png')
                    plt.close(fig)
                    buf.seek(0)
                    outlier_plot = base64.b64encode(buf.read()).decode('utf-8')

                  elif action == 'remove_outliers' and selected_columns:
                    if method == 'zscore':
                        col_zscores = np.abs(df[selected_columns].apply(stats.zscore))
                        df = df[(col_zscores < 3).all(axis=1)]
                    elif method == 'iqr':
                        Q1 = df[selected_columns].quantile(0.25)
                        Q3 = df[selected_columns].quantile(0.75)
                        IQR = Q3 - Q1
                        mask = ~((df[selected_columns] < (Q1 - 1.5 * IQR)) | (df[selected_columns] > (Q3 + 1.5 * IQR))).any(axis=1)
                        df = df[mask]
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
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(), 
        'columns_with_nulls': [(col, df[col].isna().sum()) for col in df.columns],
        'page_data': page_obj.object_list,
        'current_page': page_obj.number,
        'prev_page': page_obj.previous_page_number() if page_obj.has_previous() else None,
        'next_page': page_obj.next_page_number() if page_obj.has_next() else None,
        'total_pages': paginator.num_pages,
        'history_count': len(request.session.get('history', [])),
        'outlier_plot': outlier_plot,
    })

    return render(request, 'preprocessing/categorical_preprocessing.html', context)