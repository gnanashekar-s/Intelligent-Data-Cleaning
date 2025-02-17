from django.urls import path
from .views.preprocessing.audio_preprocessing import audio_preprocessing
from .views.preprocessing.image_preprocessing import image_preprocessing
from .views.preprocessing.categorical_preprocessing import categorical_preprocessing
from .views.preprocessing.numerical_preprocessing import numerical_preprocessing
from .views.uploadAndView import UploadAndView,preprocessing_redirect,download_csv
from .views.home import home
urlpatterns = [
    path('' , home ,name = "home"),
    path('upload/', UploadAndView ,name = "upload"),
    path('preprocess/', preprocessing_redirect, name = "preprocess"),
    path('preprocess/numerical/', numerical_preprocessing, name='numerical_preprocessing'),
    path('preprocess/audio/', audio_preprocessing, name='audio_preprocessing'),
    path('preprocess/image/', image_preprocessing, name='image_preprocessing'),
    path('preprocess/categorical/', categorical_preprocessing, name='categorical_preprocessing'),
    path('download_csv/', download_csv, name='download_csv'),
]