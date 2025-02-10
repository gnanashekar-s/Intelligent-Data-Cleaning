from django.urls import path
from .views import UploadAndView,home

urlpatterns = [
    path('', home, name='home'),
    path('upload/',UploadAndView, name = 'upload')
]