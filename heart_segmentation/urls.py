from django.urls import path

from . import views

app_name = "heart_segmentation"
urlpatterns = [
    path("", views.index, name="index"),
    path("process_image/", views.process_image, name="process_image")
]