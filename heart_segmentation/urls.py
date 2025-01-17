from django.urls import path

from . import views

from django.conf.urls.static import static
from django.conf import settings

app_name = "heart_segmentation"
urlpatterns = [
    path("", views.index, name="index"),
    path("result/", views.result, name="result"),
    path("process_image/", views.process_image, name="process_image")
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)