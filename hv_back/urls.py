from django.contrib import admin
from django.urls import path, include
from .views import index
urlpatterns = [
    path('', index),
    path('admin/', admin.site.urls),
    path('api/', include('landing.urls')),
    path('api/main/', include('mainpage.urls')), 
]