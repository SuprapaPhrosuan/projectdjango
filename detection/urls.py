from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('C03/', views.standingforwardbend, name='standingforwardbend'),
    path('C04/', views.neckstretch_R, name='overheadtricepstretch_L'),
    path('W04/', views.neckstretch_R, name='neckstretch_R'),
    
]
