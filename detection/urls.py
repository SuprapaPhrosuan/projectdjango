from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('C03/', views.standingforwardbend, name='standingforwardbend'),
    path('W04/', views.neckstretch_R, name='overheadtricepstretch_L'),
    path('C07/', views.quadricepstretch_R, name='quadricepstretch_R'),
    path('C09/', views.shoulderstretch_L, name='shoulderstretch_L'),
    path('E01/', views.elbowplank, name='elbowplank'),
    path('C04/', views.overheadtricepstretch_L, name='overheadtricepstretch_L'),
    
]
