from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('C09/', views.shoulderstretch_L_detection, name='shoulderstretch_L_detection'),
    path('C08/', views.shoulderstretch_R_detection, name='shoulderstretch_R_detection'),
    path('C07/', views.quadricepstretch_R_detection, name='quadricepstretch_R__detection'),
    path('C06/', views.quadricepstretch_L_detection, name='quadricepstretch_L_detection'),
    path('C05/', views.overheadtricepstretch_R_detection, name='overheadtricepstretch_R_detection'),
    path('C04/', views.overheadtricepstretch_L_detection, name='overheadtricepstretch_L_detection'),
    path('C03/', views.standingforwardbend_detection, name='standingforwardbend_detection'),
    path('C02/', views.childpose_detection, name='childpose_detection'),
    path('C01/', views.cobrastretch_detection, name='Cobrastretch_detection'),
    
    path('W09/', views.standingsidebend_R_detection, name='standingsidebend_R_detection'),
    path('W08/', views.standingsidebend_L_detection, name='standingsidebend_L_detection'),
    path('W07/', views.reachingDown_detection, name='reachingDown_detection'),
    path('W06/', views.reachingUp_detection, name='reachingUp_detection'),
    path('W05/', views.neckstretch_L_detection, name='neckstretch_L_detection'),
    path('W04/', views.neckstretch_R_detection, name='neckstretch_R_detection'),
    
    path('E01/', views.elbowplank_detection, name='elbowplank_detection'),
]
