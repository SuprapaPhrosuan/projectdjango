from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('C01/', views.cobrastretch, name='cobrastretch'),
    path('C02/', views.childpose, name='childpose'),
    path('C03/', views.standingforwardbend, name='standingforwardbend'),
    path('C04/', views.overheadtricepstretch_L, name='overheadtricepstretch_L'),
    path('C05/', views.overheadtricepstretch_R, name='overheadtricepstretch_R'),
    path('C06/', views.quadricepstretch_L, name='quadricepstretch_L'),
    path('C07/', views.quadricepstretch_R, name='quadricepstretch_R'),
    path('C08/', views.shoulderstretch_R, name='shoulderstretch_R'),
    path('C09/', views.shoulderstretch_L, name='shoulderstretch_L'),
    path('E01/', views.elbowplank, name='elbowplank'),
    path('W04/', views.neckstretch_R, name='neckstretch_R'),
    path('W05/', views.neckstretch_L, name='neckstretch_L'),
    path('W06/', views.reachingup, name='reachingup'),
    path('W07/', views.reachingdown, name='reachingdown'),
    path('W09/', views.standingsidebend_R, name='standingsidebend_R'),
]
