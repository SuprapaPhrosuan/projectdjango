from django.urls import path
from . import views

urlpatterns = [
    path('<str:code>/', views.exercise_view, name='exercise_view'),
]
