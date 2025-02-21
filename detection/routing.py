from django.urls import path

from . import (
    childpose, 
    cobrastretch, 
    elbowplank, 
    neckstretch_L, 
    neckstretch_R, 
    overheadtricepstretch_L, 
    overheadtricepstretch_R, 
    quadricepstretch_L, 
    quadricepstretch_R, 
    reachingup, 
    shoulderstretch_L, 
    shoulderstretch_R, 
    standingforwardbend, 
    standingsidebend_L, 
    standingsidebend_R,
)

websocket_urlpatterns = [
    path('ws/detection/childpose/<str:code>/', childpose.StreamConsumer.as_asgi()),
    path('ws/detection/cobrastretch/<str:code>/', cobrastretch.StreamConsumer.as_asgi()),
    path('ws/detection/elbowplank/<str:code>/', elbowplank.StreamConsumer.as_asgi()),
    path('ws/detection/neckstretch_L/<str:code>/', neckstretch_L.StreamConsumer.as_asgi()),
    path('ws/detection/neckstretch_R/<str:code>/', neckstretch_R.StreamConsumer.as_asgi()),
    path('ws/detection/overheadtricepstretch_L/<str:code>/', overheadtricepstretch_L.StreamConsumer.as_asgi()),
    path('ws/detection/overheadtricepstretch_R/<str:code>/', overheadtricepstretch_R.StreamConsumer.as_asgi()),
    path('ws/detection/quadricepstretch_L/<str:code>/', quadricepstretch_L.StreamConsumer.as_asgi()),
    path('ws/detection/quadricepstretch_R/<str:code>/', quadricepstretch_R.StreamConsumer.as_asgi()),
    path('ws/detection/reachingup/<str:code>/', reachingup.StreamConsumer.as_asgi()),
    path('ws/detection/shoulderstretch_L/<str:code>/', shoulderstretch_L.StreamConsumer.as_asgi()),
    path('ws/detection/shoulderstretch_R/<str:code>/', shoulderstretch_R.StreamConsumer.as_asgi()),
    path('ws/detection/standingforwardbend/<str:code>/', standingforwardbend.StreamConsumer.as_asgi()),
    path('ws/detection/standingsidebend_L/<str:code>/', standingsidebend_L.StreamConsumer.as_asgi()),
    path('ws/detection/standingsidebend_R/<str:code>/', standingsidebend_R.StreamConsumer.as_asgi()),
]
