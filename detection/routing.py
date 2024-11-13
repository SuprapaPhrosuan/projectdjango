from django.urls import path


from . import elbowplank, quadricepstretch_R
from . import shoulderstretch_L
from . import overheadtricepstretch_L
from . import neckstretch_R
from . import standingforwardbend

websocket_urlpatterns = [
    path('ws/detection/neckstretch_R/<str:code>/', neckstretch_R.StreamConsumer.as_asgi()),
    path('ws/detection/overheadtricepstretch_L/<str:code>/', overheadtricepstretch_L.StreamConsumer.as_asgi()),
    path('ws/detection/shoulderstretch_L/<str:code>/', shoulderstretch_L.StreamConsumer.as_asgi()),
    path('ws/detection/quadricepstretch_R/<str:code>/', quadricepstretch_R.StreamConsumer.as_asgi()),
    path('ws/detection/elbowplank/<str:code>/', elbowplank.StreamConsumer.as_asgi()),
]
