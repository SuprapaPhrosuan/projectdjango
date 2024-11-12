from django.urls import path

from . import overheadtricepstretch_L
from . import neckstretch_R
from . import standingforwardbend

websocket_urlpatterns = [
    path('ws/detection/neckstretch_R/<str:code>/', neckstretch_R.StreamConsumer.as_asgi()),
    path('ws/detection/overheadtricepstretch_L/<str:code>/', overheadtricepstretch_L.StreamConsumer.as_asgi()),
    
]
