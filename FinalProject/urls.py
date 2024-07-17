from django.contrib import admin
from django.urls import path, re_path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include(('myapp.urls', 'myapp'), namespace='myapp')),
    path('detection/', include(('detection.urls', 'detection'), namespace='detection')),
    path('PoseCheck/', include(('PoseCheck.urls', 'PoseCheck'), namespace='PoseCheck')),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
