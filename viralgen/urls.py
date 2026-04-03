from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from dip import views as dip_views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', dip_views.landing_view, name='landing'),
    path('auth/', include('accounts.urls')),
    path('dashboard/', dip_views.dashboard_view, name='dashboard'),
    path('process/', dip_views.process_view, name='process'),
    path('process/api/', dip_views.process_image_api, name='process_api'),
    path('bulk/', dip_views.bulk_view, name='bulk'),
    path('bulk/api/', dip_views.bulk_process_api, name='bulk_api'),
    path('analytics/', dip_views.analytics_view, name='analytics'),
    path('history/', dip_views.history_view, name='history'),
    path('download/<str:filename>/', dip_views.download_view, name='download'),
    path('ab-compare/', dip_views.ab_compare_view, name='ab_compare'),
    path('caption-histogram/', dip_views.caption_histogram_view, name='caption_histogram'),
    path('caption/api/', dip_views.generate_caption_api, name='caption_api'),
    path('histogram/api/', dip_views.generate_histogram_api, name='histogram_api'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
