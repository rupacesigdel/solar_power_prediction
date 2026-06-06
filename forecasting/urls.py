from django.urls import path

from .views import *

urlpatterns = [
    path("", base, name="base"),
    path("predict/",predict_view,name="predict"),
    path("dashboard/",dashboard, name="dashboard"),
    path("chart-data/",chart_data, name="chart_data"),
    path("export/csv/",export_csv, name="export_csv"),
    path("export/pdf/",export_pdf, name="export_pdf"),
    path("comparison/",comparison, name="comparison"),  # ✅ ADD THIS

]