from django.contrib import admin
from .models import ForecastResult


@admin.register(ForecastResult)
class ForecastResultAdmin(admin.ModelAdmin):
    list_display = (
        "model_type",
        "predicted_value",
        "actual_value",
        "error",
        "created_at"
    )

    list_filter = (
        "model_type",
        "created_at"
    )

    search_fields = (
        "model_type",
    )