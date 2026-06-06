from django.db import models


class PredictionHistory(models.Model):

    timestamp = models.DateTimeField(auto_now_add=True)

    avg_air_temp = models.FloatField()

    avg_global_rad = models.FloatField()

    avg_cell_rad = models.FloatField()

    avg_surface_temp = models.FloatField()

    avg_wind_speed = models.FloatField()

    predicted_energy = models.FloatField()

    model_used = models.CharField(max_length=20)

    def __str__(self):
        return f"{self.timestamp}"
    


class ForecastResult(models.Model):

    MODEL_CHOICES = [
        ("ANN", "ANN"),
        ("LSTM", "LSTM"),
        ("ENSEMBLE", "ENSEMBLE"),
    ]

    model_type = models.CharField(max_length=20, choices=MODEL_CHOICES)

    input_timestamp = models.DateTimeField()

    predicted_value = models.FloatField()

    actual_value = models.FloatField(null=True, blank=True)

    error = models.FloatField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.model_type} - {self.created_at}"