from django import forms


class PredictionForm(forms.Form):

    avg_air_temp = forms.FloatField()

    avg_global_rad = forms.FloatField()

    avg_cell_rad = forms.FloatField()

    avg_surface_temp = forms.FloatField()

    avg_wind_speed = forms.FloatField()
    
    file = forms.FileField()