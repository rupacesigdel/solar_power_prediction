from django.http import HttpResponse
from django.shortcuts import render
from .forms import PredictionForm
from .services.predictor import predict_energy
from .services.dashboard_service import get_dashboard_stats
from django.http import JsonResponse
import pandas as pd
import numpy as np
from .services.predictor import predict_ann, predict_lstm
from .services.dashboard_service import get_dashboard_stats
from .services.storage import save_prediction
from datetime import datetime
from .models import ForecastResult
import csv
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

def base(request):
    return render(request, "base.html")


def predict_view(request):

    prediction = None

    if request.method == "POST":

        form = PredictionForm(request.POST)

        if form.is_valid():

            data = list(form.cleaned_data.values())

            prediction = predict_energy("ann", data)

    else:
        form = PredictionForm()

    return render(request, "predict.html", {
        "form": form,
        "prediction": prediction
    })


def dashboard(request):

    stats = get_dashboard_stats()

    return render(request, "dashboard.html", stats)

def chart_data(request):

    df = pd.read_csv("data/processed/super_master_dataset.csv")

    # take last 100 samples for visualization
    df = df.tail(100)

    actual = df["ITS_Energy (kWh)"].tolist()

    features = df.drop(columns=["ITS_Energy (kWh)", "timestamp"], errors="ignore")

    # ANN predictions
    ann_pred = [
        predict_ann(row.tolist()) for _, row in features.iterrows()
    ]

    # LSTM (simplified example)
    lstm_pred = ann_pred  # replace with sequence logic later

    return JsonResponse({
        "actual": actual,
        "ann": ann_pred,
        "lstm": lstm_pred
    })




def future_forecast(request):

    df = pd.read_csv("data/processed/super_master_dataset.csv")

    feature_cols = [
        col for col in df.columns
        if col not in ["ITS_Energy (kWh)", "timestamp"]
    ]

    last_sequence = df[feature_cols].tail(24).values

    # store predictions
    lstm_preds = []
    ann_preds = []

    current_seq_lstm = last_sequence.copy()

    for i in range(24):

        lstm_pred = predict_lstm(current_seq_lstm)
        ann_pred = predict_ann(current_seq_lstm[-1])

        lstm_preds.append(lstm_pred)
        ann_preds.append(ann_pred)

        # SAVE TO DB
        save_prediction("LSTM", lstm_pred)
        save_prediction("ANN", ann_pred)

        current_seq_lstm = np.roll(current_seq_lstm, -1, axis=0)
        current_seq_lstm[-1, -1] = lstm_pred

    # -------------------------------
    # CONFIDENCE INTERVAL (simple)
    # -------------------------------
    error_margin = 0.08  # ~8% (you can replace with real MAPE)

    def add_confidence(preds):
        lower = [p * (1 - error_margin) for p in preds]
        upper = [p * (1 + error_margin) for p in preds]
        return lower, upper

    lstm_low, lstm_up = add_confidence(lstm_preds)
    ann_low, ann_up = add_confidence(ann_preds)

    return JsonResponse({
        "lstm": lstm_preds,
        "ann": ann_preds,

        "lstm_lower": lstm_low,
        "lstm_upper": lstm_up,

        "ann_lower": ann_low,
        "ann_upper": ann_up
    })


def model_metrics(request):

    ann = ForecastResult.objects.filter(model_type="ANN").exclude(actual_value=None)
    lstm = ForecastResult.objects.filter(model_type="LSTM").exclude(actual_value=None)

    def compute_metrics(queryset):

        actual = np.array([x.actual_value for x in queryset])
        pred = np.array([x.predicted_value for x in queryset])

        mae = np.mean(np.abs(actual - pred))
        rmse = np.sqrt(np.mean((actual - pred) ** 2))
        r2 = 1 - (np.sum((actual - pred) ** 2) / np.sum((actual - np.mean(actual)) ** 2))

        return {
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2)
        }

    return JsonResponse({
        "ANN": compute_metrics(ann),
        "LSTM": compute_metrics(lstm)
    })

def prediction_history(request):

    data = ForecastResult.objects.order_by("-created_at")[:100]

    return JsonResponse({
        "timestamps": [d.created_at.strftime("%H:%M") for d in data],
        "values": [d.predicted_value for d in data],
        "models": [d.model_type for d in data]
    })


def export_csv(request):

    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="forecast_results.csv"'

    writer = csv.writer(response)

    # Header
    writer.writerow([
        "Model",
        "Timestamp",
        "Predicted Value",
        "Actual Value",
        "Error"
    ])

    # Data
    results = ForecastResult.objects.all().order_by("-created_at")

    for r in results:
        writer.writerow([
            r.model_type,
            r.created_at,
            r.predicted_value,
            r.actual_value,
            r.error
        ])

    return response

def export_pdf(request):

    response = HttpResponse(content_type='application/pdf')
    response['Content-Disposition'] = 'attachment; filename="forecast_report.pdf"'

    doc = SimpleDocTemplate(response)

    styles = getSampleStyleSheet()
    elements = []

    # Title
    title = Paragraph("Solar Power Forecasting Report", styles['Title'])
    elements.append(title)
    elements.append(Spacer(1, 12))

    # Summary stats
    total = ForecastResult.objects.count()
    ann_count = ForecastResult.objects.filter(model_type="ANN").count()
    lstm_count = ForecastResult.objects.filter(model_type="LSTM").count()

    summary = Paragraph(
        f"""
        <b>Total Predictions:</b> {total}<br/>
        <b>ANN Predictions:</b> {ann_count}<br/>
        <b>LSTM Predictions:</b> {lstm_count}<br/>
        """,
        styles['Normal']
    )

    elements.append(summary)
    elements.append(Spacer(1, 12))

    # Table data
    data = [["Model", "Predicted", "Actual", "Error"]]

    for r in ForecastResult.objects.all().order_by("-created_at")[:20]:
        data.append([
            r.model_type,
            round(r.predicted_value, 3),
            r.actual_value if r.actual_value else "-",
            r.error if r.error else "-"
        ])

    table = Table(data)

    table.setStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.grey),
        ("TEXTCOLOR", (0,0), (-1,0), colors.whitesmoke),
        ("GRID", (0,0), (-1,-1), 0.5, colors.black),
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
    ])

    elements.append(table)

    doc.build(elements)

    return response

def comparison(request):
    return render(request, "comparison.html")