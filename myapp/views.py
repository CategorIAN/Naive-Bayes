from django.http import HttpResponse
from django.shortcuts import render
from django.conf import settings
import os
import pandas as pd
from pathlib import Path


def home(request):
    return render(request, "myapp/home.html")


def data(request):
    file_path = Path(settings.BASE_DIR) / "data_processed" / "soybean.csv"
    df = pd.read_csv(file_path)
    table_html = df.to_html(
        index=False,
        classes="dataframe-table",
        border=0
    )
    context = {"title": "Data", "table_html": table_html}
    return render(request, "myapp/results.html", context)



