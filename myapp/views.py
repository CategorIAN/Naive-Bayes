from django.http import HttpResponse
from django.shortcuts import render
from django.conf import settings
import os
import pandas as pd
from pathlib import Path
from myapp.models import Prediction


def highlight_class(col):
    if col.name == "Class":
        return ["background-color: #fff3cd; font-weight: bold;" for _ in col]
    return ["" for _ in col]


def home(request):
    return render(request, "myapp/home.html")


def info(request):
    file_path = Path(settings.BASE_DIR) / "documentation" / "soybean.names"
    text = file_path.read_text(encoding="utf-8")
    context = {"text": text}
    return render(request, "myapp/info.html", context)


def data(request):
    file_path = Path(settings.BASE_DIR) / "data_processed" / "soybean.csv"
    df = pd.read_csv(file_path)
    styled = (
        df.style
        .apply(highlight_class, axis=0)
        .set_table_attributes('class="dataframe-table"')
    )
    table_html = styled.to_html()
    context = {"title": "Data", "table_html": table_html}
    return render(request, "myapp/data.html", context)


def predictions(request):
    rows = Prediction.objects.all().order_by(
        "bin_size", "alpha", "test_set_index", "row_index"
    )

    return render(request, "myapp/predictions.html", {
        "rows": rows
    })




