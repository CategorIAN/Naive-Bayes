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


def hyperparameter_error(request):
    df = pd.DataFrame.from_records(
        Prediction.objects.values(
            "bin_size",
            "alpha",
            "actual",
            "predicted",
        )
    )

    if df.empty:
        table_html = "<p>No prediction rows found.</p>"
    else:
        df["error"] = (df["actual"] != df["predicted"]).astype(float)

        summary = (
            df.groupby(["bin_size", "alpha"], as_index=False)["error"]
              .mean()
              .rename(columns={"error": "avg_error"})
              .sort_values(["avg_error", "bin_size", "alpha"])
        )

        table_html = summary.to_html(
            index=False,
            classes="dataframe-table",
            border=0,
            float_format=lambda x: f"{x:.4f}"
        )
    context = {"title": "Hyperparameter Error", "table_html": table_html}

    return render(request, "myapp/hyperparameter_error.html", context)


def best_hyperparameters(request):
    df = pd.DataFrame.from_records(
        Prediction.objects.values(
            "bin_size",
            "alpha",
            "actual",
            "predicted",
        )
    )
    df["error"] = (df["actual"] != df["predicted"]).astype(float)
    summary = (
        df.groupby(["bin_size", "alpha"], as_index=False)["error"]
          .mean()
          .rename(columns={"error": "avg_error"})
          .sort_values(["avg_error", "bin_size", "alpha"])
    )
    bin_size, alpha = summary[["bin_size", "alpha"]].iloc[0]
    context = {"title": "Best Hyperparameters", "bin_size": int(bin_size), "alpha": alpha}

    return render(request, "myapp/best_hyperparameters.html", context)



