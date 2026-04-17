from django.shortcuts import render
from django.conf import settings
import pandas as pd
from pathlib import Path
from myapp.models import Prediction
import markdown

def highlight_class(col):
    if col.name == "Class":
        return ["background-color: #fff3cd; font-weight: bold;" for _ in col]
    return ["" for _ in col]


def about_model(request):
    current_dataset = request.GET.get("dataset", "soybean")
    file_path = Path(settings.BASE_DIR) / "myapp" / "about_model.md"
    text = file_path.read_text(encoding="utf-8")
    html = markdown.markdown(text, extensions=["fenced_code", "codehilite"])
    context = {
        "title": "About The Model",
        "content_html": html,
        "current_section": "about_model",
        "current_dataset": current_dataset,
    }

    return render(request, "myapp/about_model.html", context)


def about_tuning(request):
    current_dataset = request.GET.get("dataset", "soybean")
    file_path = Path(settings.BASE_DIR) / "myapp" / "about_tuning.md"
    text = file_path.read_text(encoding="utf-8")
    html = markdown.markdown(text, extensions=["fenced_code", "codehilite"])
    context = {
        "title": "About The Tuning",
        "content_html": html,
        "current_section": "about_tuning",
        "current_dataset": current_dataset,
    }

    return render(request, "myapp/about_model.html", context)


def info(request):
    current_dataset = request.GET.get("dataset", "soybean")
    if current_dataset == "":
        text = ""
    else:
        file_path = Path(settings.BASE_DIR) / "documentation" / f"{current_dataset}.names"
        text = file_path.read_text(encoding="utf-8")
    context = {
        "text": text,
        "current_page": "info",
        "current_dataset": current_dataset,
        "current_section": "about_data",
        "sidebar_mode": "data_about",
    }
    return render(request, "myapp/info.html", context)


def data(request):
    print("Line 1")
    current_dataset = request.GET.get("dataset", "soybean")
    print("Line 2")
    if current_dataset is None:
        print("Line 3")
        title = "Choose A Dataset"
        table_html = ""
    else:
        print("Line 4")
        title = current_dataset.capitalize()
        print("Line 5")
        file_path = Path(settings.BASE_DIR) / "data_processed" / f"{current_dataset}.csv"
        print("Line 6")
        df = pd.read_csv(file_path)
        print("Line 7")
        print("Line 7a")
        print(df.columns.tolist())
        try:
            styled = (
                df.style
                .apply(highlight_class, axis=0)
                .set_table_attributes('class="dataframe-table"')
            )
            print("Line 8")
        except Exception as e:
            print("STYLER ERROR:", repr(e))
            raise
        print("Line 8")
        table_html = styled.to_html()
        print("Line 9")
    context = {
        "title": title,
        "table_html": table_html,
        "current_page": "data",
        "current_dataset": current_dataset,
        "current_section": "tuning",
        "sidebar_mode": "tuning",
    }
    print("Line 10")
    return render(request, "myapp/data.html", context)


def predictions(request):
    current_dataset = request.GET.get("dataset", "soybean")
    if current_dataset is None:
        title = "Choose A Dataset"
        rows = ""
    else:
        title = current_dataset.capitalize()
        rows = (
            Prediction.objects
            .filter(dataset_name=current_dataset)
            .order_by("bin_size", "alpha", "test_set_index", "row_index")
            )
    context = {
        "title": title,
        "rows": rows,
        "current_page": "predictions",
        "current_dataset": current_dataset,
        "current_section": "tuning",
        "sidebar_mode": "tuning",
    }

    return render(request, "myapp/predictions.html", context)


def hyperparameter_error(request):
    current_dataset = request.GET.get("dataset", "soybean")
    if current_dataset is None:
        title = "Choose A Dataset"
        table_html = ""
    else:
        title = current_dataset.capitalize()
        df = pd.DataFrame.from_records(
            Prediction.objects
                .filter(dataset_name=current_dataset)
                .values(
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

        table_html = summary.to_html(
            index=False,
            classes="dataframe-table",
            border=0,
            float_format=lambda x: f"{x:.4f}"
        )
    context = {
        "title": title,
        "table_html": table_html,
        "current_page": "hyperparameter_error",
        "current_dataset": current_dataset,
        "current_section": "tuning",
        "sidebar_mode": "tuning",
    }

    return render(request, "myapp/hyperparameter_error.html", context)


def best_hyperparameters(request):
    current_dataset = request.GET.get("dataset", "soybean")
    if current_dataset is None:
        title = "Choose A Dataset"
        results = ""
    else:
        title = current_dataset.capitalize()
        df = pd.DataFrame.from_records(
            Prediction.objects
                .filter(dataset_name=current_dataset)
                .values(
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
        results = f"""
        <h3> Best Bin Size is {int(bin_size)} </h3>
        <h3> Best Alpha is {alpha} </h3>
        """
    context = {
                "title": title,
                "results": results,
                "current_page": "best_hyperparameters",
                "current_dataset": current_dataset,
                "current_section": "tuning",
                "sidebar_mode": "tuning",
    }

    return render(request, "myapp/best_hyperparameters.html", context)



