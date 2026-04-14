from django.urls import path
from . import views

urlpatterns = [
    path("", views.about_model, name="about_model"),
    path("about_tuning/", views.about_tuning, name="about_tuning"),
    path("info/", views.info, name="info"),
    path("data/", views.data, name="data"),
    path("predictions/", views.predictions, name="predictions"),
    path("hyperparameter_error/", views.hyperparameter_error, name="hyperparameter_error"),
    path("best_hyperparameters/", views.best_hyperparameters, name="best_hyperparameters"),
]
