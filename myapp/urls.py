from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("info/", views.info, name="info"),
    path("data/", views.data, name="data"),
    path("predictions/", views.predictions, name="predictions"),
    path("hyperparameter_error/", views.hyperparameter_error, name="hyperparameter_error"),
    path("best_hyperparameters/", views.best_hyperparameters, name="best_hyperparameters"),
]