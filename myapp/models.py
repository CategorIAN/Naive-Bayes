from django.db import models


class Prediction(models.Model):
    dataset_name = models.CharField(max_length=200)
    bin_size = models.IntegerField()
    alpha = models.FloatField()

    test_set_index = models.IntegerField()
    row_index = models.IntegerField()

    predicted = models.CharField(max_length=10)
    actual = models.CharField(max_length=10)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["dataset_name", "bin_size", "alpha", "test_set_index", "row_index"],
                name="unique_prediction"
            )
        ]


