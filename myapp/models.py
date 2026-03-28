from django.db import models

class Prediction(models.Model):
    bin_size = models.IntegerField()
    alpha = models.FloatField()

    test_set_index = models.IntegerField()
    row_index = models.IntegerField()

    predicted = models.FloatField()
    actual = models.FloatField()

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["bin_size", "alpha", "test_set_index", "row_index"],
                name="unique_prediction"
            )
        ]
