from django.db import models

class UserAuth(models.Model):
    subsr = models.CharField(max_length=10)
    use_ip = models.IntegerField()