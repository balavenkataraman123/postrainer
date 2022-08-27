from django.db import models
import uuid

# Create your models here.
class Item(models.Model): 
    id = models.UUIDField(
        primary_key = True, default = uuid.uuid4, editable = False
    )
    name = models.CharField(max_length=100, blank = True)
