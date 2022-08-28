from django.contrib import admin

# Register your models here.
from .models import gas_futures

admin.site.register(gas_futures)
