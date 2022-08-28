from django.contrib import admin

# Register your models here.
from .models import gas_supply,supply_forecast

admin.site.register(gas_supply)
admin.site.register(supply_forecast)