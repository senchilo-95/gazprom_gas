from django.shortcuts import render
from . import app_consum
from .models import gas_supply,supply_forecast
# Create your views here.
def index(request):
    supply = gas_supply.objects.all()
    forecast = supply_forecast.objects.all()
    # consumption = generation_and_consumption.objects.all()
    return render(request,'supply/index.html', {'supply':supply,'forecast':forecast})