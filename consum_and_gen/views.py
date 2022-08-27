from django.shortcuts import render
from . import app_consum
from .models import gas_supply
# Create your views here.
def index(request):
    supply = gas_supply.objects.all()
    # consumption = generation_and_consumption.objects.all()
    return render(request,'consum_and_gen/index.html', {'supply':supply})