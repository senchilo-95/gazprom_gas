from django.shortcuts import render
from . import plotly_app
from .models import gas_futures
# Create your views here.
def index(request):
    prices = gas_futures.objects.all()
    return render(request,'dash_RSV/index.html', {'prices':prices})



