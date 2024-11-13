from django.shortcuts import render

def standingforwardbend(request):
    return render(request, 'detection/standingforwardbend.html', {'code': 'C03'})

def neckstretch_R(request):
    return render(request, 'detection/neckstretch_R.html', {'code': 'W04'})

def quadricepstretch_R(request):
    return render(request, 'detection/quadricepstretch_R.html', {'code': 'C07'})

def shoulderstretch_L(request):
    return render(request, 'detection/shoulderstretch_L.html', {'code': 'C09'})

def elbowplank(request):
    return render(request, 'detection/elbowplank.html', {'code': 'E01'})

def overheadtricepstretch_L(request):
    return render(request, 'detection/overheadtricepstretch_L.html', {'code': 'C04'})