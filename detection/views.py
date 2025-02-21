from django.shortcuts import render

def cobrastretch(request):
    return render(request, 'detection/cobrastretch.html', {'code': 'C01'})

def childpose(request):
    return render(request, 'detection/childpose.html', {'code': 'C02'})

def standingforwardbend(request):
    return render(request, 'detection/standingforwardbend.html', {'code': 'C03'})

def overheadtricepstretch_L(request):
    return render(request, 'detection/overheadtricepstretch_L.html', {'code': 'C04'})

def overheadtricepstretch_R(request):
    return render(request, 'detection/overheadtricepstretch_R.html', {'code': 'C05'})

def quadricepstretch_L(request):
    return render(request, 'detection/quadricepstretch_L.html', {'code': 'C06'})

def quadricepstretch_R(request):
    return render(request, 'detection/quadricepstretch_R.html', {'code': 'C07'})

def shoulderstretch_R(request):
    return render(request, 'detection/shoulderstretch_R.html', {'code': 'C08'})

def shoulderstretch_L(request):
    return render(request, 'detection/shoulderstretch_L.html', {'code': 'C09'})

def elbowplank(request):
    return render(request, 'detection/elbowplank.html', {'code': 'E01'})

def neckstretch_R(request):
    return render(request, 'detection/neckstretch_R.html', {'code': 'W04'})

def neckstretch_L(request):
    return render(request, 'detection/neckstretch_L.html', {'code': 'W05'})

def reachingup(request):
    return render(request, 'detection/reachingup.html', {'code': 'W06'})

def reachingdown(request):
    return render(request, 'detection/reachingdown.html', {'code': 'W07'})

def standingsidebend_R(request):
    return render(request, 'detection/standingsidebend_R.html', {'code': 'W09'})

def standingsidebend_L(request):
    return render(request, 'detection/standingsidebend_R.html', {'code': 'W08'})
