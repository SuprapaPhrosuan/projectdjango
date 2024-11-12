from django.shortcuts import render

def standingforwardbend(request):
    return render(request, 'detection/standingforwardbend.html', {'code': 'C03'})

def neckstretch_R(request):
    return render(request, 'detection/neckstretch_R.html', {'code': 'W04'})

def neckstretch_R(request):
    return render(request, 'detection/overheadtricepstretch_L.html', {'code': 'C04'})