from django.shortcuts import render

def exercise_view(request, code):
    templates = {
        'C01': 'detection/cobrastretch.html',
        'C02': 'detection/childpose.html',
        'C03': 'detection/standingforwardbend.html',
        'C04': 'detection/overheadtricepstretch_L.html',
        'C05': 'detection/overheadtricepstretch_R.html',
        'C06': 'detection/quadricepstretch_L.html',
        'C07': 'detection/quadricepstretch_R.html',
        'C08': 'detection/shoulderstretch_R.html',
        'C09': 'detection/shoulderstretch_L.html',
        'E01': 'detection/elbowplank.html',
        'W04': 'detection/neckstretch_R.html',
        'W05': 'detection/neckstretch_L.html',
        'W06': 'detection/reachingup.html',
        'W07': 'detection/reachingdown.html',
        'W08': 'detection/standingsidebend_L.html',
        'W09': 'detection/standingsidebend_R.html', 
    }

    template = templates.get(code)
    if template:
        return render(request, template, {'code': code})
    else:
        return render(request, 'detection/error.html', {'message': 'Invalid code'})
