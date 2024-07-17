from django.shortcuts import render, get_object_or_404
from .models import Category, Exercise


def index(request):
    category = Category.objects.all()
    exercise = Exercise.objects.all()

    categ_id = request.GET.get('categoryid')
    if categ_id:
        exercise = exercise.filter(category_id = categ_id)

    return render(request, 'posecheck/index.html', {
        'category':category,
        'exercise':exercise,
        'categ_id':categ_id,
    })

def description(request, slug):
    exercise = get_object_or_404(Exercise, slug=slug)
    return render(request, 'posecheck/description.html', {
        'exercise': exercise,
    })