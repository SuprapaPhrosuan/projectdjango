import json
from django.http import HttpResponse
from django.contrib import admin
from .models import Category, Exercise

class PoseCheckAdmin(admin.ModelAdmin):
    list_display = ['code','show_image','title','category','video','time','description']
    list_filter = ['category']
    search_fields = ['code','title']
    prepopulated_fields = {'slug': ['title']}
    ordering = ['code','title']

    actions = ['export_to_json']

    def export_to_json(self, request, queryset):
        response = HttpResponse(
            content_type='application/json',
            charset='utf-8'
        )
        response['Content-Disposition'] = 'attachment; filename=exercises.json'

        exercises_data = []
        for exercise in queryset:
            exercise_data = {
                'code': exercise.code,
                'slug': exercise.slug,
                'title': exercise.title,
                'category': exercise.category.name if exercise.category else '',
                'image_url': exercise.image.url if exercise.image else '',
                'video': exercise.video,
                'time': exercise.time,
                'description': exercise.description,
                'created_at': exercise.create.strftime('%Y-%m-%d %H:%M:%S'),
                'updated_at': exercise.update.strftime('%Y-%m-%d %H:%M:%S')
            }
            exercises_data.append(exercise_data)

        json.dump(exercises_data, response, ensure_ascii=False, indent=4)
        return response

    export_to_json.short_description = "Export selected exercises to JSON"

admin.site.register(Category)
admin.site.register(Exercise, PoseCheckAdmin)
