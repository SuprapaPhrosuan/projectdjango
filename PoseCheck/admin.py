import csv
from django.http import HttpResponse
from django.contrib import admin
from .models import Category, Exercise

class PoseCheckAdmin(admin.ModelAdmin):
    list_display = ['code','show_image','title','category','video','time','description']
    list_filter = ['category']
    search_fields = ['code','title']
    prepopulated_fields = {'slug': ['title']}
    ordering = ['code','title']

    actions = ['export_to_csv']

    def export_to_csv(self, request, queryset):
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename=exercises.csv'
        
        writer = csv.writer(response)
        writer.writerow(['Code', 'Title'])

        for exercise in queryset:
            writer.writerow([exercise.code, exercise.title])

        return response

    export_to_csv.short_description = "Export selected exercises to CSV"

admin.site.register(Category)
admin.site.register(Exercise, PoseCheckAdmin)
