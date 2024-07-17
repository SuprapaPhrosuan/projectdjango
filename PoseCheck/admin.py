from django.contrib import admin
from .models import Category, Exercise

class PoseCheckAdmin(admin.ModelAdmin):
    list_display = ['code','show_image','title','category','video','time','description']
    list_filter = ['category']
    search_fields = ['code','title']
    prepopulated_fields = {'slug': ['title']}

admin.site.register(Category)
admin.site.register(Exercise, PoseCheckAdmin)