from django.db import models
from django.utils.html import format_html

class Category(models.Model):
    name = models.CharField(max_length=100, default="warm up")

    def __str__(self):
        return self.name

class Exercise(models.Model):
    code = models.CharField(max_length=10, unique=True)
    slug = models.SlugField(max_length=200, unique=True)
    title = models.CharField(max_length=200, unique=True)
    category = models.ForeignKey(Category, null=True, blank=True, on_delete=models.CASCADE)
    image = models.FileField(upload_to='upload', null=True, blank=True)
    video = models.TextField(null=True, blank=True)
    time = models.CharField(max_length=10)
    description = models.TextField(null=True, blank=True)
    create = models.DateTimeField(auto_now_add=True)
    update = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.title
    
    def show_image(self):
        if self.image:
            return format_html('<img src="' + self.image.url + '" height="50px">')
        else:
            return ''
    show_image.allow_tags = True
    
    
