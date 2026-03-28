from django.http import HttpResponse


def index(request):
    return HttpResponse("Experiments myapp is working.")