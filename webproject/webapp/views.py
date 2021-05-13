from django.shortcuts import render

# Create your views here.
def homepage(request):
    return render(request, 'webapp/homepage.html')

def demopage(request):
    return render(request, 'webapp/demopage.html')

def resultspage(request):
    return render(request, 'webapp/resultspage.html')