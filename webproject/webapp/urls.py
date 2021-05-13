from django.urls import path
from . import views

urlpatterns = [
    path('', views.homepage, name='homepage'),
    path('demo/', views.demopage, name='demopage'),
    path('demo/results', views.resultspage, name='resultspage'),
]