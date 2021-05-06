from django.http import HttpResponse
from django.template import Template, Context
from django.shortcuts import render
from interfazMod import pruebaFuncion


def loadIndex(request):


    #metodo = request.GET["metodo"]

    ctx = pruebaFuncion.funcionPrueba()
    #ctx["metodo"] = metodo
    return render(request, "index.html", ctx)


def buscar(request):


    #metodo = request.GET["metodo"]

    ctx = pruebaFuncion.funcionPrueba()
    ctx["cruce"] = request.GET["cruce"]
    ctx["mutacion"] = request.GET["mutacion"]
    ctx["estaciones"] = request.GET["estaciones"]
    ctx["metodo"] = request.GET["metodo"]
    return render(request, "index.html", ctx)



def loadGraph(request):


    #metodo = request.GET["metodo"]

    ctx = pruebaFuncion.funcionPrueba()
    #ctx["metodo"] = metodo
    return render(request, "prueba.html", ctx)

def loadPrueba2(request):


    ctx = pruebaFuncion.funcionPrueba()
    #ctx["metodo"] = metodo
    return render(request, "prueba2.html", ctx)

