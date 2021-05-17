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
    ctx["estaciones"] = request.GET["estaciones"]
    ctx["poblacion"] = request.GET["poblacion"]
    ctx["tipoCruce"] = request.GET["tipoCruce"]
    ctx["cruce"] = request.GET["cruce"]
    ctx["tipoMut"] = request.GET["tipoMut"]
    ctx["mutacion"] = request.GET["mutacion"]
    ctx["seleccion"] = request.GET["seleccion"]
    ctx["elite"] = request.GET["elite"]
    ctx["semilla"] = request.GET["semilla"]
    ctx["iteraciones"] = request.GET["iteraciones"]
    ctx["segundos"] = request.GET["segundos"]
    ctx["porcentaje"] = request.GET["porcentaje"]
    ctx["archivoIN"] = request.GET["archivoIN"]
    return render(request, "cargado.html", ctx)



def loadGraph(request):


    #metodo = request.GET["metodo"]

    ctx = pruebaFuncion.funcionPrueba()
    #ctx["metodo"] = metodo
    return render(request, "prueba.html", ctx)

def loadPrueba2(request):


    ctx = pruebaFuncion.funcionPrueba()
    #ctx["metodo"] = metodo
    return render(request, "prueba2.html", ctx)

