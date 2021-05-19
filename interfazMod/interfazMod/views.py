from django.http import HttpResponse
from django.template import Template, Context
from django.shortcuts import render
from interfazMod import pruebaFuncion
from interfazMod import model




def loadIndex(request):


    #metodo = request.GET["metodo"]

    ctx = pruebaFuncion.funcionPrueba()
    #ctx["metodo"] = metodo
    return render(request, "index.html", ctx)


def buscar(request):


    #metodo = request.GET["metodo"]

    
    estaciones = request.GET["estaciones"]
    poblacion = request.GET["poblacion"]
    tipoCruce = request.GET["tipoCruce"]
    cruce = request.GET["cruce"]
    tipoMut = request.GET["tipoMut"]
    mutacion = request.GET["mutacion"]
    seleccion = request.GET["seleccion"]
    elite = request.GET["elite"]
    semilla = request.GET["semilla"]
    iteraciones = request.GET["iteraciones"]
    segundos = request.GET["segundos"]
    porcentaje = request.GET["porcentaje"]
    archivoIN = request.GET["archivoIN"]
    ctx = model.runModel("021MITCHELL", "max_min", tipoMut,seleccion, tipoCruce, "valid", poblacion, mutacion, elite, cruce, semilla, iteraciones, segundos, porcentaje, estaciones)
    ctx["cruce"] = cruce
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

