from django.http import HttpResponse
from django.template import Template, Context
from django.shortcuts import render
from interfazMod import pruebaFuncion
from interfazMod import model
import time
import numpy as np



def loadIndex(request):


    #metodo = request.GET["metodo"]

    ctx = pruebaFuncion.funcionPrueba()
    ctx["num"] = np.random.randint(1000)
    #ctx["metodo"] = metodo
    return render(request, "index.html", ctx)


def buscar(request):

    #metodo = request.GET["metodo"]

    estaciones = request.POST["estaciones"]
    poblacion = request.POST["poblacion"]
    tipoCruce = request.POST["tipoCruce"]
    cruce = request.POST["cruce"]
    tipoMut = request.POST["tipoMut"]
    mutacion = request.POST["mutacion"]
    seleccion = request.POST["seleccion"]
    elite = request.POST["elite"]
    semilla = request.POST["semilla"]
    file = request.FILES["archivoIN"]
    content = file.read()
    funcEv = request.POST["tipoFuncionEv"]

    iteraciones = request.POST["iteraciones"]
    segundos = request.POST["segundos"]
    porcentaje = request.POST["porcentaje"]

    usaIter = request.POST["usaElmaxIterInput"]
    usaTiempo = request.POST["usaElmaxSegInput"]
    usaDiff = request.POST["usaEldifPorcentual"]

    usaIterNum = int(usaIter)
    usaTiempoNum = int(usaTiempo)
    usaDiffNum = int(usaDiff)

    if usaIterNum == 0:
        iteraciones = 100000
        print("AAAAAAAAAAAAAAAAAAAA")
        print("NOUSOITER")

    if usaTiempoNum == 0:
        segundos = 100000
        print("NOUSOSEGS")

    if usaDiffNum == 0:
        porcentaje = 0
        print("NOUSOPORCENTAJE")

    ctx = model.runModel(funcEv, tipoMut,seleccion, tipoCruce, "valid", poblacion, mutacion, elite, cruce, semilla, iteraciones, segundos, porcentaje, estaciones,content)
    ctx["cruce"] = cruce
    ctx["content"] = content
    ctx["num"] = semilla
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

def upload(request):
    content = ""
    if request.method == 'POST':
        file = request.FILES['document']
        content = file.read()

    return render(request, "archivo.html", {"content":content})