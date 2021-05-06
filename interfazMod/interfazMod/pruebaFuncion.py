def funcionPrueba():

    valoresTor = [100, 97, 94, 90, 86, 83, 83, 82, 50, 40, 35, 34, 35, 35, 34, 34, 33]
    valoresRul = [100, 80, 60, 45, 40, 35, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33]
    valoresLin = [100, 80, 50, 40, 40, 39, 38, 36, 35, 34, 34, 33, 33, 33, 33, 33, 33]

    tareaEstacion = ["1112212222322333333444"]

    return {"valoresTorn": valoresTor, "valoresRul":valoresRul ,"valoresLin":valoresLin, "grafica": "true", "solution": parseStations(tareaEstacion)}


def parseStations(solution):

    stations = []

    for i in range(len(solution)):

        stations.append(solution[i])

    return stations 