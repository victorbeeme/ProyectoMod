import os
import numpy as np

def parser(content):

    print("TIPO")
    print(type(content))


    var = content.decode("utf-8")
    separated = var.split('\r\n')

    for str in separated[:-1]:
        str = str + "\n"

    return separated
