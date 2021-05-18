import numpy as np
import matplotlib.pyplot as plt
import os
# del sys.modules["blp_ga"]
from interfazMod.blp_ga import BlpGa as Ga
from datetime import datetime

def runModel (data_set, ev_func_type, mut_type,select_type, cross_type, init_pop, pop_size, mut_prob, elit_ratio, cross_prob, seed, stop_cond,stations):

    

    DATA_SET = "148BARTHOLD"
    EV_FUNC_TYPE = "max_min"  # var_max / var / max / max_min
    MUT_TYPE = "random"  # inc_dec / random
    SELECT_TYPE = "tournament"  # tournament / roulette
    CROSS_TYPE = "two_points"  # one_point / two_points / uniform
    INIT_POP = "valid"  # random / valid
    POP_SIZE = 200
    MUT_PROB = 0.025
    ELIT_RATIO = 0.3
    CROSS_PROB = 0.7
    SEED = np.random.randint(1000)
    STOP_COND = {"max_iter": 10000, "max_secs": 5*60, "ideal_diff": 0.05}

    
    #module_dir = os.path.dirname(__file__)  # get current directory
    #file_path = os.path.join(module_dir, 'baz.txt')

    pwd = os.path.dirname(__file__)
    path = pwd + '/'+data_set + '.IN2'
    
    with open(f"{path}") as file:
        lines = file.readlines()
    N = int(lines.pop(0))
    times = []
    for i in range(N):
        times.append(int(lines.pop(0)))

    R = np.zeros((N, N))
    count = 0

    while lines:
        [i, j] = lines.pop(0).split(",")
        i, j = int(i), int(j)
        if i == -1:
            break
        R[i - 1, j - 1] = 1
        count += 1

    print(f"Problema {data_set} con {N} tareas y {count} restricciones")
    k = 3

    init = datetime.now()
    model = Ga(n_tasks=N, n_stations=k, times=times, restrictions=R, ev_func_type=EV_FUNC_TYPE, mut_type=MUT_TYPE,
           select_type=SELECT_TYPE, cross_type=CROSS_TYPE, pop_size=POP_SIZE, init_pop=INIT_POP, mut_prob=MUT_PROB,
           elit_ratio=ELIT_RATIO, cross_prob=CROSS_PROB, stop_cond=STOP_COND, seed=SEED)
    
    '''
    init = datetime.now()
    model = Ga(n_tasks=N, n_stations=k, times=times, restrictions=R, ev_func_type=ev_func_type, mut_type=mut_type,
            select_type=select_type, cross_type=cross_type, pop_size=pop_size, init_pop=init_pop, mut_prob=mut_prob,
            elit_ratio=elit_ratio, cross_prob=cross_prob, stop_cond=stop_cond, seed=seed) '''

    model.run()

    run_time = datetime.now()-init

    '''
    best_ind = model.best_ind
    print(best_ind)
    print(model.station_times(best_ind))
    print(model.check_restrictions(best_ind))
    model.ev_func(best_ind)'''
    

    '''res = input("Guardar resultado? s/n: ")
    if res == "s" or res == "S":
        # guardar resultado
        comment = input("Comentario: ")
        now = datetime.now()
        name = f"output/{now.year}_{now.month if now.month > 9 else '0' + str(now.month)}_" \
            f"{now.day if now.day > 9 else '0' + str(now.day)}_{now.hour if now.hour > 9 else '0' + str(now.hour)}_" \
            f"{now.minute if now.minute > 9 else '0' + str(now.minute)}_" \
            f"{now.second if now.second > 9 else '0' + str(now.second)}_{DATA_SET}"
        with open(f"{name}.txt", "w") as file:
            file.write(f"DATA_SET = {DATA_SET}\n")
            file.write(f"N_STATIONS = {k}\n")
            file.write(f"MUT_TYPE = {MUT_TYPE}\n")
            file.write(f"EV_FUNC_TYPE = {EV_FUNC_TYPE}\n")
            file.write(f"SELECT_TYPE = {SELECT_TYPE}\n")
            file.write(f"CROSS_TYPE = {CROSS_TYPE}\n")
            file.write(f"INIT_POP = {INIT_POP}\n")
            file.write(f"POP_SIZE = {POP_SIZE}\n")
            file.write(f"MUT_PROB = {MUT_PROB}\n")
            file.write(f"ELIT_RATIO = {ELIT_RATIO}\n")
            file.write(f"CROSS_PROB = {CROSS_PROB}\n")
            file.write(f"SEED = {SEED}\n")
            file.write(f"STOPCOND = max_iter: {STOP_COND['max_iter']}, max_secs: {STOP_COND['max_secs']}, "
                    f"ideal_diff: {STOP_COND['ideal_diff']}\n")
            file.write(f"best ind: {best_ind}\n")
            file.write(f"station times: {model.station_times(best_ind)}\n")
            file.write(f"tiempo: {int(run_time.seconds / 60)} min {run_time.seconds % 60} secs")
            file.write(f"\n{comment}\n")
        plt.savefig(f"{name}.png") '''

    # plt.clf()
    # plt.subplot(2, 1, 1)
    # plt.plot(model.best_ev_func, color="blue")
    # plt.legend(labels=['best_ev_func'])
    # plt.subplot(2, 1, 2)
    # plt.plot(model.best_max_time, color="red")
    # plt.legend(labels=['best_max_time'])
    # plt.draw()

    valoresTor = [100, 97, 94, 90, 86, 83, 83, 82, 50, 40, 35, 34, 35, 35, 34, 34, 33]
    valoresRul = [100, 80, 60, 45, 40, 35, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33]
    valoresLin = [100, 80, 50, 40, 40, 39, 38, 36, 35, 34, 34, 33, 33, 33, 33, 33, 33]

    tareaEstacion = ["1112212222322333333444"]

    return {"valoresTorn": valoresTor, "valoresRul":valoresRul ,"valoresLin":valoresLin, "grafica": "true", "solution": "La conexion es buena", "tiempo": run_time}