
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
plt.switch_backend('agg')


class BlpGa:

    def __init__(self, n_tasks, n_stations, times, restrictions,
                 ev_func_type="var",  # o "var_max" / "max" / "max_min"
                 mut_type="inc_dec",  # o "random"
                 select_type="tournament",  # o "roulette"
                 cross_type="uniform",
                 pop_size=100, init_pop="random",  # o "valid"
                 mut_prob=0.1, elit_ratio=0.01,
                 cross_prob=0.5,
                 stop_cond={"max_iter": 1000, "max_secs": None, "ideal_diff": None},
                 seed=123):

        self.n_tasks = n_tasks
        self.n_stations = n_stations
        self.mut_type = mut_type
        self.select_type = select_type
        self.cross_type = cross_type
        self.pop_size = pop_size
        self.init_pop = init_pop
        self.mut_prob = mut_prob
        self.elit_ratio = elit_ratio
        self.cross_prob = cross_prob
        self.times = times
        self.seed = seed

        # restricciones
        restr_aux = np.empty((n_tasks, n_tasks, n_tasks), dtype=int)
        restr_aux[0] = restrictions.copy()
        for i in range(n_tasks - 1):
            restr_aux[i + 1] = np.matmul(restr_aux[i], restrictions)
        for i in range(1, n_tasks):
            restrictions += restr_aux[i]
        self.restrictions = restrictions.astype(bool)

        # funcion de evaluacion
        if ev_func_type == "var":
            def eval_f(ind):
                count = self.check_restrictions(ind)
                if count != 0:
                    return sum(times) * (count + 100)
                st_times = self.station_times(ind)
                return np.var(list(st_times.values()))

        elif ev_func_type == "var_max":
            def eval_f(ind):
                count = self.check_restrictions(ind)
                if count != 0:
                    return sum(times) * (count + 100)
                st_times = self.station_times(ind)
                return np.var(list(st_times.values())) + max(st_times.values()) ** 2

        elif ev_func_type == "max":
            def eval_f(ind):
                count = self.check_restrictions(ind)
                if count != 0:
                    return sum(times) * (count + 100)
                st_times = self.station_times(ind)
                return max(st_times.values())

        elif ev_func_type == "max_min":
            def eval_f(ind):
                count = self.check_restrictions(ind)
                if count != 0:
                    return sum(times) * (count + 100)
                st_times = self.station_times(ind)
                return max(st_times.values()) - min(st_times.values())

        self.ev_func = eval_f

        # condiciones de parada
        self.max_iter = stop_cond["max_iter"]
        self.iter = 0

        self.max_secs = stop_cond["max_secs"]
        self.init_time = None

        self.stop_diff = stop_cond["ideal_diff"]

        # resultados
        self.best_ind = None
        self.best_ev_func = []
        self.best_max_time = []
        self.best_num_restric = []

    def diff_to_ideal(self, ind):
        if self.check_restrictions(ind) != 0:
            return 10000
        ideal_max_time = sum(self.times) / self.n_stations
        max_time = max(self.station_times(ind).values())
        return (max_time - ideal_max_time) / ideal_max_time

    def check_restrictions(self, ind):
        count = 0
        for i in range(self.n_tasks):
            for j in range(i + 1, self.n_tasks):
                if self.restrictions[i, j] and ind[i] > ind[j]:
                    count += 1
        return count

    def station_times(self, ind):
        st_times = dict([(st, 0) for st in range(1, self.n_stations + 1)])
        for task in range(0, self.n_tasks):
            st_times[ind[task]] += self.times[task]
        return st_times

    def check_stop(self, pop):

        if self.max_iter is not None:  # numero de interaciones
            self.iter += 1
            if self.iter >= self.max_iter:
                return True

        if self.max_secs is not None:
            if (datetime.now() - self.init_time).total_seconds() > self.max_secs:
                return True

        if self.stop_diff is not None:
            if self.diff_to_ideal(pop[0, :-1]) < self.stop_diff:
                return True

        return False

    def cross(self, parent1, parent2):

        child1 = np.empty(self.n_tasks)
        child2 = np.empty(self.n_tasks)
        genes = np.arange(self.n_tasks)

        if self.cross_type == "uniform":
            choice1 = np.random.choice(genes, size=int(self.n_tasks / 2), replace=False)
            choice2 = list(set(genes).difference(set(choice1)))

        elif self.cross_type == "one_point":
            ind = np.random.choice(genes[1:])  # punto que no sea primero
            choice1 = np.arange(0, ind)
            choice2 = np.arange(ind, self.n_tasks)

        elif self.cross_type == "two_points":
            ind = np.random.choice(genes[1:], size=2, replace=False)
            ind1, ind2 = np.min(ind), np.max(ind)
            choice1 = np.append(np.arange(0, ind1), (np.arange(ind2, self.n_tasks)))
            choice2 = np.arange(ind1, ind2)

        child1[choice1] = parent1[choice1]
        child1[choice2] = parent2[choice2]
        child2[choice1] = parent2[choice1]
        child2[choice2] = parent1[choice2]

        return child1, child2

    def mut(self, x):

        if self.mut_type == "inc_dec":

            for i in range(self.n_tasks):
                ran = np.random.random()
                if ran < self.mut_prob:
                    aux = np.random.uniform(0, 1)
                    if x[i] == 1:
                        x[i] = 2
                    elif x[i] == self.n_stations:
                        x[i] -= 1
                    elif aux >= 0.5:
                        x[i] += 1
                    else:
                        x[i] -= 1

        elif self.mut_type == "random":
            for i in range(self.n_tasks):
                ran = np.random.random()
                if ran < self.mut_prob:
                    x[i] = np.random.randint(1, self.n_stations + 1)

        return x

    def run(self):

        np.random.seed(self.seed)

        self.init_time = datetime.now()

        # inicializamos poblacion
        pop = np.random.randint(1, self.n_stations + 1, (self.pop_size, self.n_tasks))
        pop = pop.astype(float)
        if self.init_pop == "valid":
            pop[0] = np.apply_along_axis(np.round, 0, np.linspace(1, self.n_stations, self.n_tasks))
            for st in range(1, self.n_stations+1):
                pop[st] = np.array([st for i in range(self.n_tasks)])

        # calculamos numero de elites
        num_elit = self.pop_size * self.elit_ratio
        if num_elit < 2 and self.elit_ratio > 0:  # minimo dos
            num_elit = 2
        else:
            num_elit = int(num_elit)
            if num_elit % 2:  # numero par
                num_elit += 1

        # evaluamos a la poblacion
        pop_eval = np.apply_along_axis(self.ev_func, 1, pop)
        pop_eval = pop_eval.reshape((self.pop_size, 1))
        pop = np.append(pop, pop_eval, axis=1)
        # ordenamos de mejor a peor
        index = np.argsort(pop[:, -1])
        pop = pop[index, :]

        # guardamos resultados
        self.best_ev_func = [pop[0, -1]]
        self.best_max_time = [max(self.station_times(pop[0, :-1]).values())]
        self.best_num_restric = [self.check_restrictions(pop[0, :-1])]

        # grafo
        fig = plt.figure()
        if self.init_pop == "random":
            ax1 = fig.add_subplot(311)
            ax2 = fig.add_subplot(312)
            ax3 = fig.add_subplot(313)
            pl1, = ax1.plot(self.best_ev_func, color="blue", label="best_ev_func")
            pl2, = ax2.plot(self.best_max_time, color="red", label="best_max_time")
            pl3, = ax2.plot([sum(self.times) / self.n_stations], color="green", label="ideal_max_time")
            pl4, = ax3.plot(self.best_num_restric, color="brown", label="best_num_restric")
            ax1.set_xlim([0, 100])
            ax1.set_ylim([0, self.best_ev_func[0] + 0.2 * self.best_ev_func[0]])
            ax2.set_xlim([0, 100])
            ax2.set_ylim([0, self.best_max_time[0] + 0.1 * self.best_max_time[0]])
            ax3.set_xlim([0, 100])
            ax3.set_ylim([0, (self.best_num_restric[0] + 0.1 * self.best_num_restric[0])])
            ax1.legend(loc="upper right")
            ax2.legend(loc="lower right")
            ax3.legend(loc="upper right")
            plt.ion()
            plt.show()
        elif self.init_pop == "valid":
            ax1 = fig.add_subplot(211)
            ax2 = fig.add_subplot(212)
            pl1, = ax1.plot(self.best_ev_func, color="blue", label="best_ev_func")
            pl2, = ax2.plot(self.best_max_time, color="red", label="best_max_time")
            pl3, = ax2.plot([sum(self.times) / self.n_stations], color="green", label="ideal_max_time")
            ax1.set_xlim([0, 100])
            ax1.set_ylim([0, self.best_ev_func[0] + 0.2 * self.best_ev_func[0]])
            ax2.set_xlim([0, 100])
            ax2.set_ylim([0, self.best_max_time[0] + 0.1 * self.best_max_time[0]])
            ax1.legend(loc="upper right")
            ax2.legend(loc="lower right")
            plt.ion()
            plt.show()

        it_num = 1

        # comprobamos condicion de parada
        while not self.check_stop(pop):

            it_num += 1

            # aplicamos elitismo
            next_gen = np.empty((self.pop_size, self.n_tasks))
            next_gen[:num_elit, :] = pop[:num_elit, :-1].copy()

            # seleccion
            num_parents = self.pop_size - num_elit
            mating_pool = np.empty((num_parents, self.n_tasks))

            if self.select_type == "tournament":

                for n in range(num_parents):
                    ind1 = np.random.randint(self.pop_size)
                    ind2 = np.random.randint(self.pop_size)
                    if pop[ind1, -1] <= pop[ind2, -1]:
                        mating_pool[n, :] = pop[ind1, :-1].copy()
                    else:
                        mating_pool[n, :] = pop[ind2, :-1].copy()

            elif self.select_type == "roulette":

                weights = pop[:, -1]
                weights = np.max(weights) - weights
                if np.sum(weights) == 0:
                    weights = np.array([1/len(weights) for i in range(len(weights))])
                else:
                    weights = weights / np.sum(weights)

                for n in range(num_parents):
                    ind = np.random.choice(np.arange(self.pop_size), p=weights)
                    mating_pool[n, :] = pop[ind, :-1].copy()

            # cruce
            num_cross = int(self.cross_prob * num_parents)
            if num_cross % 2 != 0:
                num_cross += 1
            ind_cross = np.random.choice(np.arange(num_parents), num_cross, replace=False)  # parejas a cruzar
            parents = mating_pool[ind_cross]

            # cruces y mutaciones
            n = 0
            while n < num_cross:
                child1, child2 = self.cross(parents[n], parents[n+1])

                # mutaciones
                child1, child2 = self.mut(child1), self.mut(child2)

                next_gen[num_elit + n] = child1
                next_gen[num_elit + n + 1] = child2
                n += 2

            # introducimos los padres que no se cruzan
            ind_not_cross = list(set(np.arange(num_parents)).difference(set(ind_cross)))
            next_gen[(num_elit + num_cross):] = mating_pool[ind_not_cross].copy()

            # preparamos siguiente iteracion
            pop = next_gen.copy()

            pop_eval = np.apply_along_axis(self.ev_func, 1, pop)
            pop_eval = pop_eval.reshape((self.pop_size, 1))
            pop = np.append(pop, pop_eval, axis=1)
            # ordenamos de mejor a peor
            index = np.argsort(pop[:, -1])
            pop = pop[index, :]

            # guardamos resultados
            self.best_ev_func.append(pop[0, -1])
            self.best_max_time.append(max(self.station_times(pop[0, :-1]).values()))
            self.best_num_restric.append(self.check_restrictions(pop[0, :-1]))

            # grafo
            if datetime.now().second % 3 == 0:
                ax1.set_xlim([0, it_num])
                max_ev_func = max(self.best_ev_func)
                ax1.set_ylim([0, max_ev_func + 0.2 * max_ev_func])
                ax2.set_xlim([0, it_num])
                max_time = max(self.best_max_time)
                ax2.set_ylim([0, max_time + 0.1 * max_time])
                max_num_restric = max(self.best_num_restric)
                if self.init_pop == "random":
                    ax3.set_ylim([0, max_num_restric + 0.1 * max_num_restric])
                    ax3.set_xlim([0, it_num])
                pl1.set_ydata(self.best_ev_func)
                pl1.set_xdata(range(it_num))
                pl2.set_ydata(self.best_max_time)
                pl2.set_xdata(range(it_num))
                ideal = sum(self.times) / self.n_stations
                pl3.set_ydata([ideal for i in range(it_num)])
                pl3.set_xdata(range(it_num))
                if self.init_pop == "random":
                    pl4.set_ydata(self.best_num_restric)
                    pl4.set_xdata(range(it_num))
                plt.pause(1)

            self.best_ind = pop[0, :-1]

        ax1.set_xlim([0, it_num])
        max_ev_func = max(self.best_ev_func)
        ax1.set_ylim([0, max_ev_func + 0.2 * max_ev_func])
        ax2.set_xlim([0, it_num])
        max_time = max(self.best_max_time)
        ax2.set_ylim([0, max_time + 0.1 * max_time])
        max_num_restric = max(self.best_num_restric)
        if self.init_pop == "random":
            ax3.set_ylim([0, max_num_restric + 0.1 * max_num_restric])
            ax3.set_xlim([0, it_num])
        pl1.set_ydata(self.best_ev_func)
        pl1.set_xdata(range(it_num))
        pl2.set_ydata(self.best_max_time)
        pl2.set_xdata(range(it_num))
        ideal = sum(self.times) / self.n_stations
        pl3.set_ydata([ideal for i in range(it_num)])
        pl3.set_xdata(range(it_num))
        if self.init_pop == "random":
            pl4.set_ydata(self.best_num_restric)
            pl4.set_xdata(range(it_num))
        plt.pause(1)
