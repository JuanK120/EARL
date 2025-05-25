from pymoo.core.problem import Problem
import numpy as np


class EvolutionaryProblem(Problem):

    def __init__(self, n_var, n_obj, n_ieq_constr, xl, xu, fact, obj):
        self.fact = fact
        self.obj = obj

        self.prev_solutions = {}
        self.rew_dict = {}

        super(EvolutionaryProblem, self).__init__(n_var=n_var, n_obj=n_obj, n_ieq_constr=n_ieq_constr, xl=xl, xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):
        fitnesses = []
        constraints = []
        for solution in x:
            f, c = self.fitness_func(solution.squeeze() )
            fitnesses.append(f)
            constraints.append(c)

        out["F"] = np.array(fitnesses)
        out["G"] = np.array(constraints)

    def fitness_func(self, solution):
        return 0, 1