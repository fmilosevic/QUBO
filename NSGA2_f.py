import numpy as np
import time
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.survival.rank_and_crowding import RankAndCrowding
from pymoo.core.callback import Callback



# Callback Class 
class MyCallback(Callback):
    def __init__(self):
        super().__init__()
        self.data["best"] = []

    def notify(self, algorithm):
        # algorithm.pop.get("F") má tvar (pop_size, 2)
        F = algorithm.pop.get("F")
        best_f1 = np.min(F[:, 0])      # f1 = první cíl
        self.data["best"].append(best_f1)


# prevod na spojity problem
class BBCO_QUBO(Problem):
    def __init__(self, Q):
        super().__init__(
            n_var=Q.shape[0],
            n_obj=2,
            n_constr=0,
            xl=0.0,
            xu=1.0,
            type_var=np.float64
        )
        self.Q = Q
        
    def _evaluate(self, x, out, *args, **kwargs):
        # f1 = x^T Q x
        f1 = np.einsum('ij,bj,bi->b', self.Q, x, x)

        # f2 = soft binarity penalty
        f2 = np.sum(x * (1 - x), axis=1)

        out["F"] = np.column_stack([f1, f2])



# NSGA2 
def NSGA2_f(Q, 
            nsga2_pop_size=100, 
            nsga2_n_neighbors=15, 
            nsga2_crossover_eta=15,
            nsga2_mutation_eta=20, 
            callback=None,
            seed=None,
            verbose=False,
            termination=None,
            save_history=False):

    start_time = time.time()

    # Default callback 
    if callback is None:
        callback = MyCallback()

    problem = BBCO_QUBO(Q)

    algorithm = NSGA2(
        pop_size=nsga2_pop_size,
        n_neighbors=nsga2_n_neighbors,
        mutation=PolynomialMutation(eta=nsga2_mutation_eta),
        survival=RankAndCrowding(crowding_func="pcd"),
        crossover=SBX(eta=nsga2_crossover_eta)
    )

    # Run optimization
    res = minimize(
        problem,
        algorithm,
        termination=termination,
        seed=seed,
        verbose=verbose,
        save_history=save_history,
        callback=callback
    )

    X_cont = res.X
    X_bin = np.round(X_cont).astype(int)
    f1_bin = np.einsum('ij,bj,bi->b', Q, X_bin, X_bin)

    best_idx = np.argmin(f1_bin)
    best_solution = X_bin[best_idx]
    best_f1 = f1_bin[best_idx]

    elapsed_time = time.time() - start_time

    return {
        'X_cont': X_cont,
        'X_bin': X_bin,
        'f1_bin': f1_bin,
        'best_solution': best_solution,
        'best_obj_val': best_f1,
        'time': elapsed_time,
        'res': res,
        'callback': callback
    }


# Test 
if __name__ == "__main__":
    np.random.seed(1)
    n = 50
    A = np.random.randn(n, n)
    Q = (A + A.T) / 2

    termination = ("n_eval", 2700)

    result = NSGA2_f(
        Q,
        nsga2_pop_size=100,
        nsga2_n_neighbors=15,
        nsga2_crossover_eta=15,
        nsga2_mutation_eta=20,
        seed=1,
        verbose=True,
        termination=termination
    )

    print("Best objective:", result["best_obj_val"])
