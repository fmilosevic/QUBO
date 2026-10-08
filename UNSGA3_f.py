import numpy as np
import time
import matplotlib.pyplot as plt

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.unsga3 import UNSGA3
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.core.callback import Callback



# Callback Class 
class MyCallback(Callback):
    def __init__(self):
        super().__init__()
        self.data["best"] = []

    def notify(self, algorithm):
        F = algorithm.pop.get("F")   # shape (pop, 2)
        best_f1 = np.min(F[:, 0])
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
        f1 = np.einsum('ij,bj,bi->b', self.Q, x, x)
        f2 = np.sum(x * (1 - x), axis=1)
        out["F"] = np.column_stack([f1, f2])


# UNSGA3 
def UNSGA3_f(Q, 
             unsga3_n_partitions=50, 
             unsga3_crossover_eta=15, 
             unsga3_mutation_eta=30, 
             callback=None,
             seed=None, 
             verbose=False,
             termination=None,
             save_history=False):

    start_time = time.time()

    if callback is None:
        callback = MyCallback()
    
    problem = BBCO_QUBO(Q)

    algorithm = UNSGA3(
        ref_dirs=get_reference_directions("uniform", 2, n_partitions=unsga3_n_partitions),
        crossover=SBX(eta=unsga3_crossover_eta),
        mutation=PolynomialMutation(eta=unsga3_mutation_eta)
    )
    
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


# Test + plot
if __name__ == "__main__":
    np.random.seed(1)
    n = 500
    A = np.random.randn(n, n)
    Q = (A + A.T) / 2

    termination = ("n_eval", 1250)

    cb = MyCallback()

    result = UNSGA3_f(
        Q,
        unsga3_n_partitions=50,
        unsga3_crossover_eta=15,
        unsga3_mutation_eta=30,
        seed=1,
        verbose=True,
        termination=termination,
        callback=cb
    )

    vals = result["callback"].data["best"]
    plt.plot(np.arange(len(vals)), vals)
    plt.xlabel("Generation")
    plt.ylabel("Best f1")
    plt.title("UNSGA-III Convergence on QUBO")
    plt.grid(True)
    plt.show()
