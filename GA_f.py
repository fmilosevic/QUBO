import numpy as np
import matplotlib.pyplot as plt
from pymoo.core.problem import Problem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.operators.crossover.ux import UniformCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.termination.default import DefaultSingleObjectiveTermination
from pymoo.core.callback import Callback
import time



# Callback class 
class MyCallback(Callback):
    def __init__(self):
        super().__init__()
        self.data["best"] = []

    def notify(self, algorithm):
        # Best objective value v této generaci
        best = algorithm.pop.get("F").min()
        self.data["best"].append(best)


# QUBO problem

class QUBOProblem(Problem):
    def __init__(self, Q):
        self.Q = Q
        super().__init__(n_var=Q.shape[0], n_obj=1, n_constr=0, xl=0, xu=1, type_var=np.bool_)
   
    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = np.einsum('ij,bj,bi->b', self.Q, x, x)



# Wrapper pro GA 
def GA_f(Q, 
         ga_pop_size=50, 
         ga_crossover_prob=0.5, 
         ga_mutation_prob=0.1,
         callback=None, 
         seed=None,
         verbose=False,
         termination=None,
         eliminate_duplicates=True,
         save_history=False):
    
    if callback is None:
        callback = MyCallback()


    start_time = time.time()
    
    qubo_problem = QUBOProblem(Q)
    
    algorithm = GA(
        pop_size=ga_pop_size,
        sampling=BinaryRandomSampling(),
        crossover=UniformCrossover(prob=ga_crossover_prob),
        mutation=BitflipMutation(prob=ga_mutation_prob),
        eliminate_duplicates=eliminate_duplicates
    )
    

    res = minimize(
        qubo_problem,
        algorithm,
        seed=seed,
        termination=termination,
        verbose=verbose,
        save_history=save_history,
        callback=callback
    )
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    return {
        'best_solution': res.X,
        'best_obj_val': res.F,
        'time': elapsed_time,
        'res': res,
        'callback': callback
    }



# MAIN 
if __name__ == "__main__":
    
    np.random.seed(1)
    n = 50
    A = np.random.randn(n, n)
    Q = (A + A.T) / 2
    
    cb = MyCallback()
    
    #termination = ("n_eval", 500)
    termination  = DefaultSingleObjectiveTermination(
    xtol=1e-8,
    cvtol=1e-6,
    ftol=1e-6,
    period=20,
    n_max_gen=1000,
    n_max_evals=100000
)
    result = GA_f(
        Q,
        ga_pop_size=50,
        ga_crossover_prob=0.5,
        ga_mutation_prob=0.1,
        seed=1,
        verbose=True,
        termination=termination,
        eliminate_duplicates=True,
        callback=cb
    )
    
    # Plot konvergence 
    vals = result["callback"].data["best"]
    plt.plot(np.arange(len(vals)), vals)
    plt.xlabel("Generation")
    plt.ylabel("Best f(x)")
    plt.title("GA Convergence on QUBO")
    plt.grid(True)
    plt.show()
