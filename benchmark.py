import numpy as np
import json
from pathlib import Path
from datetime import datetime
from pymoo.termination import get_termination
from pymoo.core.callback import Callback
from GA_f import GA_f
from NSGA2_f import NSGA2_f
from MOEAD_f import MOEAD_f
from RVEA_f import RVEA_f
from UNSGA3_f import UNSGA3_f

from get_bqp import get_bqp, get_problem_count

#BQP_PROBLEMS_PATH = Path("")
#OUTPUT_DIR = Path("")

BASE_DIR = Path(__file__).resolve().parent

BQP_PROBLEMS_PATH = BASE_DIR / "test_problems"
OUTPUT_DIR = BASE_DIR / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BQP_FILES = [
    "bqp50.txt",
    "bqp100.txt", 
    "bqp250.txt",
    "bqp500.txt",
    "bqp1000.txt",
    "bqp2500.txt"
]

# tuning / benchmark split
N_TUNE_PROBLEMS = 3
N_TUNE_RUNS = 1
N_TUNE_TRIALS = 55

N_BENCH_PROBLEMS = 7
N_RUNS = 3


MAIN_DIR = OUTPUT_DIR / "main"
SOL_DIR = OUTPUT_DIR / "solutions"
CONV_DIR = OUTPUT_DIR / "convergence"
PARAM_DIR = OUTPUT_DIR / "params"

for d in [MAIN_DIR, SOL_DIR, CONV_DIR, PARAM_DIR]:
    d.mkdir(parents=True, exist_ok=True)



# TERMINATION
MaxEvalsValue = 10000
TUNING_TERMINATION = get_termination("n_eval", 3000)

TERMINATION = get_termination("n_eval", MaxEvalsValue)


# CALLBACK PRO KONVERGENCI
class ConvergenceCallback(Callback):
    def __init__(self, Q):
        super().__init__()
        self.Q = Q
        self.best_cont = []  
        self.best_bin = []   
        self.evals = []
        self.gens = []

    def notify(self, algorithm):
        #Spojitá hodnota 
        F = algorithm.pop.get("F")
        self.best_cont.append(float(np.min(F[:, 0])))
        
        #binární hodnota 
        X = algorithm.pop.get("X")
        X_bin = np.round(X).astype(int)
        
        f1_bin = np.einsum('ij,bj,bi->b', self.Q, X_bin, X_bin)
        self.best_bin.append(float(np.min(f1_bin)))

        self.evals.append(int(algorithm.evaluator.n_eval))
        self.gens.append(int(algorithm.n_gen))



ALGORITHMS = {
    "GA": GA_f,
    "NSGA2": NSGA2_f,
    "MOEAD": MOEAD_f,
    "RVEA": RVEA_f,
    "UNSGA3": UNSGA3_f,
}

DEFAULT_PARAMS = {
    "GA": dict(ga_pop_size=50, ga_crossover_prob=0.9, ga_mutation_prob=0.01),
    "NSGA2": dict(nsga2_pop_size=50, nsga2_n_neighbors=15,
                  nsga2_crossover_eta=20, nsga2_mutation_eta=20),
    "MOEAD": dict(moead_n_partitions=50, moead_n_neighbors=15,
                  moead_prob_neighbor_mating=0.9,
                  moead_crossover_eta=20, moead_mutation_eta=20),
    "RVEA": dict(rvea_n_partitions=50, rvea_alpha=2.0,
                 rvea_adapt_freq=0.1,
                 rvea_crossover_eta=20, rvea_mutation_eta=20),
    "UNSGA3": dict(unsga3_n_partitions=50,
                   unsga3_crossover_eta=20, unsga3_mutation_eta=20)
}

def safe_float(x):
    if isinstance(x, np.ndarray):
        return float(np.min(x))
    return float(x)


def run_algorithm(algo_name, algo_func, Q, params):

    callback = ConvergenceCallback(Q)

    result = algo_func(
        Q,
        **params,
        termination=TERMINATION,   
        verbose=False,
        save_history=False,
        callback=callback
    )

    best_solution = result.get("best_solution")
    if isinstance(best_solution, np.ndarray):
        best_solution = best_solution.tolist()

    return {
        "main": {
            "best_obj_val": safe_float(result["best_obj_val"]),
            "time": float(result["time"]),
            "n_eval": int(result["res"].algorithm.evaluator.n_eval)
        },
        "solution": best_solution,
        "convergence": {
            "best_cont": callback.best_cont, # Dříve "best"
            "best_bin": callback.best_bin,   # Zcela nové
            "evals": callback.evals,
            "gens": callback.gens
        }
    }



# HYPERPARAMETER TUNING
import optuna

def tune_hyperparameters(algo_name, algo_func, Q_list, default_params):

    def objective(trial):

        if algo_name == "GA":
            params = {
                "ga_pop_size": trial.suggest_int("ga_pop_size", 10, 150),
                "ga_crossover_prob": trial.suggest_float("ga_crossover_prob", 0.5, 1.0),
                "ga_mutation_prob": trial.suggest_float("ga_mutation_prob", 1e-3, 0.2),
            }

        elif algo_name == "NSGA2":
            params = {
                "nsga2_pop_size": trial.suggest_int("nsga2_pop_size", 10, 150),
                "nsga2_n_neighbors": trial.suggest_int("nsga2_n_neighbors", 2, 35),
                "nsga2_crossover_eta": trial.suggest_float("nsga2_crossover_eta", 2, 30.0),
                "nsga2_mutation_eta": trial.suggest_float("nsga2_mutation_eta", 2, 30.0),
            }

        elif algo_name == "MOEAD":
            params = {
                "moead_n_partitions": trial.suggest_int("moead_n_partitions", 10, 100),
                "moead_n_neighbors": trial.suggest_int("moead_n_neighbors", 2, 35),
                "moead_prob_neighbor_mating": trial.suggest_float(
                    "moead_prob_neighbor_mating", 0.6, 1.0
                ),
                "moead_crossover_eta": trial.suggest_float("moead_crossover_eta", 2.0, 30.0),
                "moead_mutation_eta": trial.suggest_float("moead_mutation_eta", 2.0, 30.0),
            }

        elif algo_name == "RVEA":
            params = {
                "rvea_n_partitions": trial.suggest_int("rvea_n_partitions", 10, 100),
                "rvea_alpha": trial.suggest_float("rvea_alpha", 1.0, 5.0),
                "rvea_adapt_freq": trial.suggest_float("rvea_adapt_freq", 0.05, 0.3),
                "rvea_crossover_eta": trial.suggest_float("rvea_crossover_eta", 2.0, 30.0),
                "rvea_mutation_eta": trial.suggest_float("rvea_mutation_eta", 2.0, 30.0),
            }

        elif algo_name == "UNSGA3":
            params = {
                "unsga3_n_partitions": trial.suggest_int("unsga3_n_partitions", 10, 100),
                "unsga3_crossover_eta": trial.suggest_float("unsga3_crossover_eta", 2.0, 30.0),
                "unsga3_mutation_eta": trial.suggest_float("unsga3_mutation_eta", 2.0, 30.0),
            }

        else:
            params = default_params.copy()


        scores = []

        for Q in Q_list:
            for _ in range(N_TUNE_RUNS):
                res = algo_func(
                    Q,
                    **params,
                    termination=TUNING_TERMINATION,
                    verbose=False
                )
                scores.append(safe_float(res["best_obj_val"]))

        return float(np.mean(scores))

    # --- SPO study (TPE) ---
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(
            n_startup_trials=max(5, N_TUNE_TRIALS // 5),
            multivariate=True,
            constant_liar=True
        )
    )

    study.optimize(objective, n_trials=N_TUNE_TRIALS, show_progress_bar=False)

    return study.best_params



def main():

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 70)
    print("QUBO BENCHMARK WITH HYPERPARAMETER TUNING")
    print("=" * 70)

    for bqp_file in BQP_FILES:

        path = BQP_PROBLEMS_PATH / bqp_file
        n_total = get_problem_count(str(path))

        tune_Q = []
        bench_Q = []

        for i in range(n_total):
            n, Q = get_bqp(str(path), i)
            Q=-Q                                            #benchmark problemy se májí maximalizovat, namísto minimalizovat
            if i < N_TUNE_PROBLEMS:
                tune_Q.append(Q)
            elif i < N_TUNE_PROBLEMS + N_BENCH_PROBLEMS:
                bench_Q.append((i, Q))

        size_tag = f"n{n}"

        print("\n--- HYPERPARAMETER TUNING ---")

        tuned_params = {}

        for algo_name, algo_func in ALGORITHMS.items():
            print(f"\n  Tuning {algo_name}")
            tuned_params[algo_name] = tune_hyperparameters(
                algo_name,
                algo_func,
                tune_Q,
                DEFAULT_PARAMS[algo_name]
            )

        with open(PARAM_DIR / f"tuned_params_{size_tag}_{timestamp}.json", "w") as f:
            json.dump(tuned_params, f, indent=2)

        print("\n--- BENCHMARK ---")

        main_results, sol_results, conv_results = {}, {}, {}

        for p_idx, Q in bench_Q:
            key = f"problem_{p_idx}"
            main_results[key] = {}
            sol_results[key] = {}
            conv_results[key] = {}

            print(f"\n  Problem {p_idx} | n={n}")

            for algo_name, algo_func in ALGORITHMS.items():
                print(f"    {algo_name}")

                mr, sr, cr = [], [], []

                for r in range(N_RUNS):
                    res = run_algorithm(algo_name, algo_func, Q, tuned_params[algo_name])
                    mr.append(res["main"])
                    sr.append(res["solution"])
                    cr.append(res["convergence"])

                    print(f"      run {r}: f={res['main']['best_obj_val']:.2f}")

                main_results[key][algo_name] = mr
                sol_results[key][algo_name] = sr
                conv_results[key][algo_name] = cr

        with open(MAIN_DIR / f"benchmark_main_{size_tag}_{timestamp}.json", "w") as f:
            json.dump(main_results, f, indent=2)

        with open(SOL_DIR / f"benchmark_solutions_{size_tag}_{timestamp}.json", "w") as f:
            json.dump(sol_results, f, indent=2)

        with open(CONV_DIR / f"benchmark_convergence_{size_tag}_{timestamp}.json", "w") as f:
            json.dump(conv_results, f, indent=2)

        print(f"\nSaved results for {size_tag}")

    print("\nDONE")


if __name__ == "__main__":
    main()