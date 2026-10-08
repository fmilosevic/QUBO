import json
import re
import numpy as np
from pathlib import Path
from scipy.stats import shapiro, levene, friedmanchisquare, rankdata, ttest_rel, wilcoxon

_BASE_DIR = Path(__file__).resolve().parent

_DEFAULT_CONTROL_ALGOS = {
    "n50":   "GA",
    "n100":  "GA",
    "n250":  "GA",
    "n500":  "GA",
    "n1000": "GA",
    "n2500": "GA",
}
_DEFAULT_CONTROL = "GA"


# pomocne funkce

def _parse_and_sort_files(results_dir):
    """Return [(size_int, size_str, filepath), ...] sorted by size_int."""
    json_files = list(results_dir.glob("*.json"))
    parsed = []
    for fp in json_files:
        match = re.search(r'_(n\d+)_', fp.name)
        if match:
            size_str = match.group(1)
            size_int = int(size_str.replace('n', ''))
        else:
            size_str = fp.stem
            size_int = 999999
        parsed.append((size_int, size_str, fp))
    parsed.sort(key=lambda x: x[0])
    return parsed


def _load_algo_values(data):
    """Flatten all runs across all problems into per-algorithm lists."""
    algo_values = {}
    problem_names = list(data.keys())
    if not problem_names:
        return algo_values, []
    algo_names = list(data[problem_names[0]].keys())
    for algo in algo_names:
        algo_values[algo] = []
    for problem in problem_names:
        for algo in algo_names:
            if algo in data[problem]:
                for run in data[problem][algo]:
                    algo_values[algo].append(run["best_obj_val"])
    return algo_values, algo_names


# test functions


def run_shapiro(algo_values, alpha, problem_size):

    print("\n" + "=" * 65)
    print(f"SHAPIRO-WILKŮV TEST - Velikost problému: {problem_size}")
    print("=" * 65)

    results = {}
    all_normal = True

    for algo_name, values in algo_values.items():
        n = len(values)
        if n < 3:
            print(f"{algo_name}: Nelze provést test (pouze {n} hodnot).")
            all_normal = False
            results[algo_name] = {"stat": None, "p_val": None, "is_normal": False}
            continue

        stat, p_val = shapiro(values)
        is_normal = p_val > alpha
        if not is_normal:
            all_normal = False

        conclusion = "Normální" if is_normal else "Není normální"
        print(f"Algoritmus: {algo_name}")
        print(f"  W-statistika: {stat:.4f}")
        print(f"  p-hodnota:    {p_val:.4e}")
        print(f"  Závěr:        {conclusion}\n")

        results[algo_name] = {"stat": stat, "p_val": p_val, "is_normal": is_normal}

    return results, all_normal


def run_levene(algo_values, algo_names, alpha, problem_size):

    print("\n" + "=" * 65)
    print(f"LEVENEŮV TEST - Velikost problému: {problem_size}")
    print("=" * 65)

    arrays = [algo_values[a] for a in algo_names]

    if len(arrays) < 2:
        print(f"Nedostatek algoritmů pro porovnání rozptylů.")
        return {"stat": None, "p_val": None, "is_homoscedastic": False}, False

    stat, p_val = levene(*arrays)
    is_homo = p_val > alpha
    conclusion = "Rozptyly jsou shodné" if is_homo else "Rozptyly se liší"

    print(f"Porovnávané algoritmy: {', '.join(algo_names)}")
    print(f"  W-statistika: {stat:.4f}")
    print(f"  p-hodnota:    {p_val:.4e}")
    print(f"  Závěr:        {conclusion}\n")

    return {"stat": stat, "p_val": p_val, "is_homoscedastic": is_homo}, is_homo


def run_friedman(algo_values, algo_names, alpha, problem_size):

    print("\n" + "=" * 65)
    print(f"FRIEDMANŮV TEST - Velikost problému: {problem_size}")
    print("=" * 65)

    arrays = [algo_values[a] for a in algo_names]

    if len(arrays) < 2 or len(arrays[0]) < 3:
        print("Nedostatek dat pro provedení Friedmanova testu.")
        return {"stat": None, "p_val": None, "is_diff": False,
                "best_algo": None, "avg_ranks": {}}

    matrix = np.array(arrays).T
    ranks = np.zeros_like(matrix, dtype=float)
    for i in range(matrix.shape[0]):
        ranks[i, :] = rankdata(matrix[i, :])

    avg_ranks = np.mean(ranks, axis=0)
    best_algo = algo_names[int(np.argmin(avg_ranks))]



    stat, p_val = friedmanchisquare(*arrays)
    is_diff = p_val < alpha
    conclusion = "Existuje významný rozdíl" if is_diff else "Není významný rozdíl"

    print(f"Porovnávané algoritmy: {', '.join(algo_names)}")
    print(f"  Testová statistika (Q): {stat:.4f}")
    print(f"  p-hodnota:              {p_val:.4e}")
    print(f"  Závěr:                  {conclusion}")
    print(f"  Nejlepší algoritmus:    {best_algo} (nejnižší průměrný rank)\n")

    return {
        "stat": stat, "p_val": p_val, "is_diff": is_diff,
        "best_algo": best_algo,
        "avg_ranks": dict(zip(algo_names, avg_ranks.tolist()))
    }


def run_ttest(algo_values, algo_names, control_algo, alpha, problem_size):

    print("\n" + "=" * 80)
    print(f"PÁROVÝ T-TEST S HOLMOVOU KOREKCÍ - Velikost: {problem_size}")
    print(f"Kontrolní algoritmus: {control_algo}")
    print("=" * 80)

    control_values = algo_values[control_algo]
    raw = []

    for algo in algo_names:
        if algo == control_algo:
            continue
        stat, p_val = ttest_rel(control_values, algo_values[algo])
        if np.isnan(p_val):
            stat, p_val = 0.0, 1.0
        raw.append({"algo": algo, "stat": stat, "p_val": p_val})

    raw.sort(key=lambda x: x["p_val"])
    m = len(raw)
    reject_h0 = True
    results = []

    for i, res in enumerate(raw):
        k = i + 1
        adj_alpha = alpha / (m + 1 - k)

        if reject_h0 and res["p_val"] <= adj_alpha:
            is_sig = True
        else:
            is_sig = False
            reject_h0 = False

        conclusion = "Významný" if is_sig else "Nevýznamný"
        print(f"{control_algo} vs {res['algo']}:")
        print(f"  T-statistika:      {res['stat']:.4f}")
        print(f"  P-hodnota:         {res['p_val']:.4e}")
        print(f"  Holmova hranice:   {adj_alpha:.4f} (pro k={k})")
        print(f"  Závěr:             {conclusion}\n")

        results.append({
            "comparison": f"{control_algo} vs {res['algo']}",
            "stat": res["stat"], "p_val": res["p_val"],
            "adj_alpha": adj_alpha, "is_sig": is_sig
        })

    return results


def run_wilcoxon(algo_values, algo_names, control_algo, alpha, problem_size):

    print("\n" + "=" * 80)
    print(f"WILCOXON S HOLMOVOU KOREKCÍ - Velikost: {problem_size}")
    print(f"Kontrolní algoritmus: {control_algo}")
    print("=" * 80)

    control_values = algo_values[control_algo]
    raw = []

    for algo in algo_names:
        if algo == control_algo:
            continue
        try:
            if np.allclose(control_values, algo_values[algo]):
                stat, p_val = 0.0, 1.0
            else:
                w_res = wilcoxon(control_values, algo_values[algo])
                stat, p_val = w_res.statistic, w_res.pvalue
        except ValueError:
            stat, p_val = 0.0, 1.0
        raw.append({"algo": algo, "stat": stat, "p_val": p_val})

    raw.sort(key=lambda x: x["p_val"])
    m = len(raw)
    reject_h0 = True
    results = []

    for i, res in enumerate(raw):
        k = i + 1
        adj_alpha = alpha / (m + 1 - k)

        if reject_h0 and res["p_val"] <= adj_alpha:
            is_sig = True
        else:
            is_sig = False
            reject_h0 = False

        conclusion = "Významný" if is_sig else "Nevýznamný"
        print(f"{control_algo} vs {res['algo']}:")
        print(f"  W-statistika:      {res['stat']:.4f}")
        print(f"  P-hodnota:         {res['p_val']:.4e}")
        print(f"  Holmova hranice:   {adj_alpha:.4f} (pro k={k})")
        print(f"  Závěr:             {conclusion}\n")

        results.append({
            "comparison": f"{control_algo} vs {res['algo']}",
            "stat": res["stat"], "p_val": res["p_val"],
            "adj_alpha": adj_alpha, "is_sig": is_sig
        })

    return results


# Main pipeline

def main(
    results_dir=None,
    alpha=0.05,
    control_algos=None,
    default_control=None,
):

    if results_dir is None:
        results_dir = _BASE_DIR / "results" / "main"
    if control_algos is None:
        control_algos = _DEFAULT_CONTROL_ALGOS
    if default_control is None:
        default_control = _DEFAULT_CONTROL

    results_dir = Path(results_dir)

    if not results_dir.exists() or not results_dir.is_dir():
        print(f"CHYBA: Složka {results_dir} neexistuje.")
        return {}

    parsed_files = _parse_and_sort_files(results_dir)
    if not parsed_files:
        print(f"CHYBA: Ve složce {results_dir} nebyly nalezeny žádné JSON soubory.")
        return {}

    all_results = {}

    for size_int, problem_size, filepath in parsed_files:
        print(f"# PROBLEM SIZE: {problem_size}  |  {filepath.name}")

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Chyba při čtení souboru {filepath.name}: {e}")
            continue

        algo_values, algo_names = _load_algo_values(data)
        if not algo_names:
            print("Žádná data nenalezena.")
            continue

        # --- 1. Shapiro-Wilk ---
        shapiro_results, all_normal = run_shapiro(algo_values, alpha, problem_size)

        # --- 2. Levene  ---
        levene_result, is_homoscedastic = run_levene(
            algo_values, algo_names, alpha, problem_size
        )

        # --- 3. Friedman  ---
        friedman_result = run_friedman(algo_values, algo_names, alpha, problem_size)

        # --- 4. Pairwise test  ---
        control_algo = control_algos.get(problem_size, default_control)

        if control_algo not in algo_names:
            print(f"CHYBA: Kontrolní algoritmus '{control_algo}' "
                  f"neexistuje ve velikosti {problem_size}. Párový test přeskočen.")
            pairwise_results = []
            pairwise_test = None
        else:
            use_parametric = all_normal and is_homoscedastic
            pairwise_test = "t-test" if use_parametric else "wilcoxon"

            print(f"\n>>> Shapiro-Wilk: {'všichni prošli' if all_normal else 'někdo neprošel'} | "
                  f"Levene: {'prošel' if is_homoscedastic else 'neprošel'} "
                  f"→ {'párový t-test' if use_parametric else 'Wilcoxonův test'}")

            if use_parametric:
                pairwise_results = run_ttest(
                    algo_values, algo_names, control_algo, alpha, problem_size
                )
            else:
                pairwise_results = run_wilcoxon(
                    algo_values, algo_names, control_algo, alpha, problem_size
                )

        all_results[problem_size] = {
            "shapiro":      shapiro_results,
            "levene":       levene_result,
            "friedman":     friedman_result,
            "pairwise_test": pairwise_test,
            "pairwise":     pairwise_results,
        }

    return all_results


if __name__ == "__main__":
    main()