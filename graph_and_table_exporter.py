import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re  

# config
_BASE_DIR = Path(__file__).resolve().parent

CONV_DIR = _BASE_DIR / "results" / "convergence"
MAIN_DIR = _BASE_DIR / "results" / "main"


json_files = list(CONV_DIR.glob("*.json"))

if not json_files:
    raise FileNotFoundError(
        f"Ve složce '{CONV_DIR}' nebyl nalezen žádný .json soubor"
    )


CONV_FILE_PATH = max(json_files, key=lambda p: p.stat().st_mtime)
main_filename = CONV_FILE_PATH.name.replace("convergence", "main")
MAIN_FILE_PATH = MAIN_DIR / main_filename

# Procentuální hladiny evaluací
PERCENTAGES = [0.01, 0.05, 0.10, 0.30, 0.50, 0.80, 1.00]
PERCENT_LABELS = ["1%", "5%", "10%", "30%", "50%", "80%", "100%"]

def main():
    match = re.search(r'_n(\d+)_', CONV_FILE_PATH.name)
    if match:
        problem_size = f"n={match.group(1)}"
    else:
        problem_size = "Neznámá velikost"
        
    try:
        with open(CONV_FILE_PATH, 'r', encoding='utf-8') as f:
            data_conv = json.load(f)
        with open(MAIN_FILE_PATH, 'r', encoding='utf-8') as f:
            data_main = json.load(f)
    except FileNotFoundError as e:
        print(f"CHYBA: Soubor nebyl nalezen. Zkontroluj cesty.\n{e}")
        return

    print("\n" + "=" * 80)
    print(f"ZPRACOVÁNÍ DAT: Velikost problému {problem_size}")
    print("=" * 80)

    max_evals_overall = 0
    all_algos = set()
    
    for problem_name, algos_data in data_conv.items():
        for algo, runs in algos_data.items():
            all_algos.add(algo)
            for run in runs:
                if run.get("evals"):
                    max_evals_overall = max(max_evals_overall, max(run["evals"]))
    
    if max_evals_overall == 0:
        print("CHYBA: Nebyla nalezena žádná data o evaluacích.")
        return

    target_evals = [int(p * max_evals_overall) for p in PERCENTAGES]
    plot_evals = np.linspace(0, max_evals_overall, 300)

    filename = CONV_FILE_PATH.stem
    absolutni_cesty_konvergence = []
    
    final_results = {algo: [] for algo in all_algos}

    # hlavni cyklus pro oba grafy
    typy_konvergence = [("best_cont", "Spojitá konvergence"), 
                        ("best_bin", "Binarizovaná konvergence")]

    for type_best, type_name in typy_konvergence:
        print(f"\n\n{'#' * 80}")
        print(f"ZPRACOVÁVÁM TYP: {type_name.upper()}")
        print(f"{'#' * 80}")

        table_runs = {algo: [] for algo in all_algos}
        plot_runs = {algo: [] for algo in all_algos}
        
        final_results = {algo: [] for algo in all_algos}

        for problem_name, algos_data in data_conv.items():
            for algo, runs in algos_data.items():
                for run_idx, run in enumerate(runs):
                    evals = run.get("evals")
                    
                    bests = run.get(type_best, run.get("best"))
                    
                    if not evals or not bests:
                        continue
                        
                    interp_table = np.interp(target_evals, evals, bests)
                    interp_plot = np.interp(plot_evals, evals, bests)
                    
                    #finalni hodnota z main namisto convergence
                    if problem_name in data_main and algo in data_main[problem_name]:
                        try:
                            final_val = data_main[problem_name][algo][run_idx]["best_obj_val"]
                            final_results[algo].append(final_val)
                            
                            # Nahrazení posledního bodu 
                            interp_table[-1] = final_val
                            interp_plot[-1] = final_val
                        except IndexError:
                            pass
                    
                    table_runs[algo].append(interp_table)
                    plot_runs[algo].append(interp_plot)

        algo_means_table = {}
        algo_means_plot = {}
        sorted_algos = sorted(all_algos)
        
        for algo in sorted_algos:
            if table_runs[algo]:
                algo_means_table[algo] = np.mean(table_runs[algo], axis=0)
                algo_means_plot[algo] = np.mean(plot_runs[algo], axis=0)

        print(f"\n--- CELKOVÁ TABULKA PRO MICROSOFT WORD ({type_name}) ---")
        header = "Algoritmus !" + " !".join(PERCENT_LABELS)
        print(header)
        
        for algo, means in algo_means_table.items():
            row_vals = [f"{val:.2f}".replace('.', ',') for val in means]
            print(f"{algo} !" + " !".join(row_vals))

        print(f"\n--- NEJLEPŠÍ ALGORITMUS V DANÉ FÁZI ({type_name}) ---")
        for i, pct_label in enumerate(PERCENT_LABELS):
            best_algo = None
            best_val = float('inf')
            
            for algo, means in algo_means_table.items():
                if means[i] < best_val:
                    best_val = means[i]
                    best_algo = algo
                    
            print(f"{pct_label} evaluací: Nejlepší je {best_algo} (Hodnota: {best_val:.2f})")

        plt.figure(figsize=(10, 6), dpi=300) 
        
        markers = ['o', 's', '^', 'D', 'v', 'p', 'X']
        for idx, algo in enumerate(sorted_algos):
            if algo in algo_means_plot:
                marker = markers[idx % len(markers)]
                plt.plot(plot_evals, algo_means_plot[algo], label=algo, linewidth=2, marker=marker, markevery=30, markersize=6)
            
        plt.title(f"{problem_size} - {type_name}", fontsize=14, pad=15)
        plt.xlabel("Počet evaluací", fontsize=12)
        plt.ylabel("Průměrná hodnota účelové funkce", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(fontsize=11)
        plt.tight_layout()
        
        plot_filename = f"graf_konvergence_{filename}_{type_best}.png"
        plt.savefig(plot_filename, format='png', facecolor='white')
        plt.close()
        
        absolutni_cesty_konvergence.append(Path(plot_filename).resolve())


    # boxplot
    plt.figure(figsize=(10, 6), dpi=300)
    
    algos_for_boxplot = sorted_algos
    data_for_boxplot = [final_results[algo] for algo in algos_for_boxplot if final_results[algo]]
    
    # nastaveni boxplotu
    boxprops = dict(linestyle='-', linewidth=2, color='blue')
    medianprops = dict(linestyle='-', linewidth=2, color='firebrick')
    
    plt.boxplot(data_for_boxplot, labels=algos_for_boxplot, 
                boxprops=boxprops, medianprops=medianprops, patch_artist=True)
    
    plt.title(f"{problem_size}", fontsize=14, pad=15)
    plt.ylabel("Hodnota účelové funkce ", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    boxplot_filename = f"boxplot_{filename}_celkovy.png"
    plt.savefig(boxplot_filename, format='png', facecolor='white')
    plt.close()
    
    absolutni_cesta_boxplot = Path(boxplot_filename).resolve()

    #vypis cest
    print("\n" + "=" * 80)
    print("Grafy byly úspěšně vygenerovány a uloženy sem:")
    for cesta in absolutni_cesty_konvergence:
        print(f"Konvergence: {cesta}")
    print(f"Boxplot:     {absolutni_cesta_boxplot}")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()