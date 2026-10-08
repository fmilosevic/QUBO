import json
import re
import numpy as np
from pathlib import Path

#path
_BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = _BASE_DIR / "results" / "main"

def main():
    if not RESULTS_DIR.exists() or not RESULTS_DIR.is_dir():
        print(f"CHYBA: Složka {RESULTS_DIR} neexistuje.")
        return

    # Nalezení všech JSON souborů ve složce
    json_files = list(RESULTS_DIR.glob("*.json"))
    
    if not json_files:
        print(f"CHYBA: Ve složce {RESULTS_DIR} nebyly nalezeny žádné JSON soubory.")
        return

    parsed_files = []
    for filepath in json_files:
        match = re.search(r'_n(\d+)_', filepath.name)
        if match:
            size_str = match.group(1)
            size_int = int(size_str)
        else:
            size_str = filepath.stem
            size_int = 999999 
            
        parsed_files.append((size_int, size_str, filepath))

    parsed_files.sort(key=lambda x: x[0])

    print("\n" + "=" * 80)
    print("VÝPOČET POPISNÉ STATISTIKY (KONEČNÉ HODNOTY)")
    print("Tip: Ve Wordu dejte Vložit -> Tabulka -> Převést text na tabulku")
    print("=" * 80)

    # Zpracování pro každou velikost problému 
    for size_int, problem_size, filepath in parsed_files:
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Chyba při čtení souboru {filepath.name}: {e}")
            continue

        # zjištění všech algoritmů a shromáždění konečných hodnot
        final_values = {}
        
        for problem_name, algos_data in data.items():
            for algo, runs in algos_data.items():
                if algo not in final_values:
                    final_values[algo] = []
                    
                for run in runs:
                    if "best_obj_val" in run:
                        final_values[algo].append(run["best_obj_val"])
        
        if not final_values:
            continue


        sorted_algos = sorted(final_values.keys())
        
        best_mean = {'algo': None, 'val': float('inf')}
        best_median = {'algo': None, 'val': float('inf')}
        best_q1 = {'algo': None, 'val': float('inf')}
        best_min = {'algo': None, 'val': float('inf')}

        print(f"\n--- Tabulka pro velikost: n={problem_size} ---")
        print("Algoritmus !Průměr !Medián !1. Kvartil (Q1) !Minimum")
        
        # Výpočet statistik pro tabulku
        for algo in sorted_algos:
            vals = final_values[algo]
            
            if not vals:
                continue
                
            mean_val = np.mean(vals)
            median_val = np.median(vals)
            q1_val = np.percentile(vals, 25)
            min_val = np.min(vals)
            
            mean_str = f"{mean_val:.2f}".replace('.', ',')
            median_str = f"{median_val:.2f}".replace('.', ',')
            q1_str = f"{q1_val:.2f}".replace('.', ',')
            min_str = f"{min_val:.2f}".replace('.', ',')
            
            print(f"{algo} !{mean_str} !{median_str} !{q1_str} !{min_str}")
            
            # Aktualizace vítězů
            if mean_val < best_mean['val']:
                best_mean = {'algo': algo, 'val': mean_val}
            if median_val < best_median['val']:
                best_median = {'algo': algo, 'val': median_val}
            if q1_val < best_q1['val']:
                best_q1 = {'algo': algo, 'val': q1_val}
            if min_val < best_min['val']:
                best_min = {'algo': algo, 'val': min_val}

        #Výpis vítězných algoritmů pod tabulkou
        print("\nNejlepší algoritmy v jednotlivých kategoriích:")
        print(f"  Nejnižší průměr:       {best_mean['algo']} ({best_mean['val']:.2f})")
        print(f"  Nejnižší medián:       {best_median['algo']} ({best_median['val']:.2f})")
        print(f"  Nejnižší 1. kvartil:   {best_q1['algo']} ({best_q1['val']:.2f})")
        print(f"  Absolutní minimum:     {best_min['algo']} ({best_min['val']:.2f})")
        print("-" * 65)

if __name__ == "__main__":
    main()