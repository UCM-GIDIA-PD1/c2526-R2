import wandb
import ast

PROJECT_NAME = "modelo-texto-f"
METRIC = "test_f1_macro"


def obtener_mejor_run():
    api = wandb.Api()
    runs = api.runs(PROJECT_NAME)

    best_run = None
    best_value = -float("inf")

    for run in runs:

        val = run.summary.get(METRIC)

        if val is None:
            continue

        if val > best_value:
            best_value = val
            best_run = run

    return best_run, best_value


def parse_best_params(run):

    best_params = run.summary.get("best_params", None)
    

    print("Tipo:", type(best_params))
    print("Contenido:", best_params)

    if best_params is None:
        print("⚠️ Este run no tiene best_params")
        return {}

    # Caso 1: ya es dict
    if isinstance(best_params, dict):
        return best_params

    # Caso 2: viene como string
    if isinstance(best_params, str):
        try:
            return ast.literal_eval(best_params)
        except:
            print("⚠️ No se pudo parsear best_params")
            return {}

    print("⚠️ Formato desconocido en best_params")
    return {}


def main():

    run, score = obtener_mejor_run()

    if run is None:
        print("No hay runs válidos")
        return

    print("\n🏆 Mejor run:", run.name)
    print("F1:", score)

    params = parse_best_params(run)

    print("\n⚙️ Hiperparámetros:")
    for k, v in params.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()