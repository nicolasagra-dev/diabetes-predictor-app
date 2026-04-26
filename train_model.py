from __future__ import annotations

from diabetes_model import (
    MODEL_COMPARISON_PATH,
    MODEL_PATH,
    compare_models,
    load_data,
    save_model,
    save_model_comparison,
    train_model,
)


def main() -> None:
    data = load_data()
    bundle = train_model(data)
    comparison = compare_models(data)
    save_model(bundle)
    save_model_comparison(comparison)

    metrics = bundle["metrics"]
    print(f"Modelo salvo em: {MODEL_PATH}")
    print(f"Acuracia: {metrics['accuracy']:.3f}")
    print(f"Precisao: {metrics['precision']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    print(f"F1-score: {metrics['f1']:.3f}")
    print(f"ROC AUC: {metrics['roc_auc']:.3f}")
    print(f"Comparacao salva em: {MODEL_COMPARISON_PATH}")


if __name__ == "__main__":
    main()
