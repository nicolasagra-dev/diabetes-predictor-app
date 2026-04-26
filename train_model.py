from __future__ import annotations

from diabetes_model import MODEL_PATH, load_data, save_model, train_model


def main() -> None:
    data = load_data()
    bundle = train_model(data)
    save_model(bundle)

    metrics = bundle["metrics"]
    print(f"Modelo salvo em: {MODEL_PATH}")
    print(f"Acuracia: {metrics['accuracy']:.3f}")
    print(f"Precisao: {metrics['precision']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    print(f"F1-score: {metrics['f1']:.3f}")
    print(f"ROC AUC: {metrics['roc_auc']:.3f}")


if __name__ == "__main__":
    main()

