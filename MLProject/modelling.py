"""Entry point MLflow Project untuk re-training model Telco Churn di CI.

Dipanggil oleh ``mlflow run`` (lihat file ``MLProject``). Hyperparameter diterima
sebagai argumen CLI sehingga bisa diubah dari workflow tanpa menyentuh kode.

Model dicatat dengan ``artifact_path="model"`` agar bisa langsung dipakai oleh
``mlflow models build-docker -m runs:/<run_id>/model``.
"""

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

# MLflow mencetak emoji saat menutup run; runner Windows / konsol cp1252 error
# tanpa penyesuaian ini.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
TARGET = "Churn"
EXPERIMENT_NAME = "Telco Churn - CI Retraining"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data_path",
        default="telco_churn_preprocessing/telco_churn_preprocessed.csv",
    )
    p.add_argument("--n_estimators", type=int, default=200)
    # max_depth <= 0 diterjemahkan menjadi None (pohon tumbuh penuh).
    p.add_argument("--max_depth", type=int, default=20)
    p.add_argument("--min_samples_split", type=int, default=2)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--random_state", type=int, default=42)
    return p.parse_args()


def load_data(data_path, test_size, random_state):
    path = Path(data_path)
    if not path.is_absolute() and not path.exists():
        path = HERE / data_path
    df = pd.read_csv(path)
    print(f"[data] {path} -> {df.shape}")

    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def main():
    args = parse_args()
    max_depth = args.max_depth if args.max_depth and args.max_depth > 0 else None

    # `mlflow run` sudah membuat run aktif dan mengekspor MLFLOW_RUN_ID. Memanggil
    # set_experiment() pada kondisi itu membuat experiment aktif tidak cocok dengan
    # run bawaan sehingga start_run() gagal. Jadi experiment hanya diatur saat skrip
    # dijalankan langsung; lewat MLflow Project namanya diberikan via
    # `--experiment-name` pada perintah `mlflow run`.
    running_as_project = "MLFLOW_RUN_ID" in os.environ
    if not running_as_project:
        mlflow.set_experiment(EXPERIMENT_NAME)

    X_train, X_test, y_train, y_test = load_data(
        args.data_path, args.test_size, args.random_state
    )

    # Autolog tidak dipakai agar seluruh pencatatan eksplisit dan stabil di CI.
    # run_name diatur lewat tag supaya aman baik saat membuat run baru maupun
    # saat melanjutkan run bawaan `mlflow run`.
    with mlflow.start_run() as run:
        mlflow.set_tag("mlflow.runName", "ci_retraining")
        mlflow.log_params(
            {
                "model_type": "RandomForestClassifier",
                "n_estimators": args.n_estimators,
                "max_depth": max_depth,
                "min_samples_split": args.min_samples_split,
                "test_size": args.test_size,
                "random_state": args.random_state,
                "n_features": X_train.shape[1],
                "n_train_rows": len(X_train),
                "n_test_rows": len(X_test),
            }
        )

        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=max_depth,
            min_samples_split=args.min_samples_split,
            random_state=args.random_state,
            n_jobs=-1,
        )
        start = time.perf_counter()
        model.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - start

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        fpr, tpr, _ = roc_curve(y_test, y_proba)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "log_loss": log_loss(y_test, y_proba),
            "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
            "average_precision": average_precision_score(y_test, y_proba),
            "matthews_corrcoef": matthews_corrcoef(y_test, y_pred),
            "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
            "ks_statistic": float(np.max(tpr - fpr)),
            "train_accuracy": accuracy_score(y_train, model.predict(X_train)),
            "fit_time_seconds": fit_seconds,
        }
        mlflow.log_metrics(metrics)
        for name, value in metrics.items():
            print(f"[metric] {name:20s} = {value:.4f}")

        mlflow.set_tags(
            {
                "author": "yusuf_jatikus47rz",
                "dataset": "Telco Customer Churn",
                "stage": "kriteria-3-ci",
                "trigger": "github-actions",
                "python_version": platform.python_version(),
            }
        )

        # Artefak: laporan klasifikasi, confusion matrix, dan feature importance.
        mlflow.log_dict(
            classification_report(
                y_test, y_pred, output_dict=True, target_names=["No Churn", "Churn"]
            ),
            "reports/classification_report.json",
        )

        fig, ax = plt.subplots(figsize=(4.6, 4))
        im = ax.imshow(confusion_matrix(y_test, y_pred), cmap="Blues")
        for (i, j), v in np.ndenumerate(confusion_matrix(y_test, y_pred)):
            ax.text(j, i, str(v), ha="center", va="center")
        ax.set_xticks([0, 1], ["No Churn", "Churn"])
        ax.set_yticks([0, 1], ["No Churn", "Churn"])
        ax.set_xlabel("Prediksi"); ax.set_ylabel("Aktual")
        ax.set_title("Confusion Matrix")
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        mlflow.log_figure(fig, "plots/confusion_matrix.png")
        plt.close(fig)

        importance = (
            pd.DataFrame(
                {"feature": X_train.columns, "importance": model.feature_importances_}
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
        mlflow.log_dict(
            importance.head(20).to_dict(orient="records"),
            "reports/feature_importance_top20.json",
        )

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=X_train.head(5),
            signature=mlflow.models.infer_signature(X_train, y_pred[:1]),
        )

        run_id = run.info.run_id
        print(f"[run] run_id={run_id}")
        # Ditulis ke file agar step CI berikutnya bisa membacanya tanpa parsing log.
        Path("run_id.txt").write_text(run_id)
        Path("ci_metrics.json").write_text(json.dumps(metrics, indent=2))

    baseline = 1 - float(y_test.mean())
    assert metrics["accuracy"] > baseline, (
        f"akurasi {metrics['accuracy']:.4f} tidak melampaui baseline mayoritas {baseline:.4f}"
    )
    print("[ok] re-training selesai")


if __name__ == "__main__":
    main()
