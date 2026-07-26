# MLProject — Telco Customer Churn

MLflow Project untuk re-training model. Dipanggil oleh
`.github/workflows/ci.yml` melalui `mlflow run`.

## Entry point

Satu entry point `main` dengan enam parameter:

| Parameter | Default | Keterangan |
| --- | --- | --- |
| `data_path` | `telco_churn_preprocessing/telco_churn_preprocessed.csv` | dataset siap latih |
| `n_estimators` | 200 | jumlah pohon RandomForest |
| `max_depth` | 20 | kedalaman maksimum (`0` = tanpa batas) |
| `min_samples_split` | 2 | minimum sampel untuk split |
| `test_size` | 0.2 | proporsi data uji |
| `random_state` | 42 | seed |

## Menjalankan

```bash
mlflow run . --env-manager=local -P n_estimators=300 -P max_depth=0
```

`conda.yaml` tersedia sesuai ketentuan submission. Di CI dipakai
`--env-manager=local` karena dependensi sudah dipasang lewat `pip` pada step
sebelumnya — jauh lebih cepat daripada membangun environment conda.

## Keluaran

| Berkas | Isi |
| --- | --- |
| `mlruns/` | tracking store MLflow (parameter, metrik, artefak, model) |
| `run_id.txt` | `run_id` MLflow, dibaca step CI berikutnya |
| `ci_metrics.json` | seluruh metrik dalam satu berkas JSON |

Artefak yang dicatat: `model/` (siap `mlflow models serve` maupun
`mlflow models build-docker`), `reports/classification_report.json`,
`reports/feature_importance_top20.json`, dan `plots/confusion_matrix.png`.

## 13 metrik yang dicatat

`accuracy`, `precision`, `recall`, `f1_score`, `roc_auc`, `log_loss`,
`balanced_accuracy`, `average_precision`, `matthews_corrcoef`, `specificity`,
`ks_statistic`, `train_accuracy`, `fit_time_seconds`.

Skrip memuat assertion bahwa akurasi harus melampaui baseline kelas mayoritas
(0,7346), sehingga CI gagal bila model hasil re-training tidak lebih baik daripada
sekadar menebak kelas terbanyak.
