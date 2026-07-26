# Workflow-CI

Kriteria 3 submission **Membangun Sistem Machine Learning** — MLflow Project dan
workflow CI untuk re-training otomatis model Telco Customer Churn.

## Struktur

```
Workflow-CI
├── .github/workflows/ci.yml
└── MLProject/
    ├── MLProject                       # definisi MLflow Project
    ├── conda.yaml                      # environment Python 3.12.7
    ├── modelling.py                    # entry point re-training
    ├── telco_churn_preprocessing/      # dataset siap latih
    ├── DockerHub.txt                   # tautan Docker Hub
    └── README.md
```

## Secrets yang diperlukan

`Settings > Secrets and variables > Actions`:

| Secret | Isi |
| --- | --- |
| `DOCKERHUB_USERNAME` | username Docker Hub |
| `DOCKERHUB_TOKEN` | Access Token Docker Hub |

Selama secrets belum diisi, keempat step Docker otomatis dilewati (`if:
env.DOCKERHUB_USERNAME != ''`) sehingga workflow tetap hijau dan tahap
Basic + Skilled tetap terpenuhi.

## Tahapan workflow

| # | Step | Fungsi |
| --- | --- | --- |
| 1 | Run actions/checkout@v4 | ambil kode |
| 2 | Set up Python 3.12.7 | versi yang direkomendasikan Dicoding |
| 3 | Check Env | cetak versi & isi folder untuk debugging |
| 4 | Install dependencies | mlflow 2.19.0 + scikit-learn 1.6.0 |
| 5 | Run mlflow project | `mlflow run . --env-manager=local` |
| 6 | Get latest MLflow run_id | baca `run_id.txt` yang ditulis `modelling.py` |
| 7 | Upload to GitHub | artefak via `actions/upload-artifact` |
| 8 | Commit artifacts back to repository | `mlruns/` masuk ke branch `main` |
| 9 | Log in to Docker Hub | butuh secrets |
| 10 | Build Docker Model | `mlflow models build-docker` |
| 11 | Tag Docker Image | `:latest` dan `:<run_id>` |
| 12 | Push Docker Image | publikasi ke Docker Hub |

Pemicu: `push` ke `main`, `pull_request`, jadwal Senin 01:00 UTC, dan
`workflow_dispatch` dengan input `n_estimators` serta `max_depth`.

## Menjalankan secara lokal

```bash
pip install mlflow==2.19.0 scikit-learn==1.6.0 pandas==2.2.3 numpy==2.2.1
cd MLProject
mlflow run . --env-manager=local --experiment-name "Telco Churn - CI Retraining" -P n_estimators=200 -P max_depth=20
```

Hasil yang sudah diverifikasi:

| Metrik | Nilai |
| --- | --- |
| accuracy | 0,7899 |
| precision | 0,6373 |
| recall | 0,4840 |
| f1_score | 0,5502 |
| roc_auc | 0,8185 |
| balanced_accuracy | 0,6922 |
| specificity | 0,9005 |
| ks_statistic | 0,4877 |

`run_id`: `bf782e5c313343aea618e295f6a05e1c`

## Catatan implementasi

**`MLFLOW_TRACKING_URI` di CI.** Workflow menyetel
`file://${{ github.workspace }}/MLProject/mlruns`. Ini valid di runner Linux
karena `github.workspace` diawali `/` sehingga menghasilkan `file:///...`.
Di Windows bentuk `file://D:/...` akan ditolak MLflow (`D:` dianggap host
remote) — jalankan tanpa variabel tersebut agar memakai default `./mlruns`.

**Autolog tidak dipakai.** `modelling.py` mencatat semuanya secara eksplisit agar
stabil di CI dan tidak bergantung pada kompatibilitas versi scikit-learn.

**Experiment di MLflow Project.** `mlflow run` sudah membuat run aktif dan
mengekspor `MLFLOW_RUN_ID`. Memanggil `mlflow.set_experiment()` pada kondisi itu
membuat experiment aktif tidak cocok dengan run bawaan dan `start_run()` gagal.
Karena itu `modelling.py` hanya memanggil `set_experiment()` bila dijalankan
langsung, dan nama experiment diberikan lewat `--experiment-name` saat memakai
`mlflow run`. Nama run diatur lewat tag `mlflow.runName` supaya aman di kedua jalur.
