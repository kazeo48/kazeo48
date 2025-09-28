# Latihan Codex
Repo percobaan pertama untuk belajar GitHub + ChatGPT Codex

## Modul Machine Learning: Klasifikasi Pengambilan Les Matematika

Proyek ini menyediakan modul sederhana untuk membuat dataset sintetik dan
melatih model klasifikasi yang memprediksi apakah seorang siswa mengikuti les
matematika tambahan.

### Struktur Proyek

- `ml/math_tutoring_classifier.py` – Fungsi utilitas untuk membuat dataset dan
  pipeline model.
- `generate_datasets.py` – Skrip untuk membuat dataset training (1000 baris)
  dan testing (100 baris).
- `train_model.py` – Skrip untuk melatih model dan mengevaluasi performa pada
  dataset testing.
- `data/train.csv` & `data/test.csv` – Dataset hasil generate (dengan 1000 dan
  100 baris).

### Cara Menggunakan

1. Pasang dependensi:

   ```bash
   pip install -r requirements.txt
   ```

2. Generate dataset (opsional jika ingin regenerasi):

   ```bash
   python generate_datasets.py
   ```

3. Latih dan evaluasi model:

   ```bash
   python train_model.py
   ```
  
