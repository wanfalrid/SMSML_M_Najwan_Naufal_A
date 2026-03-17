# Panduan Screenshot untuk Submission Dicoding

**Student:** M_Najwan_Naufal_A  
**Course:** Membangun Sistem Machine Learning — Dicoding

---

## Screenshot yang Dibutuhkan

Untuk submission K2 (Membangun Model), kamu perlu 2 screenshot:

### 1. `screenshoot_dashboard.jpg` — MLflow Dashboard di DagsHub

**Langkah:**
1. Buka browser, navigasi ke DagsHub repo:
   ```
   https://dagshub.com/najwanopal/wine-quality-mlops
   ```
2. Klik tab **"Experiments"** (atau langsung ke MLflow UI)
3. Kamu akan melihat MLflow dashboard dengan experiment `wine-quality-classification`
4. Pastikan terlihat:
   - ✅ Nama experiment
   - ✅ Daftar runs (termasuk `baseline-random-forest`)
   - ✅ Metrics (accuracy, f1, precision, recall, roc_auc)
   - ✅ Parameters (model parameters)
5. Screenshot seluruh halaman dashboard
6. Simpan sebagai `screenshoot_dashboard.jpg`

**Contoh tampilan yang harus terlihat:**
```
┌─────────────────────────────────────────────────────────────┐
│  DagsHub > najwanopal/wine-quality-mlops > Experiments      │
│                                                             │
│  Experiment: wine-quality-classification                    │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ Run Name              │ Accuracy │ F1    │ ROC AUC     ││
│  │ baseline-random-forest│ 0.93xx  │ 0.7xxx│ 0.92xx      ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

---

### 2. `screenshoot_artifak.jpg` — MLflow Artifacts di DagsHub

**Langkah:**
1. Dari MLflow dashboard, klik pada run `baseline-random-forest`
2. Scroll ke bawah ke bagian **"Artifacts"**
3. Kamu akan melihat tree artifacts:
   ```
   artifacts/
   ├── model/
   │   ├── MLmodel
   │   ├── model.pkl
   │   ├── conda.yaml
   │   └── requirements.txt
   ├── plots/
   │   ├── confusion_matrix.png
   │   ├── feature_importance.png
   │   └── roc_curve.png
   ├── reports/
   │   └── classification_report.txt
   └── metadata/
       └── baseline_metadata.json
   ```
4. Pastikan terlihat:
   - ✅ Folder `model/` dengan file MLmodel
   - ✅ Folder `plots/` dengan visualisasi
   - ✅ Folder `reports/` dengan classification report
5. Screenshot seluruh halaman artifacts
6. Simpan sebagai `screenshoot_artifak.jpg`

---

## Cara Mengambil Screenshot

### Windows
- **Shortcut:** `Win + Shift + S` → pilih area → paste di Paint → Save as JPG
- **Full screen:** `PrtScn` → paste di Paint → Save as JPG
- **Tool:** Snipping Tool / Snip & Sketch

### Simpan File di:
```
Membangun_model/
├── screenshoot_dashboard.jpg    ← Screenshot MLflow dashboard
└── screenshoot_artifak.jpg      ← Screenshot MLflow artifacts
```

---

## Checklist Sebelum Submit

- [ ] `screenshoot_dashboard.jpg` menunjukkan MLflow experiment dengan metrics
- [ ] `screenshoot_artifak.jpg` menunjukkan artifacts (model, plots, reports)
- [ ] Kedua screenshot jelas dan bisa dibaca (resolusi cukup)
- [ ] Nama file sesuai (perhatikan typo "screenshoot" bukan "screenshot" — ikuti format submission Dicoding)
