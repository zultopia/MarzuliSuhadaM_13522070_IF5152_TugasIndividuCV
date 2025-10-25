# Tugas Individu Computer Vision - Marzuli Suhada M (13522070)

## Deskripsi Aplikasi

Aplikasi Computer Vision integratif yang mengimplementasikan materi minggu 3-6:
- **Image Filtering**: Gaussian, Median, Sobel
- **Edge Detection & Sampling**: Canny, Sobel, Multi-scale
- **Feature/Interest Points**: Harris, SIFT, FAST
- **Camera Geometry & Calibration**: Homography, Affine Transform

---

## Instalasi dan Setup

### Persyaratan
- Python 3.8 atau lebih baru
- RAM minimal 4GB (Direkomendasikan 8GB untuk gambar high-resolution)

### Instalasi Dependencies

**Install dari requirements.txt:**
```bash
pip install -r requirements.txt
```

**Atau install manual:**
```bash
pip install numpy opencv-python scikit-image matplotlib pandas pillow
```

**Verifikasi instalasi:**
```bash
python -c "import numpy, cv2, skimage, matplotlib, pandas; print('All dependencies installed!')"
```

---

## Cara Menjalankan Aplikasi

### **PENTING: Perbedaan main.py vs main_pipeline.py**

| Aspek | main.py | main_pipeline.py |
|-------|---------|------------------|
| **Tipe** | Independent & Parallel | Sequential & Integrated |
| **Dependency** | Tidak ada | Ada chain (Filtering→Edge→Feature→Geometry) |
| **Output** | 4 folder terpisah dengan analisis parameter lengkap | 1 folder dengan visualisasi pipeline |
| **Waktu** | 15-45 menit | 3-8 menit |
| **Tujuan** | Eksplorasi parameter mendalam | Demo integrasi pipeline |

---

### **Opsi 1: main.py** - Analisis Parameter Lengkap

**Menjalankan:**
```bash
python main.py
```

**Karakteristik:**
- Setiap modul (filtering, edge, feature, geometry) berjalan **independen**
- Tidak ada dependency antar modul
- Menghasilkan analisis parameter yang **komprehensif**
- Cocok untuk **eksplorasi parameter** dan **analisis mendalam**

**Output:**
```
01_filtering/output/       → Gaussian, Median, Sobel dengan berbagai parameter
02_edge/output/           → Sobel, Canny, Sampling dengan berbagai threshold
03_featurepoints/output/  → Harris, SIFT, FAST dengan berbagai parameter
04_geometry/output/       → Homography, Affine untuk semua gambar
SUMMARY_REPORT.md         → Laporan ringkasan eksekusi
```

---

### **Opsi 2: main_pipeline.py** - Pipeline Terintegrasi (Direkomendasikan)

**Menjalankan:**
```bash
python main_pipeline.py
```

**Karakteristik:**
- Proses **sequential dengan dependency chain**:
  ```
  Raw Image → Filtering → Edge Detection → Feature Points → Geometry
  ```
- Output satu step = Input untuk step berikutnya
- Menghasilkan **visualisasi pipeline 6-panel**
- Cocok untuk **demo dan memahami workflow**

**Output:**
```
pipeline_output/
  ├── *_pipeline.png          → Visualisasi 6-panel untuk setiap gambar
  └── pipeline_analysis.csv   → Statistik setiap stage
```

**Visualisasi 6-Panel:**
1. Original (Colorized)
2. Step 1: Filtered (Gaussian)
3. Step 2: Edges (Canny dari filtered)
4. Step 3: Features (Harris pada edges)
5. Step 4: Geometry (Homography dari features)
6. Pipeline Flow Diagram

---

### **Opsi 3: Menjalankan Modul Individual**

```bash
# Image Filtering
cd 01_filtering && python filtering.py

# Edge Detection
cd 02_edge && python edge_detection.py

# Feature Points
cd 03_featurepoints && python feature_points.py

# Geometry & Calibration
cd 04_geometry && python geometry_calibration.py
```

---

## Struktur Output

```
Marzuli_Suhada_M_13522070_IF5152_TugasIndividuCV/
├── 01_filtering/output/
│   ├── *_gaussian_comparison.png
│   ├── *_median_comparison.png
│   ├── *_sobel_comparison.png
│   └── filtering_parameters.csv
│
├── 02_edge/output/
│   ├── *_sobel_edge_comparison.png
│   ├── *_canny_edge_comparison.png
│   ├── *_sampling_sobel_edge_comparison.png
│   └── edge_detection_analysis.csv
│
├── 03_featurepoints/output/
│   ├── *_harris_features.png
│   ├── *_sift_features.png
│   ├── *_fast_features.png
│   └── feature_points_statistics.csv
│
├── 04_geometry/output/
│   ├── *_homography.png
│   ├── *_affine.png
│   ├── *_homography_matrix.txt
│   ├── *_affine_matrix.txt
│   └── geometry_calibration_analysis.csv
│
├── pipeline_output/
│   ├── *_pipeline.png
│   └── pipeline_analysis.csv
│
├── gambar_pribadi/
│   ├── azul.png
│   └── madam_eva.png
│
├── main.py
├── main_pipeline.py
└── 05_laporan.pdf
```

---

## Cara Melihat Hasil

### Visualisasi Gambar
**Cara termudah:** Buka folder `output/` dengan file explorer, double-click file PNG.

**Via terminal:**
```bash
# macOS
open pipeline_output/astronaut_pipeline.png

# Linux
xdg-open pipeline_output/astronaut_pipeline.png

# Windows
start pipeline_output/astronaut_pipeline.png
```

### Analisis CSV
```python
import pandas as pd
df = pd.read_csv('pipeline_output/pipeline_analysis.csv')
print(df)
```

---

## Dataset yang Digunakan

### Gambar Standar (dari scikit-image)
1. **cameraman** - 512×512, grayscale
2. **coin** - 303×384, grayscale
3. **checkerboard** - 200×200, binary pattern
4. **astronaut** - 512×512, RGB

### Gambar Pribadi
1. **azul** - 4032×3024, high-resolution portrait
2. **madam_eva** - 1332×970, portrait

**Total:** 6 gambar

---

## Error, Kendala, dan Troubleshooting

### 1. **ImportError: No module named 'cv2'**
```bash
pip uninstall opencv-python opencv-python-headless
pip install opencv-python
```

### 2. **MemoryError pada gambar high-resolution**
**Solusi:** Resize gambar pribadi sebelum processing
```python
from PIL import Image
img = Image.open('gambar_pribadi/azul.png')
img.resize((2016, 1512)).save('gambar_pribadi/azul_resized.png')
```

### 3. **Processing terlalu lama (> 1 jam)**
**Solusi:** Gunakan `main_pipeline.py` yang lebih cepat (3-8 menit) atau jalankan modul satu per satu.

### 4. **Checkerboard detection gagal pada gambar pribadi**
**Status:** Sudah ditangani otomatis. Script membuat synthetic checkerboard untuk calibration demo.

### 5. **No display available (SSH/server tanpa GUI)**
```bash
export MPLBACKEND=Agg
python main_pipeline.py
```

### 6. **FileNotFoundError: gambar_pribadi/**
```bash
mkdir -p gambar_pribadi
# Copy gambar pribadi Anda ke folder tersebut
```

**Note:** Jika tidak ada gambar pribadi, script tetap berjalan dengan 4 gambar standar saja.

---

## Fitur Unik Implementasi

1. **Dual Visualization** - Menampilkan original color + grayscale untuk transparansi
2. **Parameter Analysis** - Systematic parameter grid search dengan statistik lengkap
3. **CSV Export** - Semua hasil tersimpan untuk post-processing dan reproducibility
4. **Integrated Pipeline** - Sequential processing dengan visualisasi aliran data
5. **Adaptive Processing** - Handle berbagai resolusi (200×200 hingga 4032×3024)
6. **Memory-Efficient** - Chunked processing untuk gambar high-resolution

---

## Performance

| Mode | Waktu Eksekusi | Output |
|------|----------------|--------|
| main_pipeline.py | 3-8 menit | 6 visualisasi pipeline + 1 CSV |
| main.py (gambar standar) | 10-15 menit | ~100 file output + 4 CSV |
| main.py (dengan gambar pribadi) | 30-45 menit | ~160 file output + 4 CSV |
| Modul individual | 2-10 menit/modul | Tergantung modul |

---

## Quick Start

### Untuk Demo Cepat (5 menit):
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Jalankan pipeline
python main_pipeline.py

# 3. Lihat hasil
open pipeline_output/astronaut_pipeline.png
```

### Untuk Analisis Lengkap:
```bash
# Jalankan semua modul
python main.py

# Periksa output
ls -la 01_filtering/output/
ls -la 02_edge/output/
ls -la 03_featurepoints/output/
ls -la 04_geometry/output/
```

---

## Kontak

**Nama:** Marzuli Suhada M  
**NIM:** 13522070  
**Mata Kuliah:** IF5152 Computer Vision  
**Email:** 13522070@std.stei.itb.ac.id

---

*Last Updated: Oktober 2024*
