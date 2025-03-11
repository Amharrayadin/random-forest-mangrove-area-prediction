# Random Forest Regressor untuk Prediksi Area

## Deskripsi
Model ini menggunakan algoritma Random Forest Regressor yang telah dilatih untuk memprediksi luas area berdasarkan variabel seperti tahun, NDVI, NIR, dan Red. Model ini telah disimpan dalam file `rfmodel.pkl` dan dapat digunakan kembali untuk melakukan prediksi pada data baru.

---

## Dependency yang Diperlukan
Sebelum menggunakan model ini, pastikan Anda telah menginstal dependensi yang diperlukan:
```bash
pip install numpy pandas
```

---

## Penggunaan Model
Model yang telah disimpan dapat digunakan kembali untuk melakukan prediksi pada data baru.

### 1. Memuat Model dan Melakukan Prediksi
Berikut adalah contoh penggunaan model untuk memprediksi nilai `area` berdasarkan input baru:
```python
import pandas as pd 
from custommodel import RandomForestRegressor as RF

# Data baru untuk prediksi
data = pd.DataFrame(data={
    'year':[2026],
    'ndvi': [4197.05],
    'nir': [1264.16],
    'red': [0.547683],
})
data['year'] = pd.to_datetime(data['year'], format='%Y')

# Memuat model dan melakukan prediksi
rf = RF.load_model('./rfmodel.pkl')
[area] = rf.predict(data)
print(f"Prediksi luas area: {area}")
```

Atribut `year` juga dapat berupa tanggal seperti berikut:
```python
import pandas as pd 
from custommodel import RandomForestRegressor as RF

# Data baru untuk prediksi
data = pd.DataFrame(data={
    'year':['2026-01-01'], # Ganti dengan tanggal
    'ndvi': [4197.05],
    'nir': [1264.16],
    'red': [0.547683],
})
data['year'] = pd.to_datetime(data['year'])

# Memuat model dan melakukan prediksi
rf = RF.load_model('./rfmodel.pkl')
[area] = rf.predict(data)
print(f"Prediksi luas area: {area}")
```

### 2. Input dan Output
| Input | Tipe Data | Deskripsi |
|--------|------------|----------------------------------------------|
| `year` | Datetime | Tahun observasi |
| `ndvi` | float | Indeks Vegetasi Normalized Difference |
| `nir` | float | Nilai reflektansi NIR |
| `red` | float | Nilai reflektansi merah |

**Output:**
- Prediksi nilai `area` (float) dalam satuan luas yang sesuai dengan data latih.

---

## Catatan
- Model harus dimuat terlebih dahulu sebelum digunakan untuk prediksi.
- Pastikan format data input sesuai dengan struktur yang digunakan saat pelatihan.
- Model ini hanya melakukan prediksi berdasarkan pola yang dipelajari dari data sebelumnya.

---
