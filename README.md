# Analisis Kualitas Udara Beijing

## Deskripsi Proyek

Proyek ini merupakan analisis data kualitas udara di Beijing, China, menggunakan dataset dari 12 stasiun pemantauan selama periode Maret 2013 hingga Februari 2017.

Analisis dilakukan untuk mengidentifikasi pola konsentrasi polutan PM2.5, membandingkan tingkat pencemaran antarstasiun, menganalisis hubungan antara polutan dan faktor meteorologi, serta melihat distribusi kualitas udara berdasarkan lokasi geografis.

Hasil analisis disajikan melalui visualisasi data dan dashboard interaktif menggunakan Streamlit.

## Fitur Dashboard

Dashboard menyediakan beberapa fitur analisis, antara lain:

- **Analisis Temporal:** Mengamati perubahan konsentrasi PM2.5 berdasarkan tahun, bulan, musim, dan jam.
- **Analisis Antarstasiun:** Membandingkan rata-rata konsentrasi PM2.5 dan tingkat pencemaran pada 12 stasiun pemantauan.
- **Analisis Korelasi:** Mengidentifikasi hubungan antara konsentrasi polutan dengan faktor meteorologi.
- **Analisis Spasial:** Menampilkan persebaran konsentrasi PM2.5 berdasarkan lokasi geografis stasiun.
- **Visualisasi Interaktif:** Menyajikan grafik, tabel, dan peta yang dapat disesuaikan menggunakan filter dashboard.

## Struktur Direktori

```
fundamental-analisis-data/
│
├── dashboard/
│   ├── dashboard.py
│   └── main_data.csv
│
├── data/
│   ├── PRSA_Data_Aotizhongxin_20130301-20170228.csv
│   ├── PRSA_Data_Changping_20130301-20170228.csv
│   ├── PRSA_Data_Dingling_20130301-20170228.csv
│   ├── PRSA_Data_Dongsi_20130301-20170228.csv
│   ├── PRSA_Data_Guanyuan_20130301-20170228.csv
│   ├── PRSA_Data_Gucheng_20130301-20170228.csv
│   ├── PRSA_Data_Huairou_20130301-20170228.csv
│   ├── PRSA_Data_Nongzhanguan_20130301-20170228.csv
│   ├── PRSA_Data_Shunyi_20130301-20170228.csv
│   ├── PRSA_Data_Tiantan_20130301-20170228.csv
│   ├── PRSA_Data_Wanliu_20130301-20170228.csv
│   └── PRSA_Data_Wanshouxigong_20130301-20170228.csv
│
├── notebook.ipynb
├── requirements.txt
├── README.md
└── url.txt
```

**Catatan:** File `main_data.csv` wajib berada satu folder dengan `dashboard.py` agar aplikasi dapat membaca dataset dengan benar.

## Installation

### 1. Clone Repository

```bash
git clone <URL_REPOSITORY>
cd fundamental-analisis-data
```

### 2. Membuat Environment Menggunakan Miniconda

Pastikan Miniconda sudah terinstal pada komputer.

```bash
conda create --name main-ds python=3.9 -y
```

Aktifkan environment:

```bash
conda activate main-ds
```

### 3. Install Dependencies

Pastikan file `requirements.txt` tersedia di direktori utama proyek.

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Menjalankan Streamlit Dashboard

Jalankan perintah berikut dari direktori utama proyek:

```bash
streamlit run dashboard/dashboard.py
```

Setelah berhasil dijalankan, dashboard dapat diakses melalui browser pada alamat lokal:

```
http://localhost:8501
```
