# 📦 Langkah-langkah penyiapan projek

## 📋 **Persyaratan**

Pastikan Anda sudah menginstal:

- **Python** (versi 3.12 atau lebih baru)
- **Git**

## 🛠️ **Penyiapan**

- Untuk memeriksa versi Python:

```bash
python --version
```

- Untuk memeriksa apakah git sudah terinstall:

```bash
git --version
```

- Salin projek ini dari GitHub dengan perintah berikut:

```bash
git clone https://github.com/avrzll/sistem-rekomendasi-ds.git
```

- Masuk ke folder projek:

```bash
cd sistem-rekomendasi-ds
```

- Siapkan Virtual Enviroment:

```bash
# untuk windows
python -m venv venv

# untuk macOs atau Linux
python3 -m venv venv
```

- Aktifkan Virtual Enviroment:

```bash
# untuk windows
venv\Scripts\activate

# untuk macOs atau Linux
source venv/bin/activate
```

- Install semua depedensi yang diperlukan:

```bash
pip install -r requirements.txt
```

- Jalankan aplikasi

```bash
streamlit run app.py
```

- Untuk keluar dari Virtual Enviroment

```bash
deactivate
```
