# What-if Simulator: Estimasi Engagement Konten Instagram Bank Sampah Digital

Aplikasi Streamlit untuk mengestimasi Total Engagement sebuah konten sebelum dipublikasikan
dan membandingkan dua skenario atribut (format, tipe, topik, hari, jam, panjang caption, hashtag).

## Isi folder
- app.py : aplikasi Streamlit
- app_artifacts.pkl : model Random Forest (pipeline lengkap) beserta rentang ketidakpastian dan opsi input
- requirements.txt : dependensi (versi sama dengan saat model dilatih)
- .streamlit/config.toml : konfigurasi tampilan dan server

## Menjalankan lokal
    pip install -r requirements.txt
    streamlit run app.py

## Deploy (Streamlit Community Cloud)
1. Upload seluruh isi folder ini ke repository GitHub.
2. Di share.streamlit.io pilih repository, branch, dan main file: app.py
3. Pada Advanced settings pilih Python 3.13 (sama dengan Colab).

## Catatan
Model dilatih pada data kecil, hasil dibaca sebagai rentang kemungkinan, bukan prediksi pasti.
Versi scikit-learn harus sama dengan saat model disimpan, jika tidak file .pkl gagal dimuat.
