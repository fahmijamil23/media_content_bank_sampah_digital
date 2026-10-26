import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title='What-if Simulator Konten', layout='wide')

@st.cache_resource
def muat():
    return joblib.load('app_artifacts.pkl')

A = muat()
model = A['model']
HARI = {'Senin': 0, 'Selasa': 1, 'Rabu': 2, 'Kamis': 3, 'Jumat': 4, 'Sabtu': 5, 'Minggu': 6}
Q = A['kuartil']


def kategori(x):
    return 'Rendah' if x < Q['Q1'] else ('Tinggi' if x > Q['Q3'] else 'Menengah')

def prediksi(s):
    p = model.predict(pd.DataFrame([s])[A['fitur']])[0]
    return {'est': float(np.expm1(p)),
            'lo': float(np.expm1(p + A['q_low'])),
            'hi': float(np.expm1(p + A['q_high']))}

def idx(daftar, nilai):
    return daftar.index(nilai) if nilai in daftar else 0

def form(k):
    d, o, r = A['default'], A['opsi'], A['rentang']
    fmt = st.selectbox('Format konten', o['Content_Format'],
                       index=idx(o['Content_Format'], d['Content_Format']), key=f'fmt_{k}')
    tipe = st.selectbox('Tipe konten', o['Content_Type'],
                        index=idx(o['Content_Type'], d['Content_Type']), key=f'tipe_{k}')
    topik = st.selectbox('Topik', o['Topik'],
                         index=idx(o['Topik'], d['Topik']), key=f'topik_{k}')
    hari = st.selectbox('Hari posting', list(HARI), index=2, key=f'hari_{k}')
    slot_post = st.selectbox('Waktu posting', o['Time_Slot'],
                             index=idx(o['Time_Slot'], d['Time_Slot']), key=f'slot_{k}')
    st.caption('Pagi < 11.00 | Siang 11.00-14.59 | Sore 15.00-17.59 | Malam ≥ 18.00')
    kata = st.slider('Jumlah kata caption', r['Caption_Word_Count'][0], r['Caption_Word_Count'][1],
                     d['Caption_Word_Count'], key=f'kata_{k}')
    tag = st.slider('Jumlah hashtag', r['Hashtag_Count'][0], r['Hashtag_Count'][1],
                    d['Hashtag_Count'], key=f'tag_{k}')
    return {'Content_Format': fmt, 'Content_Type': tipe, 'Topik': topik,
            'Caption_Word_Count': kata, 'Day_of_Week': HARI[hari],
            'Is_Weekend': int(HARI[hari] >= 5), 'Time_Slot': slot_post, 'Hashtag_Count': tag}

st.title('What-if Simulator: Estimasi Engagement Konten')
m = A['metrik']
st.info(f"Model dilatih pada {A['n_data']} postingan. Error rata-rata (skala log) model "
        f"{m['MAE_log_model']:.2f} vs tebakan median {m['MAE_log_baseline']:.2f}, jadi model belum "
        f"jauh lebih baik dari tebakan sederhana. Gunakan hasil ini sebagai rentang kemungkinan, bukan angka pasti.")

ca, cb = st.columns(2)
with ca:
    st.subheader('Skenario A (dasar)')
    sa = form('a')
with cb:
    st.subheader('Skenario B (diubah)')
    sb = form('b')

ra, rb = prediksi(sa), prediksi(sb)
delta = (rb['est'] / ra['est'] - 1) * 100
tumpang = not (rb['lo'] > ra['hi'] or ra['lo'] > rb['hi'])

st.divider()
ma, mb = st.columns(2)
with ma:
    st.metric('Estimasi engagement A', f"{ra['est']:,.0f}")
    st.caption(f"Rentang 80%: {ra['lo']:,.0f} - {ra['hi']:,.0f} | Kategori: {kategori(ra['est'])}")
with mb:
    st.metric('Estimasi engagement B', f"{rb['est']:,.0f}", delta=f'{delta:+.1f}%')
    st.caption(f"Rentang 80%: {rb['lo']:,.0f} - {rb['hi']:,.0f} | Kategori: {kategori(rb['est'])}")

if tumpang:
    st.warning('Rentang A dan B saling tumpang tindih: selisih ini masih dalam batas ketidakpastian model.')
else:
    st.success('Rentang A dan B tidak tumpang tindih: selisihnya relatif lebih meyakinkan.')

# ---------- Tips Konten Berpeluang Viral ----------
V = A['viral']
st.divider()
st.header('Tips Konten Berpeluang Viral')
st.caption(f"Viral = masuk 10% engagement tertinggi (≥ {V['batas']:,.0f} interaksi): "
           f"{V['n_viral']} dari {V['n_total']} postingan. Peluang rata-rata 10%. "
           f"Datanya sedikit, jadi ini pola, bukan jaminan.")

kolom = st.columns(len(V['ringkas']))
for c, r in zip(kolom, V['ringkas'].itertuples()):
    c.metric(r.Aspek, r.Terbaik)
    c.caption(f"{r.Persen_Viral:.0f}% postingannya viral (dari {r.Jumlah} postingan)")

unggul = V['ringkas'][V['ringkas']['Persen_Viral'] > 10]
if len(unggul):
    st.success('Paling sering muncul di konten viral: ' +
               ', '.join(f"{r.Aspek.lower()} {r.Terbaik}" for r in unggul.itertuples()) + '.')
else:
    st.info('Belum ada atribut yang jelas menonjol di data ini.')

with st.expander('Lihat detail per atribut'):
    atribut = st.selectbox('Atribut', V['profil']['Atribut'].unique().tolist())
    tabel = V['profil'][V['profil']['Atribut'] == atribut].drop(columns='Atribut')
    st.dataframe(tabel, use_container_width=True, hide_index=True)
