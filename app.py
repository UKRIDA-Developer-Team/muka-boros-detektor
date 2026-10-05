"""Demo lokal; foto unggahan diproses di memori dan tidak ditulis ke disk."""
from pathlib import Path

import streamlit as st

from faceage.inference import Predictor, available_models, decode_image
from faceage.io import read_json

ROOT=Path(__file__).resolve().parent
st.set_page_config(page_title='Estimasi Usia Wajah',page_icon='🧑',layout='wide')
st.markdown('''<style>
.block-container {max-width: 1120px; padding-top: 2.6rem;}
h1 {letter-spacing: -.03em;}
[data-testid="stMetricValue"] {color: #176b64;}
</style>''',unsafe_allow_html=True)


@st.cache_resource
def get_predictor(path, modified_ns):
    return Predictor(path,device='cpu')


def main():
    st.caption('PROYEK VISI KOMPUTER · UTKFACE · MOBILENETV3 SMALL')
    st.title('Estimasi usia dari wajah')
    st.write('Unggah satu wajah yang sudah dipotong untuk melihat perkiraan usia dari model penelitian.')
    models=available_models(ROOT/'runs')
    with st.sidebar:
        st.subheader('Model eksperimen')
        selection=st.selectbox('Pilih model',models,format_func=lambda value:
                               f'{value["name"]} · {value["metadata"]["method"].upper()}',
                               disabled=not models,index=0 if models else None)
        if selection:
            metadata=selection['metadata']
            st.caption('Checkpoint terbaik berdasarkan MAE validasi.')
            st.write(f'MAE validasi: **{metadata["best_validation_mae"]:.2f} tahun**')
            st.write(f'Jumlah data latih: **{metadata["data_counts"]["train"]:,} citra**')
        st.divider()
        st.caption('Estimasi terhadap label dataset, bukan penentuan umur resmi. Gunakan foto yang penggunaannya kamu izinkan.')
    estimation,results,about=st.tabs(['Estimasi','Hasil eksperimen','Tentang proyek'])
    with estimation:
        if not models:
            st.info('Model estimasi usia belum tersedia. Jalankan pelatihan UTKFace sesuai README, lalu buka kembali demo ini.')
        left,right=st.columns([1.05,1],gap='large')
        image=None
        with left:
            uploaded=st.file_uploader('Foto wajah',type=['jpg','jpeg','png'],max_upload_size=10,
                                      help='Satu wajah yang sudah dipotong; maksimal 10 MB.')
            if uploaded:
                try:
                    image=decode_image(uploaded.getvalue())
                    st.image(image,caption='Foto input',width='stretch')
                except (ValueError,OSError) as exc: st.error(f'Foto tidak dapat dibaca: {exc}')
        with right:
            st.subheader('Perkiraan usia')
            st.write('Gunakan foto yang jelas, pencahayaan cukup, dan wajah menghadap kamera.')
            if st.button('Estimasi usia',type='primary',disabled=image is None or selection is None,width='stretch'):
                try:
                    with st.spinner('Membaca pola visual wajah...'):
                        checkpoint=selection['path']
                        predictor=get_predictor(str(checkpoint),checkpoint.stat().st_mtime_ns)
                        prediction=predictor.predict(image)
                    age,latency=st.columns(2)
                    age.metric('Usia estimasi',f'{prediction["display_age"]:.1f} tahun')
                    latency.metric('Waktu inferensi',f'{prediction["latency_ms"]:.1f} ms')
                    st.caption('Waktu praproses dan inferensi CPU setelah model dimuat, diukur untuk unggahan ini.')
                    if prediction['method']=='b0':
                        st.info('B0 selalu menghasilkan median usia data latih. Baseline ini tidak menganalisis ciri wajah.')
                    if prediction['raw_age']!=prediction['display_age']:
                        st.caption('Tampilan dibulatkan dan dibatasi ke 0-116 tahun; evaluasi memakai angka mentah.')
                except (OSError,ValueError,RuntimeError,KeyError) as exc:
                    st.error(f'Inferensi tidak dapat dijalankan: {exc}')
            elif image is None:
                st.caption('Hasil akan muncul setelah kamu mengunggah foto dan menekan tombol estimasi.')
    with results:
        if selection:
            metrics_path=selection['path'].parent/'evaluation_validation'/'metrics.json'
            if metrics_path.is_file():
                metrics=read_json(metrics_path)['metrics']
                st.subheader('Evaluasi validasi')
                st.dataframe([{'Metrik':'MAE','Nilai':round(metrics['mae'],3),'Satuan':'tahun'},
                              {'Metrik':'RMSE','Nilai':round(metrics['rmse'],3),'Satuan':'tahun'},
                              {'Metrik':'Median absolute error','Nilai':round(metrics['median_absolute_error'],3),'Satuan':'tahun'}],
                             hide_index=True,width='stretch')
                for filename,caption in [('prediction_vs_label.png','Prediksi dibandingkan label usia'),
                                          ('age_group_mae.png','MAE per kelompok usia')]:
                    path=metrics_path.parent/filename
                    if path.is_file(): st.image(str(path),caption=caption)
            else: st.write('Metrik validasi untuk model ini belum tersedia.')
        else: st.write('Hasil eksperimen akan tersedia setelah pelatihan dan evaluasi pada dataset dijalankan.')
    with about:
        st.write('Proyek ini mempelajari regresi usia dari foto wajah tunggal. MobileNetV3 Small mengekstrak fitur visual dan head regresi menghasilkan satu angka usia.')
        st.write('UTKFace memiliki label usia hasil estimasi algoritme dan pemeriksaan manusia. Galat terhadap label tidak sama dengan pengukuran umur kronologis yang terverifikasi.')
        st.markdown('[Sumber UTKFace](https://susanqq.github.io/UTKFace/) · [Paper MobileNetV3](https://openaccess.thecvf.com/content_ICCV_2019/html/Howard_Searching_for_MobileNetV3_ICCV_2019_paper.html)')
        st.caption('Foto unggahan diproses di memori sesi. Aplikasi tidak menyimpan foto tersebut ke disk.')


if __name__=='__main__': main()
