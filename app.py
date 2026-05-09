import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier

# Sayfa Ayarları
st.set_page_config(page_title="F1 Strateji Tahmin", page_icon="🏎️")

# Modeli Yükle
@st.cache_resource
def load_model():
    model = CatBoostClassifier()
    model.load_model('catboost_pit_stop_model.cbm')
    return model

model = load_model()

# EĞİTİMDEKİ TAM KOLON SIRASI (Burayı eğitim kodundaki sırayla eşle)
feature_order = [
    'Driver', 'Compound', 'Race', 'Year', 'PitStop', 'LapNumber', 
    'Stint', 'TyreLife', 'Position', 'LapTime (s)', 'LapTime_Delta', 
    'Cumulative_Degradation', 'RaceProgress', 'Position_Change',
    'Tyre_Usage_Rate', 'Degradation_Per_Lap', 'Position_Momentum',
    'Tyre_Life_Pct', 'Time_Consistency', 'Degradation_Rate'
]

st.title("🏎️ F1 Pit Stop Tahmin Sistemi")

col1, col2 = st.columns(2)

with col1:
    driver = st.selectbox("Sürücü", ["VER", "HAM", "LEC", "NOR", "ALO", "PER", "SAI", "RUS"])
    race = st.selectbox("Yarış", ["Monaco", "Silverstone", "Interlagos", "Suzuka", "Spa", "Monza"])
    compound = st.selectbox("Lastik", ["SOFT", "MEDIUM", "HARD"])
    year = st.number_input("Yıl", value=2024)
    pit_stop = st.number_input("Pit Sayısı", min_value=0, step=1)
    lap_number = st.number_input("Tur", min_value=1, step=1)
    stint = st.number_input("Stint", min_value=1, step=1)

with col2:
    tyre_life = st.number_input("Lastik Ömrü (Tur)", min_value=0.0)
    position = st.slider("Pozisyon", 1, 20, 10)
    lap_time = st.number_input("Son Tur (s)", value=90.0)
    lap_delta = st.number_input("Delta (s)", value=0.0)
    cum_deg = st.number_input("Aşınma", value=-20.0)
    race_progress = st.slider("İlerleme", 0.0, 1.0, 0.5)
    pos_change = st.number_input("Pozisyon Değişimi", value=0)

if st.button("Stratejiyi Hesapla"):
    # 1. Ham veriyi oluştur
    data_dict = {
        'Driver': [driver], 'Compound': [compound], 'Race': [race],
        'Year': [year], 'PitStop': [pit_stop], 'LapNumber': [lap_number],
        'Stint': [stint], 'TyreLife': [tyre_life], 'Position': [position],
        'LapTime (s)': [lap_time], 'LapTime_Delta': [lap_delta],
        'Cumulative_Degradation': [cum_deg], 'RaceProgress': [race_progress],
        'Position_Change': [pos_change]
    }
    
    input_df = pd.DataFrame(data_dict)

    # 2. FEATURE ENGINEERING (Tüm yeni kolonları buraya ekliyoruz)
    input_df['Tyre_Usage_Rate'] = input_df['TyreLife'] / (input_df['LapNumber'] + 1)
    input_df['Degradation_Per_Lap'] = input_df['Cumulative_Degradation'] / (input_df['LapNumber'] + 1)
    input_df['Position_Momentum'] = input_df['Position_Change'] * input_df['RaceProgress']
    input_df['Tyre_Life_Pct'] = input_df['TyreLife'] / (input_df['LapNumber'] + 1)
    input_df['Degradation_Rate'] = input_df['Cumulative_Degradation'] / (input_df['TyreLife'] + 1)
    
    # Time_Consistency için (Eğer eğitimde sürücü ortalamasını kullandıysan, 
    # Streamlit'te basitleştirmek için 0 veya fark değerini verebilirsin)
    input_df['Time_Consistency'] = 0.0 # Veya (lap_time - sürücü_ortalaması)

    # 3. Kolon Sırasını Düzenle
    # Eğer eğitimde kullandığın bir kolon burada eksik kalırsa tekrar hata verir.
    # Bu yüzden feature_order listesinin doğruluğundan emin ol.
    input_df = input_df[feature_order]
    
    # 4. Tahmin
    proba = model.predict_proba(input_df)[0][1]
    
    st.divider()
    if proba > 0.5:
        st.error(f"🚨 TAVSİYE: PİT STOP! (Olasılık: %{proba*100:.2f})")
    else:
        st.success(f"✅ TAVSİYE: PISTTE KAL (Olasılık: %{proba*100:.2f})")