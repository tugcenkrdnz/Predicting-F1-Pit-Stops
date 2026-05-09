 # Kaggle Playground Series S6E5 -🏎️ F1 Pit Stop Prediction & Strategy Dashboard

Bu proje, Formula 1 yarış verilerini kullanarak bir pilotun bir sonraki turda pite girip girmeyeyeceğini tahmin eden yüksek başarımlı bir makine öğrenmesi modelidir. Model, Kaggle Playground Series (S6E5) kapsamında geliştirilmiş ve **0.9503 AUC** skoruna ulaşmıştır.

## 📊 Proje Özellikleri
- **Yüksek Başarım:** CatBoost algoritması ile 0.95+ AUC skoru.
- **Veri Temizliği:** Winsorization yöntemiyle aykırı değerlerin (outliers) yönetimi.
- **Özellik Mühendisliği:** Lastik ömrü, aşınma oranları ve zaman tutarlılığı gibi gelişmiş metrikler.
- **Canlı Arayüz:** Streamlit tabanlı interaktif tahmin paneli.

## 🚀 Kurulum ve Çalıştırma

### 1. Gereksinimler
Projeyi çalıştırmak için aşağıdaki kütüphanelerin yüklü olması gerekir:
```bash
pip install streamlit catboost pandas numpy matplotlib seaborn

```

### 2. Modeli Eğitme

Eğer elinizde hazır model (`.cbm`) yoksa, önce eğitim scriptini çalıştırarak modeli oluşturun:

```python
# train.py (veya jupyter notebook içinden)
model.fit(X, y)
model.save_model('catboost_pit_stop_model.cbm')

```

### 3. Streamlit Arayüzünü Başlatma

Terminal üzerinden uygulamayı ayağa kaldırın:

```bash
streamlit run app.py

```

## 🧠 Model Detayları

Model, CatBoostClassifier kullanılarak aşağıdaki hiper-parametrelerle optimize edilmiştir:

* **Derinlik (Depth):** 7
* **İterasyon:** 7000+
* **Öğrenme Oranı:** 0.015
* **Regülarizasyon (L2):** 5

### Kullanılan Önemli Özellikler (Features)

* `Tyre_Usage_Rate`: Lastik ömrünün tur sayısına oranı.
* `Time_Consistency`: Sürücünün tur zamanlarındaki kararlılığı.
* `Degradation_Per_Lap`: Tur başına düşen aşınma miktarı.
* `RaceProgress`: Yarışın tamamlanma yüzdesi.

## 📈 Görselleştirmeler

Proje kapsamında yapılan EDA (Keşifçi Veri Analizi) çalışmaları:

* **ROC Eğrisi:** Modelin doğru tespit ve yanlış alarm dengesinin analizi.
* **Box Plots:** Tur zamanlarındaki aykırı değerlerin tespiti.
* **Heatmap:** Değişkenler arası korelasyon analizi.
