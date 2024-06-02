import cv2  # OpenCV kütüphanesi, görüntü işleme için kullanılır
import numpy as np  # NumPy, matris işlemleri için kullanılır
from keras.models import load_model  # Keras, derin öğrenme modeli yüklemek için kullanılır
from keras.preprocessing import image  # Keras, görüntüyü modele uygun formata dönüştürmek için kullanılır

# Modelin yolu
model_path = 'model_yuzde91.keras'

# Modeli yükle
model = load_model(model_path)

# Sınıf etiketleri
labels = {0:'KARTON',1:'CAM',2:'METAL',3:'KAGIT',4:'PLASTIK',5:'COP'}

# Kamera aç
cap = cv2.VideoCapture(0)  # 0, bilgisayarın birincil kamera aygıtını temsil eder

while True:
    # Kameradan bir kare al
    ret, frame = cap.read()

    # Kare alınamadıysa döngüyü sonlandır
    if not ret:
        break

    # Görüntüyü yeniden boyutlandır ve ön işleme yap
    img = cv2.resize(frame, (224, 224))  # Modelin giriş boyutuna uygun boyuta yeniden boyutlandırma
    img = image.img_to_array(img)  # Görüntüyü numpy dizisine dönüştürme
    img = np.expand_dims(img, axis=0)  # Boyut ekleyerek (1, 224, 224, 3) şekline getirme
    img = img / 255.0  # Normalizasyon (0-1 aralığına getirme)

    # Sınıflandırma
    prediction = model.predict(img)  # Modeli kullanarak tahmin yapma
    predicted_class = np.argmax(prediction)  # En yüksek olasılığa sahip sınıfı belirleme

    # Tahmin edilen sınıf etiketini ekrana yazdır
    cv2.putText(frame, "TAHMINE GORE BU BIR :"  + labels[predicted_class], (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 2)  # Tahmin edilen sınıfı ekrana yazdırma

    # Kameradan alınan görüntüyü göster
    cv2.imshow('ATIK SINIFLANDIRICI', frame)

    # Çıkış için 'q' tuşuna basıldığında döngüyü sonlandır
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kamerayı kapat
cap.release()

# Pencereyi kapat
cv2.destroyAllWindows()
