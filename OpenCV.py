import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing import image

# Modeli yükle
model = load_model('model_yuzde91.keras')

# Kategoriler
labels = {0:'KARTON',1:'CAM',2:'METAL',3:'KAGIT',4:'PLASTIK',5:'COP'}

# Kamera aç
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Görüntüyü yeniden boyutlandır ve ön işleme yap
    img = cv2.resize(frame, (224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalizasyon

    # Sınıflandırma
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)

    # Tahmin edilen sınıf etiketini ekrana yazdır
    cv2.putText(frame, "TAHMINE GORE BU BIR : " + labels[predicted_class], (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Kameradan alınan görüntüyü göster
    cv2.imshow('ATIK SINIFLANDIRICI', frame)

    # Çıkış için 'q' tuşuna basıldığında döngüyü sonlandır
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kamerayı kapat
cap.release()
cv2.destroyAllWindows()
