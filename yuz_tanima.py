import cv2
import numpy as np
import datetime

# Yüz tanıma için LBPH sınıflandırıcısını oluştur
taniyici = cv2.face.LBPHFaceRecognizer_create()
taniyici.read('D:/python repos/deneme.yml')

# Yüz tespiti için Haarcascades sınıflandırıcısını yükle
yuzsiniflandirici = cv2.CascadeClassifier('D:/python repos/haarcascade_frontalface_default (1).xml')

font = cv2.FONT_HERSHEY_SIMPLEX

# VideoCapture ile kamerayı başlat
vide_cam = cv2.VideoCapture(0)

# Giriş ve çıkış saatlerini takip eden sözlük
giris_saatleri = {}
cikis_saatleri = {}
taninan_id = None

while True:
    # Kamera görüntüsünü al
    ret, kamera = vide_cam.read()
    
    if not ret:
        print("Kamera görüntüsü alınamadı.")
        break

    # Görüntüyü gri tonlamaya çevir
    gri = cv2.cvtColor(kamera, cv2.COLOR_BGR2GRAY)
    
    # Yüzleri tespit et
    yuzler = yuzsiniflandirici.detectMultiScale(gri, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in yuzler:
        # Yüz kısmını kes
        yuz_goruntu = gri[y:y + h, x:x + w]
        Id, conf = taniyici.predict(yuz_goruntu)

        if Id in [1, 2, 3]:  # Tanımlı ID'ler
            isim = f"Person{Id}"
            if Id not in giris_saatleri:
                giris_saatleri[Id] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            cikis_saatleri[Id] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        else:
            isim = "Unknown"
        
        # Yüzlerin etrafına dikdörtgen çiz
        cv2.rectangle(kamera, (x - 20, y - 20), (x + w + 20, y + h + 20), (0, 255, 0), 2)
        cv2.rectangle(kamera, (x - 22, y - 90), (x + w + 22, y - 22), (0, 255, 0), -1)
        cv2.putText(kamera, str(isim), (x, y - 40), font, 2, (255, 255, 255), 3)

    # Görüntüyü ekranda göster
    cv2.imshow('Kamera', kamera)
    
    # 'q' tuşuna basıldığında döngüyü kır
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kaynakları serbest bırak
vide_cam.release()
cv2.destroyAllWindows()

# Giriş ve çıkış saatlerini dosyaya kaydet
with open('giris_cikis_saatleri.txt', 'a') as f:
    for Id in giris_saatleri:
        f.write(f"ID: {Id}\n")
        f.write(f"Giriş Saati: {giris_saatleri[Id]}\n")
        f.write(f"Cikis Saati: {cikis_saatleri[Id]}\n")
        f.write('------------------------\n')



"""import cv2
import numpy as np
import datetime

# Yüz tanıma için LBPH sınıflandırıcısını oluştur
taniyici = cv2.face.LBPHFaceRecognizer_create()
taniyici.read('D:/python repos/deneme.yml')

# Yüz tespiti için Haarcascades sınıflandırıcısını yükle
yuzsiniflandirici = cv2.CascadeClassifier('D:/python repos/haarcascade_frontalface_default (1).xml')

font = cv2.FONT_HERSHEY_SIMPLEX

# VideoCapture ile kamerayı başlat
vide_cam = cv2.VideoCapture(0)

giris_saati = None
cikis_saati = None
taninan_id = None

while True:
    # Kamera görüntüsünü al
    ret, kamera = vide_cam.read()
    
    if not ret:
        print("Kamera görüntüsü alınamadı.")
        break

    # Görüntüyü gri tonlamaya çevir
    gri = cv2.cvtColor(kamera, cv2.COLOR_BGR2GRAY)
    
    # Yüzleri tespit et
    yuzler = yuzsiniflandirici.detectMultiScale(gri, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in yuzler:
        # Yüz kısmını kes
        yuz_goruntu = gri[y:y + h, x:x + w]
        Id, conf = taniyici.predict(yuz_goruntu)
        
        # Eğer ID tanımlanmışsa
        if Id == 1:
            isim = "Person1"
            if taninan_id is None:
                giris_saati = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                taninan_id = Id
            cikis_saati = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        else:
            isim = "Unknown"

        # Yüzlerin etrafına dikdörtgen çiz ve ismi yaz
        cv2.rectangle(kamera, (x - 20, y - 20), (x + w + 20, y + h + 20), (0, 255, 0), 2)
        cv2.rectangle(kamera, (x - 22, y - 90), (x + w + 22, y - 22), (0, 255, 0), -1)
        cv2.putText(kamera, str(isim), (x, y - 40), font, 2, (255, 255, 255), 3)

    # Görüntüyü ekranda göster
    cv2.imshow('Kamera', kamera)
    
    # 'q' tuşuna basıldığında döngüyü kır
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kaynakları serbest bırak
vide_cam.release()
cv2.destroyAllWindows()

# Giriş ve çıkış saatlerini kaydet
if taninan_id is not None:
    print(f"Giris Saati: {giris_saati}")
    print(f"Cikis Saati: {cikis_saati}")
    with open('giris_cikis_saatleri.txt', 'a') as f:
        f.write(f"ID: {taninan_id}, Giris Saati: {giris_saati}, Cikis Saati: {cikis_saati}\n")
"""