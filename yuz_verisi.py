import cv2

# Kamera cihazını başlat
vid_cam = cv2.VideoCapture(0)

# Yüz tespiti için önceden eğitilmiş Haarcascades sınıflandırıcısını yükle
yuz_dedektor = cv2.CascadeClassifier('D:/python repos/haarcascade_frontalface_default (1).xml')

# Kullanıcı ID'si ve kaydedilen görüntü sayısı
yuz_ismi = 2
sayi = 1

while True:
    # Kamera görüntüsünü al
    ret, resim_cerceve = vid_cam.read()

    if not ret:
        print("Kamera görüntüsü alınamadı.")
        break

    # Görüntüyü gri tonlamaya çevir
    gri = cv2.cvtColor(resim_cerceve, cv2.COLOR_BGR2GRAY)

    # Yüzleri tespit et
    yuzler = yuz_dedektor.detectMultiScale(gri, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in yuzler:
        # Yüzleri dikdörtgenle işaretle
        cv2.rectangle(resim_cerceve, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Yüz görüntüsünü kaydet
        cv2.imwrite("D:/python repos/veri/User" + str(yuz_ismi) + '.' + str(sayi) + ".jpg", gri[y:y+h, x:x+w])
        
        sayi += 1

    # Görüntüyü ekranda göster
    cv2.imshow('cerceve', resim_cerceve)

    # Kullanıcı 'q' tuşuna basarsa veya 50'den fazla görüntü kaydedildiyse döngüyü kır
    if cv2.waitKey(1) & 0xFF == ord('q') or sayi > 20:
        break

# Kaynakları serbest bırak
vid_cam.release()
cv2.destroyAllWindows()
