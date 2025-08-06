import cv2
import os
from datetime import datetime
import tkinter as tk
from tkinter import messagebox

if not os.path.exists("captures"):
    os.makedirs("captures")
# Liste des objets connus par le modèle
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]
net = cv2.dnn.readNetFromCaffe(
    "MobileNetSSD_deploy.prototxt",
    "MobileNetSSD_deploy.caffemodel"
)

# Fonction de détection (lancée après clic bouton)
def detecter_objet():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                     0.007843, (300, 300), 127.5)

        net.setInput(blob)
        detections = net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > 0.5:
                idx = int(detections[0, 0, i, 1])
                label = CLASSES[idx]
                box = detections[0, 0, i, 3:7] * [w, h, w, h]
                (startX, startY, endX, endY) = box.astype("int")

                # Dessiner l'objet détecté
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(frame, label, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # 📸 Sauvegarder l'image
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"captures/detected_{label}_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"[✅] Objet détecté : {label} — image enregistrée : {filename}")

                cap.release()
                cv2.destroyAllWindows()
                messagebox.showinfo("Détection terminée", f"{label} détecté. Image sauvegardée.")
                return

        cv2.imshow("Détection d'objets", frame)

        if cv2.waitKey(1) == 27:  # Touche ESC pour quitter
            break

    cap.release()
    cv2.destroyAllWindows()


# Interface graphique avec Tkinter
fenetre = tk.Tk()
fenetre.title("App IA — Détection d'objet")

# Taille et position
fenetre.geometry("300x150")
fenetre.resizable(False, False)

# Bouton de démarrage
btn = tk.Button(fenetre, text="Démarrer la détection", command=detecter_objet, font=("Arial", 12), bg="#4CAF50", fg="white")
btn.pack(pady=40)

# Boucle principale
fenetre.mainloop()
