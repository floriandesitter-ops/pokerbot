import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

TEMPLATE_VALUE_PATH = "templates/value"
TEMPLATE_SUITE_PATH = "templates/suite"

# Fonction pour trouver la meilleure correspondance parmi tous les templates
def match_template(zone, templates_folder):
    best_score = -1
    best_name = None
    for fname in os.listdir(templates_folder):
        fpath = os.path.join(templates_folder, fname)
        template = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        if template is None or zone.shape[0] < template.shape[0] or zone.shape[1] < template.shape[1]:
            continue
        res = cv2.matchTemplate(zone, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        if max_val > best_score:
            best_score = max_val
            best_name = os.path.splitext(fname)[0]
    return best_name, best_score


# Fonction principale de détection
def recognize_cards(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --------- ZONE A AJUSTER ---------
    # Crops à tester : AJUSTE ces valeurs selon ton image !
    value_left   = gray[42:59, 15:27]
    symbol_left  = gray[60:78, 15:34]
    value_right   = gray[38:58, 56:72]   # Chiffre/lettre (A)
    symbol_right  = gray[58:75, 54:70]
    # ----------------------------------

    # VISUALISATION des crops
    plt.figure(figsize=(8,2))
    plt.subplot(1,4,1); plt.imshow(value_left, cmap='gray'); plt.title("Value Left")
    plt.subplot(1,4,2); plt.imshow(symbol_left, cmap='gray'); plt.title("Symbol Left")
    plt.subplot(1,4,3); plt.imshow(value_right, cmap='gray'); plt.title("Value Right")
    plt.subplot(1,4,4); plt.imshow(symbol_right, cmap='gray'); plt.title("Symbol Right")
    plt.tight_layout()
    plt.show()

    # Si tu veux stopper ici pour l’instant, fais simplement :
    # return None

    # Sinon tu lances la détection avec tes templates
    v1, score_v1 = match_template(value_left, TEMPLATE_VALUE_PATH)
    s1, score_s1 = match_template(symbol_left, TEMPLATE_SUITE_PATH)
    v2, score_v2 = match_template(value_right, TEMPLATE_VALUE_PATH)
    s2, score_s2 = match_template(symbol_right, TEMPLATE_SUITE_PATH)

    return (v1, s1, score_v1, score_s1), (v2, s2, score_v2, score_s2)
# Interface graphique
class PokerAssistantApp:
    def __init__(self, master):
        self.master = master
        master.title("Poker Assistant - Détection automatique de main")
        self.img_label = tk.Label(master)
        self.img_label.pack()

        self.result_label = tk.Label(master, text="", font=("Helvetica", 16))
        self.result_label.pack(pady=10)

        btn_frame = tk.Frame(master)
        btn_frame.pack(pady=5)

        tk.Button(btn_frame, text="Charger une main", command=self.load_image).pack(side="left", padx=10)
        tk.Button(btn_frame, text="Détecter mes cartes", command=self.detect).pack(side="left", padx=10)

        self.main_img_path = None

    def load_image(self):
        fname = filedialog.askopenfilename(filetypes=[("PNG Files", "*.png"), ("JPEG Files", "*.jpg")])
        if fname:
            self.main_img_path = fname
            img = Image.open(fname)
            img = img.resize((300, 160))  # ajuste si besoin
            self.img = ImageTk.PhotoImage(img)
            self.img_label.config(image=self.img)
            self.result_label.config(text="")

    def detect(self):
        if not self.main_img_path:
            messagebox.showinfo("Erreur", "Merci de charger une image d'abord !")
            return
        res1, res2 = recognize_cards(self.main_img_path)
        card1 = f"{res1[0]} {res1[1]} (score: {res1[2]:.2f}/{res1[3]:.2f})"
        card2 = f"{res2[0]} {res2[1]} (score: {res2[2]:.2f}/{res2[3]:.2f})"
        self.result_label.config(
            text=f"Carte gauche : {card1}\nCarte droite : {card2}"
        )

if __name__ == "__main__":
    root = tk.Tk()
    app = PokerAssistantApp(root)
    root.mainloop()
