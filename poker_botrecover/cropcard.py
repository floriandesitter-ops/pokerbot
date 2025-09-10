import cv2
import numpy as np

def rotate_img(img, angle):
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    # On garde la taille originale (border blanc)
    return cv2.warpAffine(img, M, (w, h), borderValue=255)

img = cv2.imread("main_detectee.png")
h, w, _ = img.shape
card_left = img[:, 0:int(w*0.56)]
card_right = img[:, int(w*0.35):]

gray_left = cv2.cvtColor(card_left, cv2.COLOR_BGR2GRAY)
gray_right = cv2.cvtColor(card_right, cv2.COLOR_BGR2GRAY)

# --- Cartes gauche (légèrement inclinée vers la gauche, donc on pivote vers la droite) ---
val_gauche = gray_left[35:63, 7:35]
symb_gauche = gray_left[75:120, 35:70]
val_gauche_rot = rotate_img(val_gauche, -4.5)      # angle à tester, ex: +5°
symb_gauche_rot = rotate_img(symb_gauche, -4.5)

# --- Cartes droite (légèrement inclinée vers la droite, donc on pivote vers la gauche) ---
val_droite = gray_right[3:33, 15:42]
symb_droite = gray_right[32:57, 15:45]
val_droite_rot = rotate_img(val_droite, 6.5)      # angle à tester, ex: -5°
symb_droite_rot = rotate_img(symb_droite, 6.5)

# Sauvegarde les images droites
cv2.imwrite("valeur_gauche.png", val_gauche_rot)
cv2.imwrite("symbole_gauche.png", symb_gauche_rot)
cv2.imwrite("valeur_droite.png", val_droite_rot)
cv2.imwrite("symbole_droite.png", symb_droite_rot)
