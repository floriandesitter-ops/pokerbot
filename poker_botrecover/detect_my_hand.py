import cv2
import numpy as np
import os

# === CONFIGURATION ===
TEMPLATE_PSEUDO_DIR = "templates/pseudo"
IMAGE_PATH = "screenshot_table.png"  # capture complète
DEBUG = True

# === CHARGEMENT DE L'IMAGE ===
if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(f"Image '{IMAGE_PATH}' non trouvée.")

image = cv2.imread(IMAGE_PATH)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# === CHARGEMENT DES PSEUDOS ===
template_paths = {
    "pseudo_gris": os.path.join(TEMPLATE_PSEUDO_DIR, "pseudo_gris.png"),
    "pseudo_violet": os.path.join(TEMPLATE_PSEUDO_DIR, "pseudo_violet.png"),
    "pseudo_suivre": os.path.join(TEMPLATE_PSEUDO_DIR, "pseudo_suivre.png"),
    
}

templates = {}
for name, path in template_paths.items():
    if not os.path.exists(path):
        raise FileNotFoundError(f"Template '{path}' non trouvé.")
    templates[name] = cv2.imread(path, 0)

# === MATCH TEMPLATE POUR TROUVER TA POSITION ===
detections = []
for label, tmpl in templates.items():
    result = cv2.matchTemplate(gray, tmpl, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    detections.append((label, max_val, max_loc, tmpl.shape[::-1]))  # (w, h)

# Garder la meilleure détection
best = max(detections, key=lambda x: x[1])
label, score, (x, y), (w, h) = best

if score < 0.6:
    print(f"❌ Ton pseudo n’a pas été détecté de manière fiable (score = {score:.2f})")
    exit()

print(f"✅ Ton pseudo '{label}' détecté avec un score de {score:.2f} en position ({x},{y})")

# === CALCULER LA ZONE DE LA MAIN ===
# On suppose que les cartes sont au-dessus du pseudo (à ajuster si besoin)
hand_x = x - 30      # Décalage vers la gauche
hand_y = y - 110     # Décalage plus haut
hand_w = 160         # Plus large (deux cartes)
hand_h = 100   
hand_crop = image[hand_y:hand_y+hand_h, hand_x:hand_x+hand_w]

# === AFFICHAGE POUR VÉRIFICATION ===
if DEBUG:
    img_copy = image.copy()
    cv2.rectangle(img_copy, (x, y), (x+w, y+h), (255, 0, 255), 2)  # Rectangle sur le pseudo
    cv2.rectangle(img_copy, (hand_x, hand_y), (hand_x+hand_w, hand_y+hand_h), (0, 255, 0), 2)  # Zone main
    cv2.imshow("Détection", img_copy)
    cv2.imshow("Main détectée", hand_crop)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# === SAUVEGARDE DU CROP POUR USAGE FUTUR ===
cv2.imwrite("main_detectee.png", hand_crop)
