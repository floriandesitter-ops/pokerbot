import cv2
import numpy as np
import os

TEMPLATE_VALUES_DIR = "templates/values"
TEMPLATE_SUITS_DIR = "templates/suits"
ZONE_VALEUR_GAUCHE = "valeur_gauche.png"
ZONE_SYMBOLE_GAUCHE = "symbole_gauche.png"
ZONE_VALEUR_DROITE = "valeur_droite.png"
ZONE_SYMBOLE_DROITE = "symbole_droite.png"

def preprocess(img, margin=4):
    """Noir/blanc avec marge autour pour absorber décalages."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    # Ajoute une marge blanche tout autour
    h, w = gray.shape
    padded = np.ones((h+2*margin, w+2*margin), dtype=np.uint8) * 255
    padded[margin:margin+h, margin:margin+w] = gray
    # Deux binarisations différentes
    _, otsu = cv2.threshold(padded, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, simple = cv2.threshold(padded, 127, 255, cv2.THRESH_BINARY)
    return [otsu, simple]

def load_templates(folder_path, target_shape):
    templates = {}
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".png"):
            name = filename[:-4]
            img = cv2.imread(os.path.join(folder_path, filename), cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            for margin in [4, 6]:
                # Double version des templates, marges différentes
                for proc in preprocess(img, margin):
                    resized = cv2.resize(proc, (target_shape[1]+margin*2, target_shape[0]+margin*2))
                    templates[f"{name}_m{margin}"] = resized
    return templates

def best_template_match(zone_img, templates):
    best_score, best_name = -1.0, None
    for zone_bin in preprocess(zone_img, margin=4):
        for name, tmpl in templates.items():
            if zone_bin.shape != tmpl.shape:
                tmpl = cv2.resize(tmpl, (zone_bin.shape[1], zone_bin.shape[0]))
            # Teste aussi un shift de ±2 pixels pour tolérer un léger décalage
            for dx in [-2, 0, 2]:
                for dy in [-2, 0, 2]:
                    # Découpe centrale avec décalage
                    x1 = max(0, dx)
                    y1 = max(0, dy)
                    x2 = min(zone_bin.shape[1], zone_bin.shape[1] + dx)
                    y2 = min(zone_bin.shape[0], zone_bin.shape[0] + dy)
                    crop = zone_bin[y1:y2, x1:x2]
                    tmpl_crop = tmpl[y1:y2, x1:x2]
                    if crop.shape != tmpl_crop.shape or crop.shape[0] == 0 or crop.shape[1] == 0:
                        continue
                    res = cv2.matchTemplate(crop, tmpl_crop, cv2.TM_CCOEFF_NORMED)
                    _, score, _, _ = cv2.minMaxLoc(res)
                    if score > best_score:
                        best_score = score
                        best_name = name
    return best_name.split("_m")[0], best_score

# Charge les zones extraites
valeur_gauche_img  = cv2.imread(ZONE_VALEUR_GAUCHE, cv2.IMREAD_COLOR)
symbole_gauche_img = cv2.imread(ZONE_SYMBOLE_GAUCHE, cv2.IMREAD_COLOR)
valeur_droite_img  = cv2.imread(ZONE_VALEUR_DROITE, cv2.IMREAD_COLOR)
symbole_droite_img = cv2.imread(ZONE_SYMBOLE_DROITE, cv2.IMREAD_COLOR)

value_templates_left = load_templates(TEMPLATE_VALUES_DIR, valeur_gauche_img.shape)
value_templates_right = load_templates(TEMPLATE_VALUES_DIR, valeur_droite_img.shape)
suit_templates_left = load_templates(TEMPLATE_SUITS_DIR, symbole_gauche_img.shape)
suit_templates_right = load_templates(TEMPLATE_SUITS_DIR, symbole_droite_img.shape)

val_gauche, score_val_gauche = best_template_match(valeur_gauche_img, value_templates_left)
sym_gauche, score_sym_gauche = best_template_match(symbole_gauche_img, suit_templates_left)
val_droite, score_val_droite = best_template_match(valeur_droite_img, value_templates_right)
sym_droite, score_sym_droite = best_template_match(symbole_droite_img, suit_templates_right)

print(f"Carte gauche : {val_gauche} {sym_gauche} (score valeur : {score_val_gauche:.2f}, score symbole : {score_sym_gauche:.2f})")
print(f"Carte droite : {val_droite} {sym_droite} (score valeur : {score_val_droite:.2f}, score symbole : {score_sym_droite:.2f})")

# Debug visuel
cv2.imshow("valeur_gauche", preprocess(valeur_gauche_img, 4)[0])
cv2.imshow("symbole_gauche", preprocess(symbole_gauche_img, 4)[0])
cv2.imshow("valeur_droite", preprocess(valeur_droite_img, 4)[0])
cv2.imshow("symbole_droite", preprocess(symbole_droite_img, 4)[0])
cv2.waitKey(0)
cv2.destroyAllWindows()
