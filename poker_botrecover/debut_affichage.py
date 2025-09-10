import cv2

img = cv2.imread("main_detectee.png")
h, w, _ = img.shape
card_w = w // 2

card_left = img[:, 0:int(w*0.56)] 
card_right = img[:, int(w*0.35):]  

# Carte gauche (à ajuster pour ton jeu)
gray_left = cv2.cvtColor(card_left, cv2.COLOR_BGR2GRAY)
cv2.imshow("Carte gauche complète", card_left)
# Teste plein de valeurs pour trouver celles qui encadrent parfaitement la valeur et le symbole
cv2.imshow("Valeur gauche", gray_left[5:36, 2:30])
cv2.imshow("Symbole gauche", gray_left[35:63, 5:40])

# Carte droite (à ajuster aussi)
gray_right = cv2.cvtColor(card_right, cv2.COLOR_BGR2GRAY)
cv2.imshow("Carte droite complète", card_right)
# Teste aussi d'autres valeurs, ex:
cv2.imshow("Valeur droite", gray_right[3:33, 15:42])  # <-- à ajuster
cv2.imshow("Symbole droite", gray_right[32:62, 15:45]) # <-- à ajuster

cv2.waitKey(0)
cv2.destroyAllWindows()
