import cv2
import matplotlib.pyplot as plt

img = cv2.imread("main_detectee.png")
if img is None:
    raise FileNotFoundError("Image introuvable.")

print("Image OK :", img.shape)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Vérification visuelle")
plt.axis('off')
plt.show()
