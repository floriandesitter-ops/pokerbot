import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

IMAGE_PATH = "main_detectee.png"
NB_CARTES = 2

img = cv2.cvtColor(cv2.imread(IMAGE_PATH), cv2.COLOR_BGR2RGB)
if img is None:
    raise FileNotFoundError(f"{IMAGE_PATH} introuvable.")

fig, ax = plt.subplots()
ax.imshow(img)
ax.set_title("Cliquez coin haut-gauche PUIS bas-droit de chaque carte")
points = []

def onclick(event):
    if event.xdata is not None and event.ydata is not None:
        x, y = int(event.xdata), int(event.ydata)
        points.append((x, y))
        ax.plot(x, y, 'go')
        plt.draw()
        print(f"Point {len(points)} : {x},{y}")
        if len(points) == NB_CARTES * 2:
            plt.close(fig)

cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()

# Génération des crops
for i in range(NB_CARTES):
    (x1, y1) = points[2*i]
    (x2, y2) = points[2*i+1]
    x_min, x_max = min(x1, x2), max(x1, x2)
    y_min, y_max = min(y1, y2), max(y1, y2)
    crop = img[y_min:y_max, x_min:x_max]
    out_name = f"carte_{i+1}.png"
    cv2.imwrite(out_name, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
    print(f"Carte {i+1} sauvegardée dans {out_name} ({x_min},{y_min}) -> ({x_max},{y_max})")

print("Calibration terminée !")
