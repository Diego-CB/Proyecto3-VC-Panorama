'''
Univesidad Del Valle de Guatemala
Visión por Computadora

Integrantes:
- Diego Córdova   20212
- Paola De León   20361
- Paola Contreras 20213
'''

import numpy as np
import cv2
import matplotlib.pyplot as plt

# Display the result
def mostrar_imagen(imagen, titulo='Imagen'):
    plt.figure(figsize=(10, 5))
    plt.imshow(imagen)
    plt.title(titulo)
    plt.axis('off')
    plt.show()

# Cargar imágenes
img1 = cv2.imread('./images/libr/1.jpg')
img2 = cv2.imread('./images/libr/2.jpg')
img3 = cv2.imread('./images/libr/3.jpg')

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)

def blend_images(img1, img2, mask):
    mask = np.where(img1 != 0, 1, 0).astype(np.float32)
    blended_img = img1 * mask + img2 * (1 - mask)
    return blended_img.astype(np.uint8)

def mix2Images(img1, img2):
    # Inicializar el detector de puntos clave SIFT
    sift = cv2.SIFT_create()

    # Detectar puntos clave y descriptores
    keypoints_1, descriptors_1 = sift.detectAndCompute(img1, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(img2, None)

    # Inicializar y calcular coincidencias usando FLANN
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors_1, descriptors_2, k=2)

    # Filtrar coincidencias usando la prueba de razón de Lowe
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # Extraer la localización de los buenos puntos coincidentes
    points1 = np.float32([keypoints_1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    points2 = np.float32([keypoints_2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Calcular la matriz de homografía
    H, mask = cv2.findHomography(points2, points1, cv2.RANSAC)

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    corners2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    warped_corners2 = cv2.perspectiveTransform(corners2, H)

    corners = np.concatenate((corners1, warped_corners2), axis=0)
    [xmin, ymin] = np.int32(corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(corners.max(axis=0).ravel() + 0.5)

    t = [-xmin, -ymin]
    Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])

    warped_img2 = cv2.warpPerspective(img2, Ht @ H, (xmax - xmin, ymax - ymin))
    
    img1_towarp = np.zeros((warped_img2.shape))
    img1_towarp[t[1]:h1 + t[1], t[0]:w1 + t[0]] = img1

    # Create a black canvas
    return blend_images(img1_towarp, warped_img2, mask)

def stitch_images(image_list):
    result = image_list[0]

    for i in range(1, len(image_list)):
        img_next = image_list[i]
        result = mix2Images(result, img_next)

    return result

result = stitch_images([img1, img2, img3])
mostrar_imagen(result, 'Imagen Panorámica Final')
