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
img1 = cv2.imread('./images/1.jpg')
img2 = cv2.imread('./images/2.jpg')
img3 = cv2.imread('./images/3.jpg')
img4 = cv2.imread('./images/4.jpg')

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
img4 = cv2.cvtColor(img4, cv2.COLOR_BGR2RGB)

# Resize images to half their original size
img1 = cv2.resize(img1, (0,0), fx=0.35, fy=0.35)
img2 = cv2.resize(img2, (0,0), fx=0.35, fy=0.35)
img3 = cv2.resize(img3, (0,0), fx=0.35, fy=0.35)
img4 = cv2.resize(img4, (0,0), fx=0.35, fy=0.35)

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
    H, _ = cv2.findHomography(points2, points1, cv2.RANSAC)

    # Calcular dimensiones de salida y desplazamiento
    h1, w1, d1 = img1.shape
    h2, w2, d2 = img2.shape
    # panorama_width = int((w1 + w2) * 1.4)
    # panorama_height = int(max((h1, h2)) * 1.4)
    
    panorama_width = w1 + w2
    panorama_height = max(h1, h2)

    # Warp the second image
    img2_transformed = cv2.warpPerspective(
        img2,
        H,
        (panorama_width, panorama_height),
        # borderMode=cv2.BORDER_TRANSPARENT,
        # flags=cv2.INTER_NEAREST,
    )

    # Create a black canvas
    result = np.zeros((panorama_height, panorama_width, 3), dtype=np.uint8)

    # Place the first image on the canvas
    result[0:h1, 0:w1] = img1

    # Blend the transformed second image
    alpha = 0.5  # Define the blending factor

    def feather_blending(img1, img2):
        rows, cols, ch = img1.shape
        transition_width = int(cols * 0.35)  # Ajustar el ancho de transición aquí si es necesario

        # Crear la máscara de transición
        mask = np.zeros((rows, cols), dtype=np.float32)
        # Asegurar que la máscara comienza y termina con los valores correctos
        start = cols // 2 - transition_width // 2
        end = cols // 2 + transition_width // 2
        mask[:, :start] = 1
        mask[:, start:end] = np.linspace(1, 0, end - start)
        mask[:, end:] = 0

        # Asegurarse de que la máscara tenga tres canales si las imágenes son a color
        if ch == 3:
            mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

        # Aplicar la máscara
        blended = img1 * mask + img2 * (1 - mask)
        return np.clip(blended, 0, 255).astype(np.uint8)

    # mostrar_imagen(img2_transformed)
    # mostrar_imagen(result)
    # result = feather_blending(result, img2_transformed)
    result = cv2.addWeighted(result, 0.5, img2_transformed, 0.8, 0)
    return result

result = mix2Images(img3, img4)
result = mix2Images(img2, result)
result = mix2Images(img1, result)



mostrar_imagen(result, 'Imagen Panorámica Final')
