import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def cargar_imagenes(ruta_carpeta):
    imagenes = []
    archivos = sorted([f for f in os.listdir(ruta_carpeta) if f.endswith('.jpg')])
    for archivo in archivos:
        img = cv2.imread(os.path.join(ruta_carpeta, archivo))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imagenes.append(img)
    return imagenes

def encontrar_correspondencias(img1, img2, metodo='sift'):
    if metodo == 'sift':
        descriptor = cv2.SIFT_create()
    elif metodo == 'surf':
        descriptor = cv2.xfeatures2d.SURF_create()
    elif metodo == 'orb':
        descriptor = cv2.ORB_create()

    kp1, des1 = descriptor.detectAndCompute(img1, None)
    kp2, des2 = descriptor.detectAndCompute(img2, None)
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(des1, des2, k=2)

    buenos_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            buenos_matches.append(m)

    return kp1, kp2, buenos_matches

def calcular_homografia(kp1, kp2, matches):
    puntos1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    puntos2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
    H, _ = cv2.findHomography(puntos2, puntos1, cv2.RANSAC)
    return H

def aplicar_warping(img, H, dimensiones):
    return cv2.warpPerspective(img, H, dimensiones)


# def calcular_alpha_dinamico(img1, img2):
#     diff = np.abs(img1 - img2)
#     alpha = np.mean(diff)
#     return alpha

def normalizar_colores(img):
    return cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

def ecualizar_histogramas(img1, img2):
    # Convertir imágenes a escala de grises para trabajar con intensidades
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    # Ecualizar los histogramas
    img1_eq = cv2.equalizeHist(img1_gray)
    img2_eq = cv2.equalizeHist(img2_gray)

    # Convertir de nuevo a color para mezclar
    img1_color = cv2.cvtColor(img1_eq, cv2.COLOR_GRAY2RGB)
    img2_color = cv2.cvtColor(img2_eq, cv2.COLOR_GRAY2RGB)
    
    return img1_color, img2_color


def adaptar_luminancia(img1, img2):
    # Convertir a flotante para evitar problemas de desbordamiento
    img1_f = img1.astype(np.float32)
    img2_f = img2.astype(np.float32)

    # Calcular la media de las imágenes
    mean1 = np.mean(img1_f)
    mean2 = np.mean(img2_f)

    # Calcular un factor para ajustar la media de img1 a img2
    scale_factor1 = mean2 / mean1
    scale_factor2 = mean1 / mean2

    # Aplicar el factor de escala
    adjusted_img1 = img1_f * scale_factor1
    adjusted_img2 = img2_f * scale_factor2

    # Reescalar a uint8
    adjusted_img1 = np.clip(adjusted_img1, 0, 255).astype(np.uint8)
    adjusted_img2 = np.clip(adjusted_img2, 0, 255).astype(np.uint8)

    return adjusted_img1, adjusted_img2


def blending_simple(img1, img_warped):
    # img1, img_warped = adaptar_luminancia(img1, img_warped)
    
    alpha = 0.3

    beta = 0.7
    gamma = 0
    # Aplicar la ecualización antes de realizar el blending
    img1, img_warped = ecualizar_histogramas(img1, img_warped)

    blended = cv2.addWeighted(img1, alpha, img_warped, beta, gamma)
    # blended = cv2.addWeighted(img1, alpha, img_warped, 1-alpha, 0)
    return blended


def mostrar_imagen(imagen, titulo='Imagen'):
    plt.figure(figsize=(10, 5))
    plt.imshow(imagen)
    plt.title(titulo)
    plt.axis('off')
    plt.show()


ruta_carpeta = './images'
imagenes = cargar_imagenes(ruta_carpeta)
kp1, kp2, matches = encontrar_correspondencias(imagenes[0], imagenes[1])
H = calcular_homografia(kp1, kp2, matches)
dimensiones = (imagenes[0].shape[1], imagenes[0].shape[0] )
# print(dimensiones.shape())
# print(imagenes[0])
img_warped = aplicar_warping(imagenes[1], H, dimensiones)
imagen_final = blending_simple(imagenes[0], img_warped)
mostrar_imagen(imagen_final, 'Imagen Panorámica Final')
