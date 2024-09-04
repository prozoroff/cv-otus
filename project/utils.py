import cv2
import numpy as np
import matplotlib.pyplot as plt

import torchvision.transforms as transforms 

def match_histograms(image1, image2):
    """
    Нормализует гистограмму второго изображения относительно гистограммы первого

    :param image1: Тензор первого изображения
    :param image2: Тензор второго изображения
    :return: Исходное первое и нормализованное второе изображения
    """

    result_image = np.zeros_like(image2)

    for i in range(3):  # Проход по каждому каналу RGB
        hist1 = cv2.calcHist([image1], [i], None, [256], [0, 256])
        hist2 = cv2.calcHist([image2], [i], None, [256], [0, 256])

        cdf1 = hist1.cumsum()
        cdf1 = (cdf1 / cdf1[-1]) * 255

        cdf2 = hist2.cumsum()
        cdf2 = (cdf2 / cdf2[-1]) * 255

        lut = np.interp(cdf2, cdf1, range(256)).astype(np.uint8)
        result_image[:,:,i] = cv2.LUT(image2[:,:,i], lut)

    return image1, result_image

def extract_patches(image_tensor, patch_height, patch_width, crop_size):
    """
    Извлекает патчи из изображения.

    :param image_tensor: Тензор изображения размером (H, W, C), где H - высота, W - ширина, C - количество каналов.
    :param patch_height: Высота патча.
    :param patch_width: Ширина патча.
    :param crop_size величина для кропа
    :return: Тензор с патчами размером (num_patches, patch_height, patch_width, C).
    """
    
    crop = transforms.CenterCrop(crop_size) 
    pil_to_tensor = transforms.PILToTensor()

    image_crop = pil_to_tensor(crop(image_tensor)).permute(1, 2, 0)

    # Получаем размеры изображения
    H, W, C = image_crop.shape

    # Проверяем, что размеры изображения делятся на размеры патчей
    assert H % patch_height == 0 and W % patch_width == 0, "Размеры изображения должны делиться на размеры патчей"

    # Количество патчей
    num_patches_h = H // patch_height
    num_patches_w = W // patch_width

    # Извлекаем патчи
    patches = image_crop.unfold(0, patch_height, patch_height).unfold(1, patch_width, patch_width)

    # Переставляем размеры для получения тензора (num_patches, patch_height, patch_width, C)
    patches = patches.permute(0, 1, 3, 4, 2).contiguous().view(-1, patch_height, patch_width, C)

    return patches

def show_images(images, size=4, title=''):
    """
    Отображаем картинки в линию

    :param images: Тензоры изображений
    :param size: Размер для отображения в контексте matplotlib
    """
    fig, axs = plt.subplots(1, len(images), figsize=(len(images) * size, size))
    if title:
        fig.suptitle(title, fontsize=26)
    for ax, image in zip(axs, images):
        ax.imshow(image)
        ax.axis(False)
    plt.show()
    
def unfold_batch(images):
    """
    Уменьшает размер изображений в батче в два раза, разворачивая их в новые каналы

    :param images: Батч изображений
    :return: Тензор с патчами размером (b, c * 4, h / 2, w / 2).
    """
    b, c, h, w = images.shape
    patches = images.unfold(2, int(h / 2), int(w / 2)).unfold(3, int(h / 2), int(w / 2))
    patches = patches.contiguous().view(b, c, 2, 2, int(h / 2), int(w / 2))
    patches = patches.view(b, -1, int(h / 2), int(w / 2))
    return patches

def unfold_batch_n_times(images, n):
    """
    Уменьшает размер изображений в батче в два в степени n раз, разворачивая их в новые каналы

    :param images: Батч изображений
    :return: Тензор с патчами размером (b, c * 4 ^ n, h / (2 ^ n), w / (2 ^ n)).
    """

    result = images
    for _ in range(n):
        result = unfold_batch(result)
    return result
