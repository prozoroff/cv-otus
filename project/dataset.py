import cv2
import torch
import kornia

from torch.utils.data import Dataset
from PIL import Image


from utils import match_histograms, extract_patches

IMG_PAIRS = [
    ('1_2014_7.jpg', '1_2021_7.jpg'),
    ('2_2014_7.jpg', '2_2021_7.jpg'),
    ('3_2014_7.jpg', '3_2021_7.jpg'),
    ('4_2014_7.jpg', '4_2021_7.jpg'),
    ('5_2014_7.jpg', '5_2021_7.jpg'),
    ('6_2014_7.jpg', '6_2021_7.jpg'),
    ('7_2014_7.jpg', '7_2021_7.jpg'),
    ('8_2014_7.jpg', '8_2021_7.jpg'),
    ('9_2014_7.jpg', '9_2021_7.jpg'),
    ('10_2014_7.jpg', '10_2021_7.jpg'),
    ('11_2014_7.jpg', '11_2021_7.jpg'),
    ('12_2014_7.jpg', '12_2021_7.jpg'),
    ('13_2014_7.jpg', '13_2021_7.jpg'),
    ('14_2014_7.jpg', '14_2021_7.jpg'),
    ('15_2014_7.jpg', '15_2021_7.jpg'),
    ('16_2014_7.jpg', '16_2021_7.jpg'),
    ('17_2014_7.jpg', '17_2021_7.jpg'),
    ('18_2014_7.jpg', '18_2021_7.jpg'),
    ('19_2014_7.jpg', '19_2021_7.jpg'),
    ('20_2014_7.jpg', '20_2021_7.jpg'),
]

CANNY_RANGE = [5, 30]
DIFF_THRESHOLD = 20
EQUALITY_THRESHOLD = 0.75
PATSH_SIZE = 128
CHANNELS = 3
CANNY_KERNEL_SIZE = (15, 15)
ROOT_DIR = '../images/'


def has_enough_details(img):
    factor = 1 / (PATSH_SIZE * PATSH_SIZE)
    canny_sum = kornia.filters.canny(img.unsqueeze(0).float(), kernel_size=CANNY_KERNEL_SIZE)[0].sum() * factor
    return canny_sum > CANNY_RANGE[0] and canny_sum < CANNY_RANGE[1]

def has_correct_diff(img_1, img_2):
    factor = 1 / (CHANNELS * PATSH_SIZE * PATSH_SIZE)
    diff_sum = (torch.abs(img_1 - img_2) > DIFF_THRESHOLD).sum().item() * factor
    return diff_sum < EQUALITY_THRESHOLD

class PatchesDataset(Dataset):
    def __init__(self, transform_low_res, transform_high_res, img_pairs_list=IMG_PAIRS):
        super(PatchesDataset, self).__init__()
        self.img_pairs_list = img_pairs_list
        self.transform_low_res = transform_low_res
        self.transform_high_res = transform_high_res
        
        self.images_1 = []
        self.images_2 = []

        for img_path_1, img_path_2 in self.img_pairs_list:
            image_1 = cv2.imread(ROOT_DIR + img_path_1)
            image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB)
            
            image_2 = cv2.imread(ROOT_DIR + img_path_2)
            image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2RGB)

            image_1, image_2 = match_histograms(image_1, image_2)

            patches_1 = extract_patches(Image.fromarray(image_1), PATSH_SIZE, PATSH_SIZE, PATSH_SIZE * 42)
            patches_2 = extract_patches(Image.fromarray(image_2), PATSH_SIZE, PATSH_SIZE, PATSH_SIZE * 42)
            
            for patch_1, patch_2 in zip(patches_1, patches_2):
                ok_diff = has_correct_diff(patch_1, patch_2)
                ok_details_1 = has_enough_details(patch_1.permute(2, 0, 1))
                ok_details_2 = has_enough_details(patch_2.permute(2, 0, 1))
                if ok_diff and ok_details_1 and ok_details_2:
                    self.images_1.append(patch_1)
                    self.images_2.append(patch_2)  

    def __len__(self):
        return len(self.images_1)

    def __getitem__(self, idx):
        img_1 = self.images_1[idx]
        img_2 = self.images_2[idx]
        return self.transform_low_res(img_1), self.transform_low_res(img_2), self.transform_high_res(img_1)
