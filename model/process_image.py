"""画像を加工するクラス"""
import math

import cv2
import numpy as np
from PIL import Image


pers_num_path = "../pers_num.npy"
pts = np.load(pers_num_path)[0]


class ProcessImage:
    def __init__(self):
        self.kernel = np.ones((3, 3), np.uint8)

    @staticmethod
    def degree(x1, y1, x2, y2):
        """
        長辺が水平になるように、回転角を決める
        """
        rad = math.atan2(y1 - y2, x1 - x2)
        deg = math.degrees(rad)
        if 90 < deg:
            deg -= 180
        if deg < -90:
            deg += 180

        if deg < -45:
            deg += 90
        if deg > 45:
            deg -= 90
        return deg

    @staticmethod
    def rotation(img, deg):
        """
        回転角に従って画像を回転する
        """
        img_pil = Image.fromarray(img)
        img_rot = img_pil.rotate(deg, resample=Image.BILINEAR, expand=True)
        img_rot = np.asarray(img_rot)
        return img_rot

    @staticmethod
    def gray_scale(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def canny_edge_detect(img):
        return cv2.Canny(img, 100, 200)

    def morphology_close(self, img_th):
        return cv2.morphologyEx(img_th, cv2.MORPH_CLOSE, self.kernel)

    @staticmethod
    def invert(img_th):
        """ネガポジ変換"""
        return cv2.bitwise_not(img_th)

    def erode(self, img_th):
        return cv2.erode(img_th, self.kernel, iterations=1)

    @staticmethod
    def draw_contours(img_th):
        img_th_copy = img_th.copy()
        contours, hierarchy = cv2.findContours(img_th_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 輪郭検出
        for cnt in contours:
            cv2.drawContours(img_th_copy, [cnt], -1, 255, -1)
        return img_th_copy

    def get_contours(self, img_th):
        """キャニー画像から輪郭を検出"""
        img_canny = self.morphology_close(img_th)
        # エッジの穴の部分を埋めるための輪郭検出
        contours, _ = cv2.findContours(img_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 輪郭検出
        return contours
