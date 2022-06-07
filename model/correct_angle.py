from decimal import Decimal, ROUND_DOWN, ROUND_UP
import math
import statistics

import cv2
import numpy as np
from PIL import Image


class CorrectAngle:
    def __init__(self):
        self.threshold = 500
        self.min_length = 500
        self.check_count = 0
        self.finish = False
        self.get_result_deg = True

    def func1(self, img_pers_th, max_gap=30):
        """射影変換済みのキャニー画像から直線検出し、その直線の角度をリスト化"""
        deg_list = []
        while self.threshold > 0:
            lines = self._hough_lines_720(img_pers_th, self.threshold, self.min_length, max_gap)  # 角度は0.5°ずつ検出
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    deg = self._degree(x1, y1, x2, y2)  # 角度を求める
                    deg_list.append(deg)
                self.finish = True
            else:
                self.threshold -= 50
                self.min_length -= 50
                self.check_count += 1
            if self.finish is True:
                return deg_list, lines  # 重複のあるリスト

    def func2(self, deg_list, lines):
        """角度のリストを基にループを掛けて、条件に当てはまる角度(result_deg)を求める"""
        result_deg = 0
        new_deg_list = []
        while self.threshold > 0:
            for deg in deg_list:
                # print(self.threshold, self.min_length)
                """画像に直線がないとここでlines=Noneになり、エラーが出る"""
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    new_x, new_y = self._rotation2((x1, y1), (x2, y2), 0 - deg)
                    new_deg_ = self._degree(new_x, new_y, x2, y2)
                    new_deg = self._rounding(new_deg_)
                    new_deg_list.append(new_deg)
                a = self.func3(new_deg_list)
                if a == 1:
                    result_deg = deg
                elif a == 0:
                    new_deg_list = []
                    continue
                break

            if result_deg is not 0:
                return result_deg
            elif result_deg is 0:
                self.threshold -= 50
                new_deg_list = []
            if self.check_count > 10:
                self.finish = True
                self.get_result_deg = False
                break
        return result_deg

    def func3(self, new_deg_list):
        """new_deg_listに対して判定を掛ける関数"""
        max_deg = max(new_deg_list)
        min_deg = min(new_deg_list)
        if max_deg <= 0.5 and min_deg >= -0.5:
            return 1
        else:
            self.check_count += 1
            return 0

    def func4(self, result_deg, deg_list):
        """self.get_result_deg=False(10回角度補正して上手くいかない)とき角度リストの中央値を取得"""
        if self.get_result_deg is True:
            return result_deg
        else:
            return statistics.median(deg_list)

    @staticmethod
    def _degree(x1, y1, x2, y2):
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
    def _rotation2(pt1, pt2, deg):
        """回転行列"""
        # 度数単位の角度をラジアンに変換
        t = np.deg2rad(deg)

        xy = np.array(pt1)
        r_axis = np.array(pt2)

        # 回転行列
        R = np.array([[np.cos(t), -np.sin(t)],
                      [np.sin(t), np.cos(t)]])

        a = np.dot(R, xy - r_axis) + r_axis
        x, y = a[0], a[1]
        return x, y

    @staticmethod
    def _hough_lines_720(img_canny_2, threshold, min_length, max_gap):
        """直線を検出する"""
        return cv2.HoughLinesP(img_canny_2, 1, np.pi / 720,  # 角度は0.5°ずつ検出
                               threshold=threshold, minLineLength=min_length, maxLineGap=max_gap)

    @staticmethod
    def _rounding(deg):
        """数値の切り上げ・切り捨て"""
        if deg < 0:
            deg_decimal = Decimal(str(deg)).quantize(Decimal("0.1"), rounding=ROUND_DOWN)
        else:
            deg_decimal = Decimal(str(deg)).quantize(Decimal("0.1"), rounding=ROUND_UP)
        return float(deg_decimal)

    @staticmethod
    def _median(deg_list):
        """リストの中央値取得"""
        return statistics.median(deg_list)


if __name__ == '__main__':
    ca = CorrectAngle()
    _img_pers = cv2.imread(r"")
    _img_canny_2 = cv2.imread(r"", 0)

    _deg_list, _lines = ca.func1(_img_canny_2)
    _result_deg = ca.func2(_deg_list, _lines)
    _result = ca.rotation(_img_pers, _result_deg)

    cv2.namedWindow("_result", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("_result", 1200, 900)
    cv2.imshow("_result", _result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
