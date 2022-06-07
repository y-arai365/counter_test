"""全体の統合"""
import cv2
import numpy as np

from correct_angle import CorrectAngle
from perspective_transform import PerspectiveTransformer
from process_image import ProcessImage

pers_num_path = "../pers_num.npy"
pts = np.load(pers_num_path)[0]


class Model:
    def __init__(self, w, h):
        """画像から矩形の座標一覧を取得"""
        self.cor = CorrectAngle()
        self.pers = PerspectiveTransformer(w, h, pts)
        self.pro_img = ProcessImage()

    def get_perspective_image(self, img):
        """カメラ画像から射影変換した画像の取得"""
        return self.pers.transform(img)  # 射影変換

    def get_perspective_and_canny_image(self, img):
        """カメラ画像から射影変換したキャニー画像の取得"""
        img_gray = self.pro_img.gray_scale(img)  # グレースケール
        img_canny = self.pro_img.canny_edge_detect(img_gray)  # エッジ検出
        return self.pers.transform(img_canny)  # 射影変換

    def get_margin_edges(self, img_th):
        """取得したキャニー画像の輪郭を間引き"""
        img_close = self.pro_img.morphology_close(img_th)
        img_close = self.pro_img.draw_contours(img_close)
        img_canny_inv = self.pro_img.invert(img_close)
        img_canny_inv = self.pro_img.erode(img_canny_inv)
        return self.pro_img.canny_edge_detect(img_canny_inv)


if __name__ == '__main__':
    # _image_path = r"\\192.168.11.6\develop-data\撮影データ\個数カウントv2.3.0\\499-98\中8\frame.jpg"
    _image_path = r"\\192.168.11.6\develop-data\撮影データ\個数カウントマルワ\バックライトでの撮影\sample2\上2_10\frame.jpg"
    print(_image_path)

    n = np.fromfile(_image_path, dtype=np.uint8)
    _img = cv2.imdecode(n, cv2.IMREAD_COLOR)
    _height, _width = _img.shape[:2]
    _model = Model(_width, _height)
    _ca = CorrectAngle()

    _img_pers = _model.get_perspective_image(_img)
    _img_pers_canny = _model.get_perspective_and_canny_image(_img)
    _img_pers_canny = _model.get_margin_edges(_img_pers_canny)

    _deg_list, _lines = _ca.func1(_img_pers_canny)
    _result_deg = _ca.func2(_deg_list, _lines)
    _result_deg = _ca.func4(_result_deg, _deg_list)
    _result = _ca.rotation(_img_pers, _result_deg)

    cv2.namedWindow("result", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("result", 1200, 900)
    cv2.imshow("result", _result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
