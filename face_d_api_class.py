import torch
from utils.ssd_model import SSD
from utils.ssd_predict_show import SSDPredictShow
import cv2

class AnimeFaceDetect:
    def __init__(self, weight_path='./weights/ssd_best8.pth'):
        # 初期設定: クラス分類とネットワーク設定
        self.voc_classes = ['girl', 'girl_low', 'man', 'man_low']
        self.ssd_cfg = {
            'num_classes': 5,  # 背景クラスを含めたクラス数
            'input_size': 300,  # 入力画像サイズ
            'bbox_aspect_num': [4, 6, 6, 6, 4, 4],  # DBoxのアスペクト比
            'feature_maps': [38, 19, 10, 5, 3, 1],  # 特徴マップのサイズ
            'steps': [8, 16, 32, 64, 100, 300],  # DBoxのサイズを決定
            'min_sizes': [21, 45, 99, 153, 207, 261],  # 最小サイズ
            'max_sizes': [45, 99, 153, 207, 261, 315],  # 最大サイズ
            'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
        }
        self.net = SSD(phase="inference", cfg=self.ssd_cfg)
        net_weights = torch.load(weight_path, map_location={'cuda:0': 'cpu'})
        self.net.load_state_dict(net_weights)
        print('ネットワーク設定完了：学習済みの重みをロードしました')

    def face_det(self, img_data, confidence_level):
        # 基本的な顔検出を行うメソッド
        ssd = SSDPredictShow(eval_categories=self.voc_classes, net=self.net)
        rgb_img, predict_bbox, pre_dict_label_index, scores = ssd.ssd_predict(img_data, confidence_level)
        dnum = len(pre_dict_label_index)
        return dnum, rgb_img, predict_bbox, pre_dict_label_index, scores

    def face_det_sq(self, img_data, confidence_level):
        # 正方形の顔検出を行うメソッド
        ssd = SSDPredictShow(eval_categories=self.voc_classes, net=self.net)
        print("検出開始")
        rgb_img, predict_bbox, pre_dict_label_index, scores = ssd.ssd_predict(img_data, confidence_level)
        dnum = len(pre_dict_label_index)
        for i in range(dnum):
            box = predict_bbox[i]
            if (box[2] - box[0]) > (box[3] - box[1]):
                box_hlf = (box[3] - box[1]) / 2
                horizontal_center = (box[2] + box[0]) / 2
                box[0] = horizontal_center - box_hlf
                box[2] = horizontal_center + box_hlf
            else:
                box_hlf = (box[2] - box[0]) / 2
                vertical_center = (box[3] + box[1]) / 2
                box[1] = vertical_center - box_hlf
                box[3] = vertical_center + box_hlf
        return dnum, rgb_img, predict_bbox, pre_dict_label_index, scores

    def face_det_head(self, img_data, ratio, shift, confidence_level):
        print("confidence_level=",confidence_level)
        voc_classes = ['girl', 'girl_low', 'man', 'man_low']
        try:
            ssd = SSDPredictShow(eval_categories=self.voc_classes, net=self.net)
            print("step1")
            print("step2")
            # ボックスなどを取得
            rgb_img, predict_bbox, pre_dict_label_index, scores = ssd.ssd_predict(img_data,confidence_level)
            dnum=len(pre_dict_label_index)
            for i in range(dnum):
                box=predict_bbox[i]
                print(box)
                if (box[2] - box[0]) > (box[3] - box[1]):
                    box_hlf=(box[3] - box[1])/2
                    horizontal_center = (box[2] + box[0])/2
                    print("horizontal_center=",horizontal_center, "box_hlf=",box_hlf)
                    box[0]=horizontal_center + box_hlf
                    box[2]=horizontal_center - box_hlf
                else:
                    box_hlf=(box[2] - box[0])/2
                    vurtical_center=(box[3] + box[1])/2
                    print("vurtical_center=",vurtical_center, "box_hlf=",box_hlf)
                    box[1]=vurtical_center - box_hlf
                    box[3]=vurtical_center + box_hlf
                #head expantion
                new_horizontal_center = (box[2] + box[0])/2
                new_vurtical_center   = (box[3] + box[1])/2
                new_box_hlf=(box[2] - box[0])/2
                new_box_half=new_box_hlf*ratio
                new_horizontal_center = (box[2] + box[0])/2
                new_vurtical_center=(box[3] + box[1])/2
                box[0]=new_horizontal_center - new_box_half
                box[2]=new_horizontal_center + new_box_half           
                box[1]=new_vurtical_center - new_box_half + new_box_half*shift
                box[3]=new_vurtical_center + new_box_half + new_box_half*shift
            return dnum,rgb_img, predict_bbox, pre_dict_label_index, scores
        except:
            print("SSD error")
            return False


# 使用例
# detector = AnimeFaceDetect()
# results = detector.face_det(img_data, 0.5)
