from collections import OrderedDict
from .coco_wholebody_evaluation.evaluation_wholebody import test_lefthand, test_righthand
from xtcocotools.coco import COCO


def test_hand_oks(gt_json_file, pred_json_file):
    """测试手部的OKS,使用COCO-WHOLEBODY提供的API,由于API是分别计算左右手OKS,没有只计算手OKS的API。
    所以这里简单的调用左右手OKS API,分别得到左手和右手的OKS,再取均值。

        预测结果存储的的格式：
            [
                {   # 第一个预测结果
                    "image_id": int,
                    "category_id": int,
                    "keypoints": list([x, y, v] * 17),
                    "lefthand_kpts": list([x, y, v] * 21),
                    "righthand_kpts": list([x, y, v] * 21),
                    "score": float,
                    "lefthand_score": float,
                    "righthand_score": float,
                },
                    ....
            ]
    Args:
        gt_json_file (str): 真值标注文件的路径
        pred_json_File (str): 预测结果的路径
    """

    coco = COCO(str(gt_json_file))
    coco_dt = coco.loadRes(str(pred_json_file))
    
    # we change the return value of the following function from 0 to coco_eval
    left_coco_eval = test_lefthand(coco, coco_dt)
    right_coco_eval = test_righthand(coco, coco_dt)

    left_oks = left_coco_eval.stats
    right_oks = right_coco_eval.stats

    hand_oks = [(l+r)/2 for l, r in zip(left_oks, right_oks)]
    metric_name = ['AP', 'AP50', 'AP75', 'APm', 'APl',
                'AR', 'AR50', 'AR75', 'ARm', 'ARl']
    
    name_values=OrderedDict({k:v for k, v in zip(metric_name, hand_oks)})
    return name_values
