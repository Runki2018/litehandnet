from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from collections import OrderedDict


def test_hand_ap(gt_json_file, pred_json_file):
    """测试手部边界框的AP。

        预测结果存储的的格式：
            [
                {   # 第一个预测结果
                    "image_id": int,
                    "category_id": int,
                    "bbox": list[lx, ly, w, h]
                    "score": float,
                },
                    ....
            ]
    Args:
        gt_json_file (str): 真值标注文件的路径
        pred_json_File (str): 预测结果的路径
    """

    coco = COCO(str(gt_json_file))
    coco_dt = coco.loadRes(str(pred_json_file))
    coco_eval = COCOeval(coco, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    metric_name = ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl',
                   'AR1', 'AR10', 'AR100', 'ARs', 'ARm', 'ARl']
    name_values=OrderedDict({k:v for k, v in zip(metric_name, coco_eval.stats)})
    return name_values
