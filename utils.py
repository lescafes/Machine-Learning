import numpy as np
import torch
def compute_iou(bbox1, bbox2):

    # TODO Compute IoU of 2 bboxes.

    if isinstance(bbox1, torch.Tensor):
        bbox1 = bbox1.detach().numpy()         #张量numpy化
    if isinstance(bbox2, torch.Tensor):
        bbox2 = bbox2.detach().numpy()         #张量numpy化

    area1 = (bbox1[:, 2] - bbox1[:, 0]) * (bbox1[:, 3] - bbox1[:, 1])     #第一个bbox面积 = 长×宽
    area2 = (bbox2[:, 2] - bbox2[:, 0]) * (bbox2[:, 3] - bbox2[:, 1])     #第二个bbox面积 = 长×宽
    xa = np.maximum(bbox1[:, 0], bbox2[:, 0])                #并集左上角顶点横坐标
    xb = np.minimum(bbox1[:, 2], bbox2[:, 2])                #并集右下角顶点横坐标
    ya = np.maximum(bbox1[:, 1], bbox2[:, 1])                #并集左上角顶点纵坐标
    yb = np.minimum(bbox1[:, 3], bbox2[:, 3])                #并集右下角顶点纵坐标

    inter = np.maximum(xb - xa, 0) * np.maximum(yb - ya, 0)    #并集面积
    union = area1 + area2 - inter
    return inter / union
    ...

    # End of todo