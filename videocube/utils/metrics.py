from __future__ import absolute_import, division

import numpy as np

def center_error(rects1, rects2):
    r"""Center error.
    OTB-2015

    Args:
        rects1 (numpy.ndarray): An N x 4 numpy array, each line represent a rectangle
            (left, top, width, height).
        rects2 (numpy.ndarray): An N x 4 numpy array, each line represent a rectangle
            (left, top, width, height).
    """
    centers1 = rects1[..., :2] + (rects1[..., 2:] - 1) / 2
    centers2 = rects2[..., :2] + (rects2[..., 2:] - 1) / 2
    errors = np.sqrt(np.sum(np.power(centers1 - centers2, 2), axis=-1))

    return errors


def normalized_center_error(rects1, rects2, bound):
    r"""Normalized center error.
    Novel metrics.

    Args:
        rects1 (numpy.ndarray): Prediction box. An N x 4 numpy array, each line represent a rectangle (left, top, width, height).
        rects2 (numpy.ndarray): Groudntruth box. An N x 4 numpy array, each line represent a rectangle (left, top, width, height).
        bound (numpy.ndarray): A 4 dimensional array, denotes the bound (min_left, min_top, max_width, max_height) for ``rects1`` and ``rects2``.
    """
    centers1 = rects1[..., :2] + (rects1[..., 2:] - 1) / 2 # prediction box
    centers2 = rects2[..., :2] + (rects2[..., 2:] - 1) / 2 # groundtruth box
    width, height = bound

    # Calculate the Euclidean distance of two center points
    dists = np.sqrt(np.sum(np.power(centers1 - centers2, 2), axis=-1))
    
    # Calculate the distance between the groundtruth center point and the vertexz of the image
    thr_ul = np.sqrt(np.power(centers2[..., 0], 2)+np.power(centers2[..., 1], 2)) # Upper left
    thr_ur = np.sqrt(np.power((width-centers2[..., 0]), 2)+np.power(centers2[..., 1], 2)) # Upper right
    thr_ll = np.sqrt(np.power(centers2[..., 0], 2)+np.power((height-centers2[..., 1]), 2)) # Lower left
    thr_lr = np.sqrt(np.power((width-centers2[..., 0]), 2)+np.power((height-centers2[..., 1]), 2)) # Lower right

    def calculate_dist(point1, point2):
        return np.sqrt(np.power(point1[0]-point2[0],2)+np.power(point1[1]-point2[1],2))

    def calculate_detla(point, gt):
        # judge the prediction box center point with groundtruth

        # the center point of prediction box
        box_cx = point[0]
        box_cy = point[1]
        # the groundtruth four points information
        gt_xmin = gt[0]
        gt_ymin = gt[1]
        gt_xmax = gt[2]+gt[0]
        gt_ymax = gt[3]+gt[1]

        # flag calculates the points in area 5 (groundtruth box)
        flag = False

        # delta represents the shortest distence for center point of prediction box with the groundtruth boundary

        # judge in area 1 (upper left area)
        # for area 1, delta represents the distence for center point of prediction box with the upper left vertex of groundtruth box
        if box_cx <= gt_xmin and box_cy <= gt_ymin:
            delta = calculate_dist(point, (gt_xmin, gt_ymin))
        # judge in area 2 (upper area)
        # for area 2, delta represents the distence for center point of prediction box with the upper edge of groundtruth box
        if (gt_xmin < box_cx and box_cx <= gt_xmax) and box_cy <= gt_ymin:
            delta = gt_ymin - box_cy
        # judge in area 3 (upper right area)
        # for area 3, delta represents the distence for center point of prediction box with the upper right vertex of groundtruth box
        if gt_xmax < box_cx  and box_cy <= gt_ymin:
            delta = calculate_dist(point, (gt_xmax, gt_ymin))
        # judge in area 4 (left area)
        # for area 4, delta represents the distence for center point of prediction box with the left edge of groundtruth box
        if box_cx <= gt_xmin and (gt_ymin < box_cy and box_cy <= gt_ymax):
            delta = gt_xmin - box_cx
        # judge in area 5 (groundtruth box)
        # for area 5, delta is 0 since the center point of prediction box locates in the groundtruth box
        if (gt_xmin < box_cx and box_cx <= gt_xmax) and (gt_ymin < box_cy and box_cy <= gt_ymax):
            delta = 0
            flag = True
        # judge in area 6 (right area)
        # for area 6, delta represents the distence for center point of prediction box with the right edge of groundtruth box
        if gt_xmax < box_cx and (gt_ymin < box_cy and box_cy <= gt_ymax):
            delta = box_cy - gt_ymax
        # judge in area 7 (lower left area)
        # for area 7, delta represents the distence for center point of prediction box with the lower left vertex of groundtruth box
        if box_cx <= gt_xmin and gt_ymax < box_cy:
            delta = calculate_dist(point, (gt_xmin, gt_ymax))
        # judge in area 8 (lower area)
        # for area 8, delta represents the distence for center point of prediction box with the lower edge of groundtruth box
        if (gt_xmin < box_cx and box_cx <= gt_xmax) and gt_ymax < box_cy:
            delta = box_cy - gt_ymax
        # judge in area 9 (lower right area)
        # for area 9, delta represents the distence for center point of prediction box with the lower right vertex of groundtruth box
        if gt_xmax < box_cx and gt_ymax < box_cy:
            delta = calculate_dist(point, (gt_xmax, gt_ymax))

        return delta, flag
    
    def normalization(min, max, num):
        return (num-min)/(max-min)
            
    errors = np.zeros(len(rects1))
    flags = np.zeros(len(rects1))

    for i in range(len(rects1)):
        delta, flag = calculate_detla(centers1[i], rects2[i])

        # sum the points in groundtruth area
        if flag == False:
            flags[i] = 0
        else:
            flags[i] = 1
        
        # add the delta value as penalty factor
        error = dists[i] + delta
        
        # the max error is the distence for center point of groundtrut box with one of the four vertex in existing frame 
        thr_max = max((thr_ul[i]+calculate_detla((0, 0), rects2[i])[0]), (thr_ur[i]+calculate_detla((width, 0), rects2[i])[0]), (thr_ll[i]+calculate_detla((0, height), rects2[i])[0]), (thr_lr[i]+calculate_detla((width, height), rects2[i])[0]))

        # use the max value as threshold and normalize the error value
        error = normalization(0, thr_max, error)
        errors[i] = error
    return errors, flags


def iou(rects1, rects2, bound=None):
    r"""Intersection over union.

    Args:
        rects1 (numpy.ndarray): An N x 4 numpy array, each line represent a rectangle
            (left, top, width, height).
        rects2 (numpy.ndarray): An N x 4 numpy array, each line represent a rectangle
            (left, top, width, height).
        bound (numpy.ndarray): A 4 dimensional array, denotes the bound
            (min_left, min_top, max_width, max_height) for ``rects1`` and ``rects2``.
    """
    assert rects1.shape == rects2.shape
    if bound is not None:
        # bounded rects1
        rects1[:, 0] = np.clip(rects1[:, 0], 0, bound[0])
        rects1[:, 1] = np.clip(rects1[:, 1], 0, bound[1])
        rects1[:, 2] = np.clip(rects1[:, 2], 0, bound[0] - rects1[:, 0])
        rects1[:, 3] = np.clip(rects1[:, 3], 0, bound[1] - rects1[:, 1])
        # bounded rects2
        rects2[:, 0] = np.clip(rects2[:, 0], 0, bound[0])
        rects2[:, 1] = np.clip(rects2[:, 1], 0, bound[1])
        rects2[:, 2] = np.clip(rects2[:, 2], 0, bound[0] - rects2[:, 0])
        rects2[:, 3] = np.clip(rects2[:, 3], 0, bound[1] - rects2[:, 1])

    rects_inter = _intersection(rects1, rects2)
    areas_inter = np.prod(rects_inter[..., 2:], axis=-1)

    areas1 = np.prod(rects1[..., 2:], axis=-1)
    areas2 = np.prod(rects2[..., 2:], axis=-1)
    areas_union = areas1 + areas2 - areas_inter

    eps = np.finfo(float).eps
    ious = areas_inter / (areas_union + eps)
    ious = np.clip(ious, 0.0, 1.0)

    return ious


def _intersection(rects1, rects2):
    r"""Rectangle intersection.

    Args:
        rects1 (numpy.ndarray): An N x 4 numpy array, each line represent a rectangle
            (left, top, width, height).
        rects2 (numpy.ndarray): An N x 4 numpy array, each line represent a rectangle
            (left, top, width, height).
    """
    assert rects1.shape == rects2.shape
    x1 = np.maximum(rects1[..., 0], rects2[..., 0])
    y1 = np.maximum(rects1[..., 1], rects2[..., 1])
    x2 = np.minimum(rects1[..., 0] + rects1[..., 2],
                    rects2[..., 0] + rects2[..., 2])
    y2 = np.minimum(rects1[..., 1] + rects1[..., 3],
                    rects2[..., 1] + rects2[..., 3])

    w = np.maximum(x2 - x1, 0)
    h = np.maximum(y2 - y1, 0)

    return np.stack([x1, y1, w, h]).T


def giou(box1, box2):
    r"""Generalized-IoU Loss.

    From: 
        Generalized Intersection over Union: a Metric and a Loss for Bounding Box Regression

    Args:
        rects1 (numpy.ndarray): An N x 4 numpy array, each line represent a rectangle
            (left, top, width, height).
        rects2 (numpy.ndarray): An N x 4 numpy array, each line represent a rectangle
            (left, top, width, height).
        bound (numpy.ndarray): A 4 dimensional array, denotes the bound
            (min_left, min_top, max_width, max_height) for ``rects1`` and ``rects2``.
    """
    assert box1.shape == box2.shape
    xmin_intersection = np.maximum(box1[..., 0], box2[..., 0])
    ymin_intersection = np.maximum(box1[..., 1], box2[..., 1])
    xmax_intersection = np.minimum(box1[..., 0] + box1[..., 2],
                    box2[..., 0] + box2[..., 2])
    ymax_intersection = np.minimum(box1[..., 1] + box1[..., 3],
                    box2[..., 1] + box2[..., 3])
    w_intersection = np.maximum(xmax_intersection - xmin_intersection, 0)
    h_intersection = np.maximum(ymax_intersection - ymin_intersection, 0)

    intersection = np.stack([xmin_intersection, ymin_intersection, w_intersection, h_intersection]).T
    area_intersection = np.prod(intersection[...,2:], axis=-1)

    area1 = np.prod(box1[...,2:], axis=-1)
    area2 = np.prod(box2[...,2:], axis=-1)
    area_union = area1 + area2 - area_intersection

    eps = np.finfo(float).eps
    iou = area_intersection / (area_union + eps)
    iou = np.clip(iou, 0.0, 1.0)

    xmin_enclose = np.minimum(box1[..., 0], box2[..., 0])
    ymin_enclose = np.minimum(box1[..., 1], box2[..., 1])
    xmax_enclose = np.maximum(box1[..., 0] + box1[..., 2],
                    box2[..., 0] + box2[..., 2])
    ymax_enclose = np.maximum(box1[..., 1] + box1[..., 3],
                    box2[..., 1] + box2[..., 3])
    w_enclose = np.maximum(xmax_enclose - xmin_enclose, 0)
    h_enclose = np.maximum(ymax_enclose - ymin_enclose, 0)

    enclose = np.stack([xmin_enclose, ymin_enclose, w_enclose, h_enclose]).T
    area_enclose = np.prod(enclose[...,2:], axis=-1)

    giou = iou - (area_enclose - area_union)/area_enclose
    return giou


def diou(box1, box2):
    r"""Distance-IoU Loss.

    From:
        Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression

    Args:
        rects1 (numpy.ndarray): An N x 4 numpy array, each line represent a rectangle
            (left, top, width, height).
        rects2 (numpy.ndarray): An N x 4 numpy array, each line represent a rectangle
            (left, top, width, height).
        bound (numpy.ndarray): A 4 dimensional array, denotes the bound
            (min_left, min_top, max_width, max_height) for ``rects1`` and ``rects2``.
    """
    assert box1.shape == box2.shape
    xmin_intersection = np.maximum(box1[..., 0], box2[..., 0])
    ymin_intersection = np.maximum(box1[..., 1], box2[..., 1])
    xmax_intersection = np.minimum(box1[..., 0] + box1[..., 2],
                    box2[..., 0] + box2[..., 2])
    ymax_intersection = np.minimum(box1[..., 1] + box1[..., 3],
                    box2[..., 1] + box2[..., 3])
    w_intersection = np.maximum(xmax_intersection - xmin_intersection, 0)
    h_intersection = np.maximum(ymax_intersection - ymin_intersection, 0)

    intersection = np.stack([xmin_intersection, ymin_intersection, w_intersection, h_intersection]).T
    area_intersection = np.prod(intersection[...,2:], axis=-1)

    area1 = np.prod(box1[...,2:], axis=-1)
    area2 = np.prod(box2[...,2:], axis=-1)
    area_union = area1 + area2 - area_intersection

    eps = np.finfo(float).eps
    iou = area_intersection / (area_union + eps)
    iou = np.clip(iou, 0.0, 1.0)

    xmin_enclose = np.minimum(box1[..., 0], box2[..., 0])
    ymin_enclose = np.minimum(box1[..., 1], box2[..., 1])
    xmax_enclose = np.maximum(box1[..., 0] + box1[..., 2],
                    box2[..., 0] + box2[..., 2])
    ymax_enclose = np.maximum(box1[..., 1] + box1[..., 3],
                    box2[..., 1] + box2[..., 3])
    w_enclose = np.maximum(xmax_enclose - xmin_enclose, 0)
    h_enclose = np.maximum(ymax_enclose - ymin_enclose, 0)

    enclose = np.stack([xmin_enclose, ymin_enclose, w_enclose, h_enclose]).T
    diag_enclose = np.square(w_enclose) + np.square(h_enclose) + 1e-6

    xcenter1 = box1[..., 0] + box1[..., 2]/2
    xcenter2 = box2[..., 0] + box2[..., 2]/2
    ycenter1 = box1[..., 1] + box1[..., 3]/2
    ycenter2 = box2[..., 1] + box2[..., 3]/2

    diag_center = np.square(xcenter2 - xcenter1) + np.square(ycenter2 - ycenter1)

    diou = iou - diag_center/diag_enclose

    return diou



# 为评测其他数据集而添加

def rect_iou(rects1, rects2, bound=None):
    r"""Intersection over union.

    Args:
        rects1 (numpy.ndarray): An N x 4 numpy array, each line represent a rectangle
            (left, top, width, height).
        rects2 (numpy.ndarray): An N x 4 numpy array, each line represent a rectangle
            (left, top, width, height).
        bound (numpy.ndarray): A 4 dimensional array, denotes the bound
            (min_left, min_top, max_width, max_height) for ``rects1`` and ``rects2``.
    """
    assert rects1.shape == rects2.shape
    if bound is not None:
        # bounded rects1
        rects1[:, 0] = np.clip(rects1[:, 0], 0, bound[0])
        rects1[:, 1] = np.clip(rects1[:, 1], 0, bound[1])
        rects1[:, 2] = np.clip(rects1[:, 2], 0, bound[0] - rects1[:, 0])
        rects1[:, 3] = np.clip(rects1[:, 3], 0, bound[1] - rects1[:, 1])
        # bounded rects2
        rects2[:, 0] = np.clip(rects2[:, 0], 0, bound[0])
        rects2[:, 1] = np.clip(rects2[:, 1], 0, bound[1])
        rects2[:, 2] = np.clip(rects2[:, 2], 0, bound[0] - rects2[:, 0])
        rects2[:, 3] = np.clip(rects2[:, 3], 0, bound[1] - rects2[:, 1])

    rects_inter = _intersection(rects1, rects2)
    areas_inter = np.prod(rects_inter[..., 2:], axis=-1)

    areas1 = np.prod(rects1[..., 2:], axis=-1)
    areas2 = np.prod(rects2[..., 2:], axis=-1)
    areas_union = areas1 + areas2 - areas_inter

    eps = np.finfo(float).eps
    ious = areas_inter / (areas_union + eps)
    ious = np.clip(ious, 0.0, 1.0)

    return ious


def poly_iou(polys1, polys2, bound=None):
    r"""Intersection over union of polygons.

    Args:
        polys1 (numpy.ndarray): An N x 4 numpy array, each line represent a rectangle
            (left, top, width, height); or an N x 8 numpy array, each line represent
            the coordinates (x1, y1, x2, y2, x3, y3, x4, y4) of 4 corners.
        polys2 (numpy.ndarray): An N x 4 numpy array, each line represent a rectangle
            (left, top, width, height); or an N x 8 numpy array, each line represent
            the coordinates (x1, y1, x2, y2, x3, y3, x4, y4) of 4 corners.
        bound (numpy.ndarray, optional): A 2 dimensional array, denotes the image bound
            (width, height) for ``rects1`` and ``rects2``.
    """
    assert polys1.ndim in [1, 2]
    if polys1.ndim == 1:
        polys1 = np.array([polys1])
        polys2 = np.array([polys2])
    assert len(polys1) == len(polys2)

    polys1 = _to_polygon(polys1)
    polys2 = _to_polygon(polys2)
    if bound is not None:
        bound = box(0, 0, bound[0], bound[1])
        polys1 = [p.intersection(bound) for p in polys1]
        polys2 = [p.intersection(bound) for p in polys2]
    
    eps = np.finfo(float).eps
    ious = []
    for poly1, poly2 in zip(polys1, polys2):
        area_inter = poly1.intersection(poly2).area
        area_union = poly1.union(poly2).area
        ious.append(area_inter / (area_union + eps))
    ious = np.clip(ious, 0.0, 1.0)

    return ious


def _to_polygon(polys):
    r"""Convert 4 or 8 dimensional array to Polygons

    Args:
        polys (numpy.ndarray): An N x 4 numpy array, each line represent a rectangle
            (left, top, width, height); or an N x 8 numpy array, each line represent
            the coordinates (x1, y1, x2, y2, x3, y3, x4, y4) of 4 corners.
    """
    def to_polygon(x):
        assert len(x) in [4, 8]
        if len(x) == 4:
            return box(x[0], x[1], x[0] + x[2], x[1] + x[3])
        elif len(x) == 8:
            return Polygon([(x[2 * i], x[2 * i + 1]) for i in range(4)])
    
    if polys.ndim == 1:
        return to_polygon(polys)
    else:
        return [to_polygon(t) for t in polys]
