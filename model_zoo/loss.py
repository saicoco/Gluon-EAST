# pylint: disable=arguments-differ
"""Custom losses.
Losses are subclasses of gluon.loss.Loss which is a HybridBlock actually.
"""
from __future__ import absolute_import
from mxnet import gluon

class EASTLoss(gluon.loss.Loss):

    def __init__(self, cls_weight=0.01, iou_weight=1.0, angle_weight=20, weight=None, batch_axis=0, **kwargs):
        super(EASTLoss, self).__init__(weight=weight, batch_axis=batch_axis, **kwargs)
        self.cls_weight = cls_weight
        self.iou_weight = iou_weight
        self.angle_weight = angle_weight

    def hybrid_forward(self, F, score_gt, score_pred, geo_gt, geo_pred, training_masks,  *args, **kwargs):

        # classification loss
        eps = 1e-5

        intersection = F.sum(score_gt * score_pred * training_masks)
        union = F.sum(training_masks * score_gt) + F.sum(training_masks * score_pred) + eps
        dice_loss = 1. - (2 * intersection / union)

        # AABB loss
        top_gt, right_gt, bottom_gt, left_gt, angle_gt = F.split(geo_gt, axis=1, num_outputs=5, squeeze_axis=1)
        top_pred, right_pred, bottom_pred, left_pred, angle_pred = F.split(geo_pred, axis=1, num_outputs=5, squeeze_axis=1)

        area_gt = (top_gt + bottom_gt) * (left_gt + right_gt)
        area_pred = (top_pred + bottom_pred) * (left_pred + right_pred)
        w_union = F.minimum(left_gt, left_pred) + F.minimum(right_gt, right_pred)
        h_union = F.minimum(top_gt, top_pred) + F.minimum(bottom_gt, bottom_pred)

        area_inte = w_union * h_union
        area_union = area_gt + area_pred - area_inte
        L_AABB = -1.0 * F.log((area_inte + 1.0) / (area_union + 1.0))
        L_theta = 1.0 - F.cos(angle_gt - angle_pred)
        L_g = L_AABB + 20. * L_theta

        return F.mean(L_g * score_gt * training_masks) + dice_loss