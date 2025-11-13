# import copy
# import torch
# from torch import nn
# from torchvision_wj.models.batch_image import BatchImage
# from torchvision_wj.utils.losses import *
# from torch.jit.annotations import Tuple, List, Dict, Optional


# __all__ = ["UNetWithBox"]

import copy
import torch
from torch import nn
from torchvision_wj.models.batch_image import BatchImage
from torchvision_wj.utils.losses import *
from torch.jit.annotations import Tuple, List, Dict, Optional


__all__ = ["UNetWithBox"]


class UNetWithBox(nn.Module):
    """
    Implements UNetWithBox.

    The input to the model is expected to be a list of tensors, each of shape [C, H, W], one for each
    image, and should be in 0-1 range. Different images can have different sizes.

    The behavior of the model changes depending if it is in training or evaluation mode.
    """

    def __init__(
        self,
        model,
        losses,
        loss_weights,
        softmax,
        batch_image=BatchImage,
        size_divisible=32,
    ):

        super(UNetWithBox, self).__init__()
        self.model = model
        nb_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("##### trainable parameters in model: {}".format(nb_params))
        self.losses_func = losses
        self.loss_weights = loss_weights
        self.softmax = softmax
        self.batch_image = batch_image(size_divisible)

        # Pre-compute loss function parameter names for faster lookup
        self._loss_param_names = []
        for loss_func in self.losses_func:
            param_names = loss_func.__call__.__code__.co_varnames
            self._loss_param_names.append(set(param_names))

    def _compute_boxes_and_mask(self, preds, targets, stride, device):
        """Optimized box and mask computation using vectorized operations"""
        batch_size, num_classes, h, w = preds.shape

        # Pre-allocate mask
        mask = torch.zeros_like(preds, device=device)

        # Collect all boxes and labels in batch
        all_boxes = []
        all_labels = []
        all_indices = []

        for n_img, target in enumerate(targets):
            if len(target["boxes"]) == 0:
                continue

            boxes = torch.round(target["boxes"] / stride).type(torch.int32)
            labels = target["labels"]

            all_boxes.append(boxes)
            all_labels.append(labels)
            all_indices.extend([n_img] * len(boxes))

        if not all_boxes:
            return (
                mask,
                torch.empty((0, 5), device=device),
                torch.empty((0, 6), device=device),
            )

        # Concatenate all boxes and labels
        boxes_cat = torch.cat(all_boxes, dim=0)
        labels_cat = torch.cat(all_labels, dim=0)
        indices = torch.tensor(all_indices, device=device, dtype=torch.long)

        # Vectorized mask assignment
        for i in range(len(boxes_cat)):
            n_img = indices[i]
            c = labels_cat[i]
            box = boxes_cat[i]

            # Clamp coordinates to valid range
            x1 = torch.clamp(box[0], 0, w - 1)
            y1 = torch.clamp(box[1], 0, h - 1)
            x2 = torch.clamp(box[2], 0, w - 1)
            y2 = torch.clamp(box[3], 0, h - 1)

            if x2 > x1 and y2 > y1:
                mask[n_img, c, y1 : y2 + 1, x1 : x2 + 1] = 1

        # Compute crop_boxes and gt_boxes in vectorized manner
        valid_mask = (boxes_cat[:, 2] > boxes_cat[:, 0]) & (
            boxes_cat[:, 3] > boxes_cat[:, 1]
        )

        if not valid_mask.any():
            return (
                mask,
                torch.empty((0, 5), device=device),
                torch.empty((0, 6), device=device),
            )

        boxes_valid = boxes_cat[valid_mask]
        labels_valid = labels_cat[valid_mask]
        indices_valid = indices[valid_mask]

        # Vectorized computation of centers and radii
        heights = (boxes_valid[:, 3] - boxes_valid[:, 1] + 1) / 2.0
        widths = (boxes_valid[:, 2] - boxes_valid[:, 0] + 1) / 2.0
        radii = torch.sqrt(heights**2 + widths**2)
        centers_x = (boxes_valid[:, 2] + boxes_valid[:, 0] + 1) // 2
        centers_y = (boxes_valid[:, 3] + boxes_valid[:, 1] + 1) // 2

        # Create crop_boxes and gt_boxes tensors
        crop_boxes = torch.stack(
            [indices_valid, labels_valid, centers_x, centers_y, radii], dim=1
        )

        gt_boxes = torch.stack(
            [
                indices_valid,
                labels_valid,
                boxes_valid[:, 0],
                boxes_valid[:, 1],
                boxes_valid[:, 2],
                boxes_valid[:, 3],
            ],
            dim=1,
        ).to(torch.int32)

        return mask, crop_boxes, gt_boxes

    def _compute_losses(self, seg_preds, targets, image_shape, compute_func):
        """Optimized loss computation with reduced overhead"""
        device = seg_preds[0].device
        ytrue = torch.stack([t["masks"] for t in targets], dim=0).long()
        seg_losses = {}

        for nb_level, preds in enumerate(seg_preds):
            stride = image_shape[-1] / preds.shape[-1]

            # Use optimized box computation
            mask, crop_boxes, gt_boxes = self._compute_boxes_and_mask(
                preds, targets, stride, device
            )

            # Prepare kwargs for loss functions
            kwargs_opt = {
                "ypred": preds,
                "ytrue": ytrue,
                "mask": mask,
                "gt_boxes": gt_boxes,
                "crop_boxes": crop_boxes,
            }

            # Compute losses with pre-computed parameter names
            for i, (loss_func, loss_w, param_names) in enumerate(
                zip(self.losses_func, self.loss_weights, self._loss_param_names)
            ):
                # Filter parameters efficiently
                loss_params = {
                    key: kwargs_opt[key] for key in kwargs_opt if key in param_names
                }
                loss_v = loss_func(**loss_params) * loss_w

                # Use f-string for faster string formatting
                key_prefix = f"{type(loss_func).__name__}/{nb_level}/"
                seg_losses.update(
                    {f"{key_prefix}{n}": loss_v[n] for n in range(len(loss_v))}
                )

        return seg_losses

    def sigmoid_compute_seg_loss(self, seg_preds, targets, image_shape):
        """Optimized sigmoid loss computation"""
        return self._compute_losses(
            seg_preds, targets, image_shape, self._compute_boxes_and_mask
        )

    def softmax_compute_seg_loss(self, seg_preds, targets, image_shape, eps=1e-6):
        """Optimized softmax loss computation"""
        return self._compute_losses(
            seg_preds, targets, image_shape, self._compute_boxes_and_mask
        )

    def _filter_invalid_boxes(self, targets):
        """Vectorized box filtering"""
        filtered_targets = []

        for target in targets:
            boxes = target["boxes"]

            if len(boxes) == 0:
                filtered_targets.append(target)
                continue

            # Vectorized validation check
            widths = boxes[:, 2] - boxes[:, 0]
            heights = boxes[:, 3] - boxes[:, 1]
            valid_mask = (widths > 0) & (heights > 0)

            if valid_mask.all():
                filtered_targets.append(target)
            elif valid_mask.any():
                # Create filtered target
                filtered_target = target.copy()
                filtered_target["boxes"] = boxes[valid_mask]
                if "labels" in target and len(target["labels"]) == len(boxes):
                    filtered_target["labels"] = target["labels"][valid_mask]
                filtered_targets.append(filtered_target)
            else:
                # Create empty target
                filtered_target = target.copy()
                filtered_target["boxes"] = torch.empty(
                    (0, 4), dtype=boxes.dtype, device=boxes.device
                )
                if "labels" in target and len(target["labels"]) == len(boxes):
                    filtered_target["labels"] = torch.empty(
                        0, dtype=target["labels"].dtype, device=boxes.device
                    )
                filtered_targets.append(filtered_target)

        return filtered_targets

    def _validate_boxes(self, targets):
        """Fast box validation with early termination"""
        for target_idx, target in enumerate(targets):
            boxes = target["boxes"]
            if len(boxes) > 0:
                # Fast degenerate box check
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    bb_idx = degenerate_boxes.any(dim=1).nonzero()[0, 0]
                    degen_bb = boxes[bb_idx].tolist()
                    raise ValueError(
                        f"All bounding boxes should have positive height and width. "
                        f"Found invalid box {degen_bb} for target at index {target_idx}."
                    )

    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).
        """

        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        if self.training:
            assert targets is not None
            # Fast box structure validation
            for target in targets:
                boxes = target["boxes"]
                if not (
                    isinstance(boxes, torch.Tensor)
                    and len(boxes.shape) == 2
                    and boxes.shape[-1] == 4
                ):
                    raise ValueError(
                        f"Expected target boxes to be a tensor of shape [N, 4], got {boxes.shape}."
                    )

        # Get original image sizes (optimized)
        original_image_sizes = [(img.shape[-2], img.shape[-1]) for img in images]

        images, targets = self.batch_image(images, targets)

        # Fast NaN/Inf check
        if torch.isnan(images.tensors).any():
            print("image is nan ..............")
        if torch.isinf(images.tensors).any():
            print("image is inf ..............")

        # Optimized box processing
        if targets is not None:
            # Filter invalid boxes first
            targets = self._filter_invalid_boxes(targets)
            # Then validate remaining boxes
            self._validate_boxes(targets)

        # Model forward pass
        seg_preds = self.model(images.tensors)

        # Calculate losses
        assert targets is not None
        if self.softmax:
            losses = self.softmax_compute_seg_loss(
                seg_preds, targets, images.tensors.shape
            )
        else:
            losses = self.sigmoid_compute_seg_loss(
                seg_preds, targets, images.tensors.shape
            )

        return losses, seg_preds


class UNetWithBox2(nn.Module):
    """
    Implements UNetWithBox.

    The input to the model is expected to be a list of tensors, each of shape [C, H, W], one for each
    image, and should be in 0-1 range. Different images can have different sizes.

    The behavior of the model changes depending if it is in training or evaluation mode.
    """

    def __init__(
        self,
        model,
        losses,
        loss_weights,
        softmax,
        batch_image=BatchImage,
        size_divisible=32,
    ):

        super(UNetWithBox, self).__init__()
        self.model = model
        nb_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("##### trainable parameters in model: {}".format(nb_params))
        self.losses_func = losses
        self.loss_weights = loss_weights
        self.softmax = softmax
        self.batch_image = batch_image(size_divisible)

    def sigmoid_compute_seg_loss(self, seg_preds, targets, image_shape):
        device = seg_preds[0].device
        dtype = seg_preds[0].dtype
        all_labels = [t["labels"] for t in targets]
        ytrue = torch.stack([t["masks"] for t in targets], dim=0).long()
        label_unique = torch.unique(torch.cat(all_labels, dim=0))
        seg_losses = {}
        for nb_level in range(len(seg_preds)):
            preds = seg_preds[nb_level]
            stride = image_shape[-1] / preds.shape[-1]

            mask = preds.new_full(preds.shape, 0, device=preds.device)
            crop_boxes = []
            gt_boxes = []
            for n_img, target in enumerate(targets):
                boxes = torch.round(target["boxes"] / stride).type(torch.int32)
                labels = target["labels"]
                for n in range(len(labels)):
                    box = boxes[n, :]
                    c = labels[n]  # .item()
                    mask[n_img, c, box[1] : box[3] + 1, box[0] : box[2] + 1] = 1

                    height, width = (box[2] - box[0] + 1) / 2.0, (
                        box[3] - box[1] + 1
                    ) / 2.0
                    r = torch.sqrt(height**2 + width**2)
                    cx = (box[2] + box[0] + 1) // 2
                    cy = (box[3] + box[1] + 1) // 2
                    # print('//// box ////',box, cx, cy, r)
                    crop_boxes.append(torch.tensor([n_img, c, cx, cy, r]))
                    gt_boxes.append(
                        torch.tensor(
                            [n_img, c, box[0], box[1], box[2], box[3]],
                            dtype=torch.int32,
                            device=device,
                        )
                    )
            if len(crop_boxes) == 0:
                crop_boxes = torch.empty((0, 5), device=device)
            else:
                crop_boxes = torch.stack(crop_boxes, dim=0)
            if len(gt_boxes) == 0:
                gt_boxes = torch.empty((0, 6), device=device)
            else:
                gt_boxes = torch.stack(gt_boxes, dim=0)

            # print('#boxes',crop_boxes.shape[0],gt_boxes.shape[0])
            assert crop_boxes.shape[0] == gt_boxes.shape[0]

            kwargs_opt = {
                "ypred": preds,
                "ytrue": ytrue,
                "mask": mask,
                "gt_boxes": gt_boxes,
                "crop_boxes": crop_boxes,
            }
            for loss_func, loss_w in zip(self.losses_func, self.loss_weights):
                loss_keys = loss_func.__call__.__code__.co_varnames
                loss_params = {
                    key: kwargs_opt[key]
                    for key in kwargs_opt.keys()
                    if key in loss_keys
                }
                loss_v = loss_func(**loss_params) * loss_w

                key_prefix = type(loss_func).__name__ + "/" + str(nb_level) + "/"
                loss_v = {key_prefix + str(n): loss_v[n] for n in range(len(loss_v))}
                seg_losses.update(loss_v)

        return seg_losses

    def softmax_compute_seg_loss(self, seg_preds, targets, image_shape, eps=1e-6):
        device = seg_preds[0].device
        dtype = seg_preds[0].dtype
        all_labels = [t["labels"] for t in targets]
        ytrue = torch.stack([t["masks"] for t in targets], dim=0).long()
        label_unique = torch.unique(torch.cat(all_labels, dim=0))
        seg_losses = {}
        for nb_level in range(len(seg_preds)):
            preds = seg_preds[nb_level]
            stride = image_shape[-1] / preds.shape[-1]

            mask = preds.new_full(preds.shape, 0, device=preds.device)
            crop_boxes = []
            gt_boxes = []
            for n_img, target in enumerate(targets):
                boxes = torch.round(target["boxes"] / stride).type(torch.int32)
                labels = target["labels"]
                for n in range(len(labels)):
                    box = boxes[n, :]
                    c = labels[n]  # .item()
                    mask[n_img, c, box[1] : box[3] + 1, box[0] : box[2] + 1] = 1

                    height, width = (box[2] - box[0] + 1) / 2.0, (
                        box[3] - box[1] + 1
                    ) / 2.0
                    r = torch.sqrt(height**2 + width**2)
                    cx = (box[2] + box[0] + 1) // 2
                    cy = (box[3] + box[1] + 1) // 2
                    # print('//// box ////',box, cx, cy, r)
                    crop_boxes.append(torch.tensor([n_img, c, cx, cy, r]))
                    gt_boxes.append(
                        torch.tensor(
                            [n_img, c, box[0], box[1], box[2], box[3]],
                            dtype=torch.int32,
                            device=device,
                        )
                    )
            if len(crop_boxes) == 0:
                crop_boxes = torch.empty((0, 5), device=device)
            else:
                crop_boxes = torch.stack(crop_boxes, dim=0)
            if len(gt_boxes) == 0:
                gt_boxes = torch.empty((0, 6), device=device)
            else:
                gt_boxes = torch.stack(gt_boxes, dim=0)

            # print('#boxes',crop_boxes.shape[0],gt_boxes.shape[0])
            assert crop_boxes.shape[0] == gt_boxes.shape[0]

            kwargs_opt = {
                "ypred": preds,
                "ytrue": ytrue,
                "mask": mask,
                "gt_boxes": gt_boxes,
                "crop_boxes": crop_boxes,
            }
            for loss_func, loss_w in zip(self.losses_func, self.loss_weights):
                loss_keys = loss_func.__call__.__code__.co_varnames
                loss_params = {
                    key: kwargs_opt[key]
                    for key in kwargs_opt.keys()
                    if key in loss_keys
                }
                loss_v = loss_func(**loss_params) * loss_w

                key_prefix = type(loss_func).__name__ + "/" + str(nb_level) + "/"
                loss_v = {key_prefix + str(n): loss_v[n] for n in range(len(loss_v))}
                seg_losses.update(loss_v)

        return seg_losses

    def forward(self, images, targets=None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        """
        Arguments:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """

        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        if self.training:
            assert targets is not None
            for target in targets:
                boxes = target["boxes"]
                if isinstance(boxes, torch.Tensor):
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                        raise ValueError(
                            "Expected target boxes to be a tensor"
                            "of shape [N, 4], got {:}.".format(boxes.shape)
                        )
                else:
                    raise ValueError(
                        "Expected target boxes to be of type "
                        "Tensor, got {:}.".format(type(boxes))
                    )

        original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], [])
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.batch_image(images, targets)
        if torch.isnan(images.tensors).sum() > 0:
            print("image is nan ..............")
        if torch.isinf(images.tensors).sum() > 0:
            print("image is inf ..............")

        # Check for degenerate boxes
        # TODO: Move this to a function
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]

                # Windows-specific fix: Filter out boxes with zero width/height before the check
                if len(boxes) > 0:
                    widths = boxes[:, 2] - boxes[:, 0]
                    heights = boxes[:, 3] - boxes[:, 1]
                    valid_mask = (widths > 0) & (heights > 0)

                    # Remove invalid boxes
                    if valid_mask.any():
                        target["boxes"] = boxes[valid_mask]
                        if "labels" in target and len(target["labels"]) == len(boxes):
                            target["labels"] = target["labels"][valid_mask]
                    else:
                        # If no valid boxes, create empty arrays
                        target["boxes"] = torch.empty(
                            (0, 4), dtype=boxes.dtype, device=boxes.device
                        )
                        if "labels" in target and len(target["labels"]) == len(boxes):
                            target["labels"] = torch.empty(
                                0, dtype=target["labels"].dtype, device=boxes.device
                            )

                # Now check for degenerate boxes (should be none after filtering)
                boxes = target["boxes"]
                if len(boxes) > 0:
                    degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                    if degenerate_boxes.any():
                        # print the first degenerate box
                        bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                        degen_bb: List[float] = boxes[bb_idx].tolist()
                        raise ValueError(
                            "All bounding boxes should have positive height and width."
                            " Found invalid box {} for target at index {}.".format(
                                degen_bb, target_idx
                            )
                        )

        seg_preds = self.model(images.tensors)

        ## calculate losses
        assert targets is not None
        if self.softmax:
            losses = self.softmax_compute_seg_loss(
                seg_preds, targets, images.tensors.shape
            )
        else:
            losses = self.sigmoid_compute_seg_loss(
                seg_preds, targets, images.tensors.shape
            )

        return losses, seg_preds


# import copy
# import torch
# from torch import nn
# from torchvision_wj.models.batch_image import BatchImage
# from torchvision_wj.utils.losses import *
# from torch.jit.annotations import Tuple, List, Dict, Optional


# __all__ = ["UNetWithBox"]


class UNetWithBoxold(nn.Module):
    """
    Implements UNetWithBox.

    The input to the model is expected to be a list of tensors, each of shape [C, H, W], one for each
    image, and should be in 0-1 range. Different images can have different sizes.

    The behavior of the model changes depending if it is in training or evaluation mode.
    """

    def __init__(
        self,
        model,
        losses,
        loss_weights,
        softmax,
        batch_image=BatchImage,
        size_divisible=32,
    ):

        super(UNetWithBox, self).__init__()
        self.model = model
        nb_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("##### trainable parameters in model: {}".format(nb_params))
        self.losses_func = losses
        self.loss_weights = loss_weights
        self.softmax = softmax
        self.batch_image = batch_image(size_divisible)

    def sigmoid_compute_seg_loss(self, seg_preds, targets, image_shape):
        device = seg_preds[0].device
        dtype = seg_preds[0].dtype
        all_labels = [t["labels"] for t in targets]
        ytrue = torch.stack([t["masks"] for t in targets], dim=0).long()
        label_unique = torch.unique(torch.cat(all_labels, dim=0))
        seg_losses = {}
        for nb_level in range(len(seg_preds)):
            preds = seg_preds[nb_level]
            stride = image_shape[-1] / preds.shape[-1]

            # masks = preds.new_full(preds.shape,1,device=device)
            # ## compute masks based on bounding box
            # gt_boxes_org = []
            # for n_img, target in enumerate(targets):
            #     if target['boxes_org'].shape[0]==0:
            #         continue
            #     bb = torch.round(target['boxes_org']/stride).type(torch.int32)
            #     ext = torch.tensor([[n_img,0]], device=device)
            #     gt_boxes_org.append(torch.cat([ext,bb], dim=1))
            # gt_boxes_org = torch.cat(gt_boxes_org, dim=0)
            # h, w = preds.shape[-2:]
            # bbox  = copy.deepcopy(gt_boxes_org[:,2:])
            # bbox[:,0::2] = torch.clamp(bbox[:,0::2], 0, w)
            # bbox[:,1::2] = torch.clamp(bbox[:,1::2], 0, h)
            # flag = (bbox[:,2]-bbox[:,0]>5)&(bbox[:,3]-bbox[:,1]>5)
            # gt_boxes_org = gt_boxes_org[flag,:]

            mask = preds.new_full(preds.shape, 0, device=preds.device)
            crop_boxes = []
            gt_boxes = []
            for n_img, target in enumerate(targets):
                boxes = torch.round(target["boxes"] / stride).type(torch.int32)
                labels = target["labels"]
                for n in range(len(labels)):
                    box = boxes[n, :]
                    c = labels[n]  # .item()
                    mask[n_img, c, box[1] : box[3] + 1, box[0] : box[2] + 1] = 1

                    height, width = (box[2] - box[0] + 1) / 2.0, (
                        box[3] - box[1] + 1
                    ) / 2.0
                    r = torch.sqrt(height**2 + width**2)
                    cx = (box[2] + box[0] + 1) // 2
                    cy = (box[3] + box[1] + 1) // 2
                    # print('//// box ////',box, cx, cy, r)
                    crop_boxes.append(torch.tensor([n_img, c, cx, cy, r]))
                    gt_boxes.append(
                        torch.tensor(
                            [n_img, c, box[0], box[1], box[2], box[3]],
                            dtype=torch.int32,
                            device=device,
                        )
                    )
            if len(crop_boxes) == 0:
                crop_boxes = torch.empty((0, 5), device=device)
            else:
                crop_boxes = torch.stack(crop_boxes, dim=0)
            if len(gt_boxes) == 0:
                gt_boxes = torch.empty((0, 6), device=device)
            else:
                gt_boxes = torch.stack(gt_boxes, dim=0)

            # print('#boxes',crop_boxes.shape[0],gt_boxes.shape[0])
            assert crop_boxes.shape[0] == gt_boxes.shape[0]

            kwargs_opt = {
                "ypred": preds,
                "ytrue": ytrue,
                "mask": mask,
                "gt_boxes": gt_boxes,
                "crop_boxes": crop_boxes,
            }
            for loss_func, loss_w in zip(self.losses_func, self.loss_weights):
                loss_keys = loss_func.__call__.__code__.co_varnames
                loss_params = {
                    key: kwargs_opt[key]
                    for key in kwargs_opt.keys()
                    if key in loss_keys
                }
                loss_v = loss_func(**loss_params) * loss_w

                key_prefix = type(loss_func).__name__ + "/" + str(nb_level) + "/"
                loss_v = {key_prefix + str(n): loss_v[n] for n in range(len(loss_v))}
                seg_losses.update(loss_v)

            # from imageio import imwrite
            # import numpy as np
            # for k in range(ytrue.shape[0]):
            #     true = ytrue[k,0].cpu().numpy().astype(float)
            #     msk  = mask[k,0].cpu().numpy()
            #     print(np.unique(true),np.unique(msk))
            #     concat = true+msk
            #     concat = 255*(concat-concat.min())/(concat.max()-concat.min()+1e-6)
            #     concat = concat.astype(np.uint8)
            #     # concat = (np.hstack([true,msk])*255).astype(np.uint8)
            #     imwrite(str(k)+'.png', concat)
            # sys.exit()
        return seg_losses

    def softmax_compute_seg_loss(self, seg_preds, targets, image_shape, eps=1e-6):
        device = seg_preds[0].device
        dtype = seg_preds[0].dtype
        all_labels = [t["labels"] for t in targets]
        ytrue = torch.stack([t["masks"] for t in targets], dim=0).long()
        label_unique = torch.unique(torch.cat(all_labels, dim=0))
        seg_losses = {}
        for nb_level in range(len(seg_preds)):
            preds = seg_preds[nb_level]
            stride = image_shape[-1] / preds.shape[-1]

            mask = preds.new_full(preds.shape, 0, device=preds.device)
            crop_boxes = []
            gt_boxes = []
            for n_img, target in enumerate(targets):
                boxes = torch.round(target["boxes"] / stride).type(torch.int32)
                labels = target["labels"]
                for n in range(len(labels)):
                    box = boxes[n, :]
                    c = labels[n]  # .item()
                    mask[n_img, c, box[1] : box[3] + 1, box[0] : box[2] + 1] = 1

                    height, width = (box[2] - box[0] + 1) / 2.0, (
                        box[3] - box[1] + 1
                    ) / 2.0
                    r = torch.sqrt(height**2 + width**2)
                    cx = (box[2] + box[0] + 1) // 2
                    cy = (box[3] + box[1] + 1) // 2
                    # print('//// box ////',box, cx, cy, r)
                    crop_boxes.append(torch.tensor([n_img, c, cx, cy, r]))
                    gt_boxes.append(
                        torch.tensor(
                            [n_img, c, box[0], box[1], box[2], box[3]],
                            dtype=torch.int32,
                            device=device,
                        )
                    )
            if len(crop_boxes) == 0:
                crop_boxes = torch.empty((0, 5), device=device)
            else:
                crop_boxes = torch.stack(crop_boxes, dim=0)
            if len(gt_boxes) == 0:
                gt_boxes = torch.empty((0, 6), device=device)
            else:
                gt_boxes = torch.stack(gt_boxes, dim=0)

            # print('#boxes',crop_boxes.shape[0],gt_boxes.shape[0])
            assert crop_boxes.shape[0] == gt_boxes.shape[0]

            kwargs_opt = {
                "ypred": preds,
                "ytrue": ytrue,
                "mask": mask,
                "gt_boxes": gt_boxes,
                "crop_boxes": crop_boxes,
            }
            for loss_func, loss_w in zip(self.losses_func, self.loss_weights):
                loss_keys = loss_func.__call__.__code__.co_varnames
                loss_params = {
                    key: kwargs_opt[key]
                    for key in kwargs_opt.keys()
                    if key in loss_keys
                }
                loss_v = loss_func(**loss_params) * loss_w

                key_prefix = type(loss_func).__name__ + "/" + str(nb_level) + "/"
                loss_v = {key_prefix + str(n): loss_v[n] for n in range(len(loss_v))}
                seg_losses.update(loss_v)

        return seg_losses

    def forward(self, images, targets=None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        """
        Arguments:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """

        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        if self.training:
            assert targets is not None
            for target in targets:
                boxes = target["boxes"]
                if isinstance(boxes, torch.Tensor):
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                        raise ValueError(
                            "Expected target boxes to be a tensor"
                            "of shape [N, 4], got {:}.".format(boxes.shape)
                        )
                else:
                    raise ValueError(
                        "Expected target boxes to be of type "
                        "Tensor, got {:}.".format(type(boxes))
                    )

        original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], [])
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.batch_image(images, targets)
        if torch.isnan(images.tensors).sum() > 0:
            print("image is nan ..............")
        if torch.isinf(images.tensors).sum() > 0:
            print("image is inf ..............")

        # Check for degenerate boxes
        # TODO: Move this to a function
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    raise ValueError(
                        "All bounding boxes should have positive height and width."
                        " Found invalid box {} for target at index {}.".format(
                            degen_bb, target_idx
                        )
                    )

        seg_preds = self.model(images.tensors)
        # print('--------------image and seg outputs: ')
        # print(images.tensors.shape, seg_preds[0].shape)
        # for n_img, target in enumerate(targets):
        #     print(n_img, target['masks'].shape)

        ## calculate losses
        assert targets is not None
        if self.softmax:
            losses = self.softmax_compute_seg_loss(
                seg_preds, targets, images.tensors.shape
            )
        else:
            losses = self.sigmoid_compute_seg_loss(
                seg_preds, targets, images.tensors.shape
            )

        return losses, seg_preds


# import copy
# import torch
# from torch import nn
# from torchvision_wj.models.batch_image import BatchImage
# from torchvision_wj.utils.losses import *
# from torch.jit.annotations import Tuple, List, Dict, Optional


# __all__ = ["UNetWithBox"]


# class UNetWithBox(nn.Module):
#     """
#     Implements UNetWithBox.

#     The input to the model is expected to be a list of tensors, each of shape [C, H, W], one for each
#     image, and should be in 0-1 range. Different images can have different sizes.

#     The behavior of the model changes depending if it is in training or evaluation mode.
#     """

#     def __init__(
#         self,
#         model,
#         losses,
#         loss_weights,
#         softmax,
#         batch_image=BatchImage,
#         size_divisible=32,
#     ):

#         super(UNetWithBox, self).__init__()
#         self.model = model
#         nb_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
#         print("##### trainable parameters in model: {}".format(nb_params))
#         self.losses_func = losses
#         self.loss_weights = loss_weights
#         self.softmax = softmax
#         self.batch_image = batch_image(size_divisible)

#     def sigmoid_compute_seg_loss(self, seg_preds, targets, image_shape):
#         device = seg_preds[0].device
#         dtype = seg_preds[0].dtype
#         all_labels = [t["labels"] for t in targets]
#         ytrue = torch.stack([t["masks"] for t in targets], dim=0).long()
#         label_unique = torch.unique(torch.cat(all_labels, dim=0))
#         seg_losses = {}
#         for nb_level in range(len(seg_preds)):
#             preds = seg_preds[nb_level]
#             stride = image_shape[-1] / preds.shape[-1]

#             # masks = preds.new_full(preds.shape,1,device=device)
#             # ## compute masks based on bounding box
#             # gt_boxes_org = []
#             # for n_img, target in enumerate(targets):
#             #     if target['boxes_org'].shape[0]==0:
#             #         continue
#             #     bb = torch.round(target['boxes_org']/stride).type(torch.int32)
#             #     ext = torch.tensor([[n_img,0]], device=device)
#             #     gt_boxes_org.append(torch.cat([ext,bb], dim=1))
#             # gt_boxes_org = torch.cat(gt_boxes_org, dim=0)
#             # h, w = preds.shape[-2:]
#             # bbox  = copy.deepcopy(gt_boxes_org[:,2:])
#             # bbox[:,0::2] = torch.clamp(bbox[:,0::2], 0, w)
#             # bbox[:,1::2] = torch.clamp(bbox[:,1::2], 0, h)
#             # flag = (bbox[:,2]-bbox[:,0]>5)&(bbox[:,3]-bbox[:,1]>5)
#             # gt_boxes_org = gt_boxes_org[flag,:]

#             mask = preds.new_full(preds.shape, 0, device=preds.device)
#             crop_boxes = []
#             gt_boxes = []
#             for n_img, target in enumerate(targets):
#                 boxes = torch.round(target["boxes"] / stride).type(torch.int32)
#                 labels = target["labels"]
#                 for n in range(len(labels)):
#                     box = boxes[n, :]
#                     c = labels[n]  # .item()
#                     mask[n_img, c, box[1] : box[3] + 1, box[0] : box[2] + 1] = 1

#                     height, width = (box[2] - box[0] + 1) / 2.0, (
#                         box[3] - box[1] + 1
#                     ) / 2.0
#                     r = torch.sqrt(height**2 + width**2)
#                     cx = (box[2] + box[0] + 1) // 2
#                     cy = (box[3] + box[1] + 1) // 2
#                     # print('//// box ////',box, cx, cy, r)
#                     crop_boxes.append(torch.tensor([n_img, c, cx, cy, r]))
#                     gt_boxes.append(
#                         torch.tensor(
#                             [n_img, c, box[0], box[1], box[2], box[3]],
#                             dtype=torch.int32,
#                             device=device,
#                         )
#                     )
#             if len(crop_boxes) == 0:
#                 crop_boxes = torch.empty((0, 5), device=device)
#             else:
#                 crop_boxes = torch.stack(crop_boxes, dim=0)
#             if len(gt_boxes) == 0:
#                 gt_boxes = torch.empty((0, 6), device=device)
#             else:
#                 gt_boxes = torch.stack(gt_boxes, dim=0)

#             # print('#boxes',crop_boxes.shape[0],gt_boxes.shape[0])
#             assert crop_boxes.shape[0] == gt_boxes.shape[0]

#             kwargs_opt = {
#                 "ypred": preds,
#                 "ytrue": ytrue,
#                 "mask": mask,
#                 "gt_boxes": gt_boxes,
#                 "crop_boxes": crop_boxes,
#             }
#             for loss_func, loss_w in zip(self.losses_func, self.loss_weights):
#                 loss_keys = loss_func.__call__.__code__.co_varnames
#                 loss_params = {
#                     key: kwargs_opt[key]
#                     for key in kwargs_opt.keys()
#                     if key in loss_keys
#                 }
#                 loss_v = loss_func(**loss_params) * loss_w

#                 key_prefix = type(loss_func).__name__ + "/" + str(nb_level) + "/"
#                 loss_v = {key_prefix + str(n): loss_v[n] for n in range(len(loss_v))}
#                 seg_losses.update(loss_v)

#             # from imageio import imwrite
#             # import numpy as np
#             # for k in range(ytrue.shape[0]):
#             #     true = ytrue[k,0].cpu().numpy().astype(float)
#             #     msk  = mask[k,0].cpu().numpy()
#             #     print(np.unique(true),np.unique(msk))
#             #     concat = true+msk
#             #     concat = 255*(concat-concat.min())/(concat.max()-concat.min()+1e-6)
#             #     concat = concat.astype(np.uint8)
#             #     # concat = (np.hstack([true,msk])*255).astype(np.uint8)
#             #     imwrite(str(k)+'.png', concat)
#             # sys.exit()
#         return seg_losses

#     def softmax_compute_seg_loss(self, seg_preds, targets, image_shape, eps=1e-6):
#         device = seg_preds[0].device
#         dtype = seg_preds[0].dtype
#         all_labels = [t["labels"] for t in targets]
#         ytrue = torch.stack([t["masks"] for t in targets], dim=0).long()
#         label_unique = torch.unique(torch.cat(all_labels, dim=0))
#         seg_losses = {}
#         for nb_level in range(len(seg_preds)):
#             preds = seg_preds[nb_level]
#             stride = image_shape[-1] / preds.shape[-1]

#             mask = preds.new_full(preds.shape, 0, device=preds.device)
#             crop_boxes = []
#             gt_boxes = []
#             for n_img, target in enumerate(targets):
#                 boxes = torch.round(target["boxes"] / stride).type(torch.int32)
#                 labels = target["labels"]
#                 for n in range(len(labels)):
#                     box = boxes[n, :]
#                     c = labels[n]  # .item()
#                     mask[n_img, c, box[1] : box[3] + 1, box[0] : box[2] + 1] = 1

#                     height, width = (box[2] - box[0] + 1) / 2.0, (
#                         box[3] - box[1] + 1
#                     ) / 2.0
#                     r = torch.sqrt(height**2 + width**2)
#                     cx = (box[2] + box[0] + 1) // 2
#                     cy = (box[3] + box[1] + 1) // 2
#                     # print('//// box ////',box, cx, cy, r)
#                     crop_boxes.append(torch.tensor([n_img, c, cx, cy, r]))
#                     gt_boxes.append(
#                         torch.tensor(
#                             [n_img, c, box[0], box[1], box[2], box[3]],
#                             dtype=torch.int32,
#                             device=device,
#                         )
#                     )
#             if len(crop_boxes) == 0:
#                 crop_boxes = torch.empty((0, 5), device=device)
#             else:
#                 crop_boxes = torch.stack(crop_boxes, dim=0)
#             if len(gt_boxes) == 0:
#                 gt_boxes = torch.empty((0, 6), device=device)
#             else:
#                 gt_boxes = torch.stack(gt_boxes, dim=0)

#             # print('#boxes',crop_boxes.shape[0],gt_boxes.shape[0])
#             assert crop_boxes.shape[0] == gt_boxes.shape[0]

#             kwargs_opt = {
#                 "ypred": preds,
#                 "ytrue": ytrue,
#                 "mask": mask,
#                 "gt_boxes": gt_boxes,
#                 "crop_boxes": crop_boxes,
#             }
#             for loss_func, loss_w in zip(self.losses_func, self.loss_weights):
#                 loss_keys = loss_func.__call__.__code__.co_varnames
#                 loss_params = {
#                     key: kwargs_opt[key]
#                     for key in kwargs_opt.keys()
#                     if key in loss_keys
#                 }
#                 loss_v = loss_func(**loss_params) * loss_w

#                 key_prefix = type(loss_func).__name__ + "/" + str(nb_level) + "/"
#                 loss_v = {key_prefix + str(n): loss_v[n] for n in range(len(loss_v))}
#                 seg_losses.update(loss_v)

#         return seg_losses

#     def _filter_targets(self, targets):
#         """Filter degenerate boxes from targets"""
#         filtered_targets = []
#         for target in targets:
#             boxes = target["boxes"]
#             if len(boxes) == 0:
#                 filtered_targets.append(target)
#                 continue

#             widths = boxes[:, 2] - boxes[:, 0]
#             heights = boxes[:, 3] - boxes[:, 1]
#             valid_mask = (widths > 0) & (heights > 0)

#             if valid_mask.all():
#                 filtered_targets.append(target)
#             else:
#                 filtered_target = {}
#                 for key, value in target.items():
#                     if isinstance(value, torch.Tensor) and len(value) == len(boxes):
#                         filtered_target[key] = value[valid_mask]
#                     else:
#                         filtered_target[key] = value
#                 filtered_targets.append(filtered_target)

#         return filtered_targets

#     def forward(self, images, targets=None):
#         # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
#         """
#         Arguments:
#             images (list[Tensor]): images to be processed
#             targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

#         Returns:
#             result (list[BoxList] or dict[Tensor]): the output from the model.
#                 During training, it returns a dict[Tensor] which contains the losses.
#                 During testing, it returns list[BoxList] contains additional fields
#                 like `scores`, `labels` and `mask` (for Mask R-CNN models).

#         """

#         if self.training and targets is None:
#             raise ValueError("In training mode, targets should be passed")

#         if targets is not None:
#             targets = self._filter_targets(targets)

#         if self.training:
#             assert targets is not None
#             for target in targets:
#                 boxes = target["boxes"]
#                 if isinstance(boxes, torch.Tensor):
#                     if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
#                         raise ValueError(
#                             "Expected target boxes to be a tensor"
#                             "of shape [N, 4], got {:}.".format(boxes.shape)
#                         )
#                 else:
#                     raise ValueError(
#                         "Expected target boxes to be of type "
#                         "Tensor, got {:}.".format(type(boxes))
#                     )

#         original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], [])
#         for img in images:
#             val = img.shape[-2:]
#             assert len(val) == 2
#             original_image_sizes.append((val[0], val[1]))

#         images, targets = self.batch_image(images, targets)
#         if torch.isnan(images.tensors).sum() > 0:
#             print("image is nan ..............")
#         if torch.isinf(images.tensors).sum() > 0:
#             print("image is inf ..............")

#         # Check for degenerate boxes
#         # TODO: Move this to a function
#         if targets is not None:
#             for target_idx, target in enumerate(targets):
#                 boxes = target["boxes"]
#                 degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
#                 if degenerate_boxes.any():
#                     # print the first degenerate box
#                     bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
#                     degen_bb: List[float] = boxes[bb_idx].tolist()
#                     raise ValueError(
#                         "All bounding boxes should have positive height and width."
#                         " Found invalid box {} for target at index {}.".format(
#                             degen_bb, target_idx
#                         )
#                     )

#         seg_preds = self.model(images.tensors)
#         # print('--------------image and seg outputs: ')
#         # print(images.tensors.shape, seg_preds[0].shape)
#         # for n_img, target in enumerate(targets):
#         #     print(n_img, target['masks'].shape)

#         ## calculate losses
#         assert targets is not None
#         if self.softmax:
#             losses = self.softmax_compute_seg_loss(
#                 seg_preds, targets, images.tensors.shape
#             )
#         else:
#             losses = self.sigmoid_compute_seg_loss(
#                 seg_preds, targets, images.tensors.shape
#             )

#         return losses, seg_preds


# # this error only occure in windows
# #  File "d:\w\noor\pytorch\main.py", line 14, in <module>
# #     main2()
# #     ~~~~~^^
# #   File "d:\w\noor\pytorch\main.py", line 9, in main2
# #     main()
# #     ~~~~^^
# #   File "d:\w\noor\pytorch\tools\train_promise_unetwithbox.py", line 555, in main
# #     metric_logger = train_one_epoch(
# #         model,
# #     ...<5 lines>...
# #         train_params["print_freq"],
# #     )
# #   File "d:\w\noor\pytorch\torchvision_wj\utils\engine.py", line 26, in train_one_epoch
# #     loss_dict, _ = model(images, targets)
# #                    ~~~~~^^^^^^^^^^^^^^^^^
# #   File "D:\w\noor\pytorch\.venv\Lib\site-packages\torch\nn\modules\module.py", line 1775, in _wrapped_call_impl
# #     return self._call_impl(*args, **kwargs)
# #            ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
# #   File "D:\w\noor\pytorch\.venv\Lib\site-packages\torch\nn\modules\module.py", line 1786, in _call_impl
# #     return forward_call(*args, **kwargs)
# #   File "d:\w\noor\pytorch\torchvision_wj\models\segwithbox\unetwithbox.py", line 221, in forward
# #     raise ValueError("All bounding boxes should have positive height and width."
# #                      " Found invalid box {} for target at index {}."
# #                      .format(degen_bb, target_idx))
# # ValueError: All bounding boxes should have positive height and width. Found invalid box [83.0, 102.0, 83.0, 102.0] for target at index 2.
# # (pytorch) PS D:\w\noor\pytorch>
