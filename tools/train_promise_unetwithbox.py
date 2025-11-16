import datetime
import os
import time
import numpy as np
import pandas as pd
import matplotlib

from torchvision_wj.utils.engin2 import TensorBoardLogger

matplotlib.use("PDF")
import matplotlib.pylab as plt
import warnings

warnings.filterwarnings("ignore")

import torch

torch.multiprocessing.set_sharing_strategy("file_system")
from torchvision_wj.datasets.transform import random_transform_generator
from torchvision_wj.datasets.image import random_visual_effect_generator
from torchvision_wj.datasets.samplers import PatientSampler
from torchvision_wj.models.segwithbox.unetwithbox import UNetWithBox
from torchvision_wj.models.segwithbox.default_unet_net import *
from torchvision_wj.utils.losses import *
from torchvision_wj.utils.early_stop import EarlyStopping
from torchvision_wj.utils import config, utils
from torchvision_wj.utils.promise_utils import get_promise
from torchvision_wj.utils.engine import train_one_epoch, validate_loss
import torchvision_wj.utils.transforms as T
from torchvision_wj._C_promise import _C as _C2

# where is the best place to save output images in the main function or in train_one_epoch function


# def train_one_epoch(
#     model, optimizer, data_loader, device, epoch, clipnorm=0.001, print_freq=20
# ):
#     time.sleep(2)  # Prevent possible deadlock during epoch transition
#     model.train()
#     metric_logger = utils.MetricLogger(delimiter="  ")
#     metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
#     header = "Epoch: [{}]".format(epoch)

#     lr_scheduler = None
#     if epoch == 0:
#         warmup_factor = 1.0 / 1000
#         warmup_iters = min(1000, len(data_loader) - 1)

#         lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

#     for images, targets in metric_logger.log_every(data_loader, print_freq, header):
#         images = list(image.to(device) for image in images)
#         targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

#         loss_dict, _ = model(images, targets)

#         losses = sum(loss for loss in loss_dict.values())

#         # reduce losses over all GPUs for logging purposes
#         loss_dict_reduced = utils.reduce_dict(loss_dict)
#         losses_reduced = sum(loss for loss in loss_dict_reduced.values())

#         loss_value = losses_reduced.item()

#         if not math.isfinite(loss_value):
#             print("Loss is {}, stopping training".format(loss_value))
#             print(loss_dict_reduced)
#             sys.exit(1)

#         optimizer.zero_grad()
#         losses.backward()
#         if clipnorm is not None:
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clipnorm)
#         optimizer.step()

#         if lr_scheduler is not None:
#             lr_scheduler.step()

#         metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
#         metric_logger.update(lr=optimizer.param_groups[0]["lr"])

#     return metric_logger


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n_exp", default=0, type=int, help="the index of experiments")
    args = parser.parse_args()

    utils.init_distributed_mode(args)
    print(args)

    n_exp = args.n_exp
    cfg = {
        "train_params": {"patience": 8, "lr": 1e-4, "batch_size": 6},
        "data_params": {"workers": 4},
    }
    _C = config.config_updates(_C2, cfg)
    if n_exp == 0:  # full supervision
        config1 = {
            "net_params": {
                "softmax": False,
                "losses": [("CrossEntropyLoss", {"mode": "all"}, 1)],
                # "model_name": "deep_labv3",
                # "model_name": "unet_residual",
                # "model_name": "boosting",
                # "model_name": "enet32",
                # "model_name": "sequential_boosting_v3",
                # "model_name": "adaptive_boosting_v3",
                # "model_name": "progressive_boosting_v3",
                # "model_name": "simple_unet",
                # "model_name": "simple_deep_labv3",
                # "model_name": "simple_res_net",
                # "model_name": "sequential_boosting_1",
                # "model_name": "sequential_boosting_2",
                # "model_name": "sequential_boosting_3",
                "model_name": "sequential_boosting_4",
                # "model_name": "sequential_boosting_5",
                # "model_name": "sequential_boosting_6",
            },
            # "save_params": {"experiment_name": "deep_lab_v3"},
            # "save_params": {"experiment_name": "residual_unet"},
            # "save_params": {"experiment_name": "boosting"},
            # "save_params": {"experiment_name": "enet32"},
            # "save_params": {"experiment_name": "progressive_boosting_v3"},
            # "save_params": {"experiment_name": "simple_unet"},
            # "save_params": {"experiment_name": "simple_res_net"},
            # "save_params": {"experiment_name": "simple_deep_labv3"},
            # "save_params": {"experiment_name": "sequential_boosting_1"},
            # "save_params": {"experiment_name": "sequential_boosting_2"},
            # "save_params": {"experiment_name": "sequential_boosting_3"},
            "save_params": {"experiment_name": "sequential_boosting_4"},
            # "save_params": {"experiment_name": "sequential_boosting_5"},
            # "save_params": {"experiment_name": "sequential_boosting_6"},
        }
        _C_array = []
        for c in [config1]:
            _C_array.append(config.config_updates(_C, c))

    for _C_used in _C_array:
        assert (
            _C_used["save_params"]["experiment_name"] is not None
        ), "experiment_name has to be set"

        train_params = _C_used["train_params"]
        data_params = _C_used["data_params"]
        net_params = _C_used["net_params"]
        dataset_params = _C_used["dataset"]
        save_params = _C_used["save_params"]
        data_visual_aug = data_params["data_visual_aug"]
        data_transform_aug = data_params["data_transform_aug"]

        output_dir = os.path.join(
            save_params["dir_save"], save_params["experiment_name"]
        )
        os.makedirs(output_dir, exist_ok=True)
        if not train_params["test_only"]:
            config.save_config_file(os.path.join(output_dir, "config.yaml"), _C_used)
            print("saving files to {:s}".format(output_dir))

        device = torch.device(_C_used["device"])

        def get_transform():
            transforms = []
            transforms.append(T.ToTensor())
            transforms.append(T.Normalizer(mode=data_params["normalizer_mode"]))
            return T.Compose(transforms)

        if data_transform_aug["aug"]:
            transform_generator = random_transform_generator(
                min_rotation=data_transform_aug["min_rotation"],
                max_rotation=data_transform_aug["max_rotation"],
                min_translation=data_transform_aug["min_translation"],
                max_translation=data_transform_aug["max_translation"],
                min_shear=data_transform_aug["min_shear"],
                max_shear=data_transform_aug["max_shear"],
                min_scaling=data_transform_aug["min_scaling"],
                max_scaling=data_transform_aug["max_scaling"],
                flip_x_chance=data_transform_aug["flip_x_chance"],
                flip_y_chance=data_transform_aug["flip_y_chance"],
            )
        else:
            transform_generator = None
        if data_visual_aug["aug"]:
            visual_effect_generator = random_visual_effect_generator(
                contrast_range=data_visual_aug["contrast_range"],
                brightness_range=data_visual_aug["brightness_range"],
                hue_range=data_visual_aug["hue_range"],
                saturation_range=data_visual_aug["saturation_range"],
            )
        else:
            visual_effect_generator = None
        print("---data augmentation---")
        print("transform: ", transform_generator)
        print("visual: ", visual_effect_generator)

        # Data loading code
        print("Loading data")
        dataset = get_promise(
            root=dataset_params["root_path"],
            image_folder=dataset_params["train_path"][0],
            gt_folder=dataset_params["train_path"][1],
            transforms=get_transform(),
            transform_generator=transform_generator,
            visual_effect_generator=visual_effect_generator,
        )
        dataset_test = get_promise(
            root=dataset_params["root_path"],
            image_folder=dataset_params["valid_path"][0],
            gt_folder=dataset_params["valid_path"][1],
            transforms=get_transform(),
            transform_generator=None,
            visual_effect_generator=None,
        )

        print("Creating data loaders")
        train_sampler = torch.utils.data.RandomSampler(dataset)
        train_batch_sampler = torch.utils.data.BatchSampler(
            train_sampler, train_params["batch_size"], drop_last=True
        )

        # test_sampler = torch.utils.data.SequentialSampler(dataset_test)
        test_patient_sampler = PatientSampler(
            dataset_test, dataset_params["grp_regex"], shuffle=False
        )

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=train_batch_sampler,
            num_workers=data_params["workers"],
            collate_fn=utils.collate_fn,
            pin_memory=True,
        )

        data_loader_test = torch.utils.data.DataLoader(
            dataset_test,
            batch_size=1,
            batch_sampler=test_patient_sampler,
            num_workers=data_params["workers"],
            collate_fn=utils.collate_fn,
            pin_memory=True,
        )

        print("Creating model with parameters: {}".format(net_params))
        net = eval(net_params["model_name"])(
            net_params["input_dim"],
            net_params["seg_num_classes"],
            net_params["softmax"],
        )
        losses, loss_weights = [], []
        for loss in net_params["losses"]:
            losses.append(eval(loss[0])(**loss[1]))
            loss_weights.append(loss[2])
        model = UNetWithBox(net, losses, loss_weights, softmax=net_params["softmax"])
        model.to(device)

        params = [p for p in model.parameters() if p.requires_grad]
        if train_params["optimizer"] == "SGD":
            optimizer = torch.optim.SGD(
                params,
                lr=train_params["lr"],
                momentum=train_params["momentum"],
                weight_decay=train_params["weight_decay"],
            )
        elif train_params["optimizer"] == "Adam":
            optimizer = torch.optim.Adam(
                params, lr=train_params["lr"], betas=train_params["betas"]
            )

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=train_params["factor"],
            patience=train_params["patience"],
        )
        early_stop = EarlyStopping(patience=2 * train_params["patience"])
        # model_without_ddp = model.model
        if train_params["resume"]:
            print("resuming model {}".format(train_params["resume"]))
            # checkpoint = torch.load(
            #     os.path.join(output_dir, train_params["resume"]), map_location="cpu"
            # )
            # model_without_ddp.load_state_dict(checkpoint["model"])
            # optimizer.load_state_dict(checkpoint["optimizer"])
            # lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            # train_params["start_epoch"] = checkpoint["epoch"] + 1

        tb_logger = TensorBoardLogger(log_dir=os.path.join(output_dir, "tensorboard"))
        torch.autograd.set_detect_anomaly(True)
        model.training = True
        if train_params["test_only"]:
            val_metric_logger = validate_loss(
                model, data_loader_test, device, tb_logger=tb_logger, is_test=True
            )
        else:
            print("Start training")
            start_time = time.time()
            summary = {"epoch": []}
            # Initialize TensorBoard logger

            for epoch in range(train_params["start_epoch"], train_params["epochs"]):
                if args.distributed:
                    train_sampler.set_epoch(epoch)

                metric_logger = train_one_epoch(
                    model,
                    optimizer,
                    data_loader,
                    device,
                    epoch,
                    train_params["clipnorm"],
                    train_params["print_freq"],
                    tb_logger=tb_logger,
                    save_images_freq=50,  # Save images every 50 batches
                )

                if output_dir:
                    utils.save_on_master(
                        {
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "lr_scheduler": lr_scheduler.state_dict(),
                            "args": args,
                            "epoch": epoch,
                        },
                        os.path.join(output_dir, "model_{:02d}.pth".format(epoch)),
                    )

                # evaluate after every epoch
                val_metric_logger = validate_loss(
                    model, data_loader_test, device, tb_logger=tb_logger, epoch=epoch
                )

                # collect the results and save
                summary["epoch"].append(epoch)
                for name, meter in metric_logger.meters.items():
                    if name == "lr":
                        v = meter.global_avg
                    else:
                        v = float(np.around(meter.global_avg, 4))
                    if epoch == 0:
                        summary[name] = [v]
                    else:
                        summary[name].append(v)
                for name, meter in val_metric_logger.meters.items():
                    v = float(np.around(meter.global_avg, 4))
                    if epoch == 0:
                        summary[name] = [v]
                    else:
                        summary[name].append(v)
                summary_save = pd.DataFrame(summary)
                summary_save.to_csv(
                    os.path.join(output_dir, "summary.csv"), index=False
                )

                # update lr scheduler
                val_loss = val_metric_logger.meters["val_loss"].global_avg
                lr_scheduler.step(val_loss)

                # # early stop check
                # if early_stop.step(val_loss):
                #     print('Early stop at epoch = {}'.format(epoch))
                #     break

            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print("Training time {}".format(total_time_str))

            ## plot training and validation loss
            plt.figure()
            plt.plot(summary_save["epoch"], summary_save["loss"], "-ro", label="train")
            plt.plot(
                summary_save["epoch"], summary_save["val_loss"], "-g+", label="valid"
            )
            plt.legend(loc=0)
            plt.savefig(os.path.join(output_dir, "loss.jpg"))
