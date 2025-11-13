# import torch
# from torchvision_wj.models.segwithbox.boosting_ensemble import (
#     test_create_boosting_ensemble_fixed,
# )

# from torchvision_wj.models.segwithbox.boosting_ensemble_test import test
# import types, torch

# torch._six = types.SimpleNamespace(string_classes=(str,), int_classes=(int,))


from torchvision_wj.utils.engin2 import test_tensorboard_logging


def main2():
    # test_create_boosting_ensemble_fixed()
    test_tensorboard_logging()
    # print("Hello from pytorch!", torch.cuda.is_available())


if __name__ == "__main__":
    main2()
