# import torch
from tools.train_promise_unetwithbox import main
import types, torch

torch._six = types.SimpleNamespace(string_classes=(str,), int_classes=(int,))


def main2():
    main()
    print("Hello from pytorch!", torch.cuda.is_available())


if __name__ == "__main__":
    main2()
