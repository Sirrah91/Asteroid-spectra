import numpy as np
from typing import List

from modules.NN_config_taxonomy import classes, classes2


def convert_classes(args: np.ndarray) -> List[float]:
    # This function convert classes to numbers which is required by the neural network

    return [classes[x] for x in args.ravel()]


def convert_classes2(args: np.ndarray) -> List[str]:
    # This function convert numbers to classes

    return [classes2[x] for x in args.ravel()]


def print_conversion_chart() -> None:
    print("Conversion chart:")
    print("".join("{key}\t=\t{value}\t\t".format(key=k, value=v)
                  if (v % 5 > 0 or v == 0) else "\n{key}\t=\t{value}\t\t".format(key=k, value=v)
                  for k, v in classes.items()))
