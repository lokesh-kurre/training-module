import numpy as np
import pandas as pd
from functools import partial
from torch.utils.data import IterableDataset

from .data_gen import getGenerator
from training.utils.io import read_image

def read_func(args):
    image = read_image(args[0], cv2_imdecode_mode= 0)

    image = cv2.resize(image, (224, 224))
    image = np.stack([image, image, image], axis= 0) # HW -> CHW with C=3
    image = np.stack([image, image, image], axis= 0) # HW -> HWC with C=1

    return


class BaseDataset(IterableDataset):

    def __init__(self, data, read_func, input_dtypes= "(224, 224, 3)f4", **kwargs):
        super().__init__()
        self.data = pd.read_feather(data) if isinstance(data, str) else data
        self.generator = getGenerator(
            self.data.values.tolist(),
            read_func,
            input_dtypes,
            **kwargs
        )

    def __iter__(self):
        yield from self.generator()
