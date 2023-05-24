import numpy as np
import torch

numpy_array = np.array([[1, 2, 3], [4, 5, 6]])
tensor = torch.from_numpy(numpy_array)
print(tensor)