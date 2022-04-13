import torch
import numpy as np
import random

def fix_seeds(seed: int = 3407) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

class history_buffer:
    def __init__(self, buffer_size=50):
        self.current_size = 0
        self.buffer = list()
        self.buffer_size = buffer_size
    def __call__(self, newData):
        return_data = list()
        for data in newData:
            data = torch.unsqueeze(data, 0)
            if len(self.buffer) < self.buffer_size:
                self.buffer.append(data)
                return_data.append(data)
            else:
                random_val = random.uniform(0, 1)
                # If random_val > 0.5, return a random data from the history buffer
                if random_val > 0.5:
                    randomIdx = random.randint(0, self.buffer_size-1)
                    return_data.append(self.buffer[randomIdx])
                    self.buffer[randomIdx] = data
                else:
                    return_data.append(data)
        return_data = torch.cat(return_data, dim=0)
        return return_data