import numpy as np
from torch.utils.data import Dataset, DataLoader
import soundfile as sf
from torch import Tensor
import os

def pad_random(x, max_len=64600):
    x_len = x.shape[0]

    if x_len > max_len:
        stt = np.random.randint(x_len - max_len)
        return x[stt:stt + max_len]

    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, num_repeats)[:max_len]
    return padded_x


def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]

    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


class EvalDataset(Dataset):
    def __init__(self, ids, dir_path, pad_fn=pad_random, cut=64600):
        self.ids = ids
        self.dir_path = dir_path
        self.cut = cut
        self.pad_fn = pad_fn

    def __getitem__(self, index):
        path_to_wav = f"{self.dir_path}/{self.ids[index]}"
        audio, _ = sf.read(path_to_wav)
        x_pad = self.pad_fn(audio, self.cut)
        x_inp = Tensor(x_pad)
        return x_inp, self.ids[index]

    def __len__(self):
        return len(self.ids)


def get_data_for_dataset(path):
    ids_list = []
    label_list = []
    with open(path, "r") as file:
        for line in file:
            line = line.split()
            id, label = line[1], line[-1]
            ids_list.append(id)
            label = 1 if label == "bonafide" else 0
            label_list.append(label)
    return ids_list, label_list


def get_data_for_evaldataset(path):
    ids_list = os.listdir(path)
    return ids_list


def get_dataloaders(datasets, config):
    eval_loader = DataLoader(
        datasets,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"]
    )
    return eval_loader
