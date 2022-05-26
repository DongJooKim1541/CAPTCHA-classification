import os
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import random

#data_path = "./Large_captcha_dataset_png_small/"
data_path = "./Large_Captcha_Dataset/"

file_list = os.listdir(data_path)
print("len(file_list): ", len(file_list))

for idx, file in enumerate(file_list):
    if len(file.split(".")[0]) != 5:
        print("idx, file: ", idx, file)

# dataset random sampling, (file_list,data_size)
sampleList = random.sample(file_list, len(file_list))

file_list_train, file_list_test = train_test_split(sampleList, random_state=0)
print("len(file_list_train), len(file_list_test): ", len(file_list_train), len(file_list_test))

file_split = [file.split(".")[0] for file in sampleList]
file_split = "".join(file_split)
letters = sorted(list(set(list(file_split))))
print("len(letters): ", len(letters))
print("letters: ", letters)

vocabulary = ["-"] + letters
print("len(vocabulary): ", len(vocabulary))
print("vocabulary: ", vocabulary)
# mapping vocab and idx
idx2char = {k: v for k, v in enumerate(vocabulary, start=0)}
print("idx2char: ", idx2char)
char2idx = {v: k for k, v in idx2char.items()}
print("char2idx: ", char2idx)


# define dataloader class
class CAPTCHADataset(Dataset):

    def __init__(self, keyword):
        if keyword == "train":
            self.data_dir = data_path
            self.file_list = file_list_train
        elif keyword == "test":
            self.data_dir = data_path
            self.file_list = file_list_test

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        label = self.file_list[index]
        img = os.path.join(self.data_dir, label)
        img = Image.open(img).convert('RGB')
        img = self.transform(img)
        label = label.split(".")[0]
        return img, label

    def transform(self, image):
        transform_ops = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        return transform_ops(image)
