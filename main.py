import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from Network import *
from captcha_dataset import CAPTCHADataset, char2idx, idx2char, file_list_train, file_list_test

""" Device Confirmation """
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print('Using PyTorch version: ', torch.__version__, ' Device: ', device)

"""Paramenter generalization"""
batch_size = 64
num_epochs = 100
lr = 0.001
weight_decay = 1e-3
clip_norm = 5

"""Data preprocessing"""
trainset = CAPTCHADataset("train")
testset = CAPTCHADataset("test")
train_loader = DataLoader(trainset, batch_size=batch_size, num_workers=0, shuffle=True)
test_loader = DataLoader(testset, batch_size=batch_size, num_workers=0, shuffle=True)

"""Confirm data"""
print("len(train_loader), len(test_loader): ", len(train_loader), len(test_loader))

image, label = iter(train_loader).__next__()
print("image.size(), label: ", image.size(), label)

num_chars = len(char2idx)
print("num_chars: ", num_chars)
rnn_hidden_size = 256

"""Confirm optimizer, objective function"""

crnn = CRNN(num_chars, rnn_hidden_size=rnn_hidden_size)
crnn.apply(weights_init)
crnn = crnn.to(device)
optimizer = optim.Adam(crnn.parameters(), lr=lr, weight_decay=weight_decay)
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=5)

label_logits = crnn(image.to(device))  # torch.Size([T, batch_size, num_classes])
print("label: ", label)
print("label_logits.shape: ", label_logits.shape)  # torch.Size([T, batch_size, num_classes])

# If output is "aaaabbbbbccddeeff", it can be summarized as "abcedf"
criterion = nn.CTCLoss(blank=0)

"""Encoding label to tensor"""


def encode_label(label):
    label_targets_lens = [len(text) for text in label]
    # len(label_targets_lens): 64, label_targets_lens: [5,5,...,5]
    label_targets_lens = torch.IntTensor(label_targets_lens)
    # print(label_targets_lens) # tensor([5, 5,...,5], dtype=torch.int32)
    label_concat = "".join(label)
    # print(label_concat) # v42r81I8r2226378o...
    label_targets = [char2idx[c] for c in label_concat]
    # print(label_targets) # [58, 5, 3, 54, 9, 2,...,10, 7, 2]
    label_targets = torch.IntTensor(label_targets)
    # print(label_targets) # tensor([58, 5, 3,..., 10,  7,  2], dtype=torch.int32)

    return label_targets, label_targets_lens


"""Compute loss"""


def compute_loss(label, label_logits):
    label_logps = F.log_softmax(label_logits, 2)  # [T, batch_size, num_classes], num_classes computation
    label_logps_lens = torch.full(size=(label_logps.size(1),),  # batch_size
                                  fill_value=label_logps.size(0),  # num of char,T
                                  dtype=torch.int32).to(device)  # [batch_size]
    label_targets, label_targets_lens = encode_label(label)
    loss = criterion(label_logps, label_targets, label_logps_lens, label_targets_lens)
    # CTCLoss([T, batch_size, num_classes],[T * batch_size],[batch_size],[batch_size])

    return loss


"""Compute loss"""
compute_loss(label, label_logits)

"""decode prediction labels to text"""


def decode_predictions(label_logits):
    label_tokens = F.softmax(label_logits, 2).argmax(2)  # [T, batch_size], softmax for num_chars
    # print(F.softmax(label_logits, 2).size()) #torch.Size([T, batch_size, num_chars])
    # print("F.softmax(label_logits, 2).argmax(0)", F.softmax(label_logits, 2).argmax(0)) #[batch_size,num_chars], max of T
    # print("F.softmax(label_logits, 2).argmax(1)",F.softmax(label_logits, 2).argmax(1)) #[T,num_chars], max of batch_size
    # print("F.softmax(label_logits, 2).argmax(2)", F.softmax(label_logits, 2).argmax(2)) #[T,batch_size], max of num_chars
    # print(label_tokens.size()) # label_tokens.size():  torch.Size([T, batch_size])
    label_tokens = label_tokens.numpy().T  # [batch_size, T], transpose matrix
    # print("label_tokens: ", label_tokens) # [batch_size, T]
    # decode idx to char
    label_tokens_new = []
    for text_tokens in label_tokens:
        text = [idx2char[idx] for idx in text_tokens]
        text = "".join(text)
        label_tokens_new.append(text)
    # print("label_tokens_new: ", label_tokens_new)
    return label_tokens_new


"""Compare prediction and ground truth to gain accuracy"""


def compare_label(label, label_pred):
    correct = 0
    for i in range(len(label_pred)):
        if label[i] == label_pred[i]:
            correct += 1
    return correct, len(label_pred)


""" Model train """
train_epoch_loss = []
train_epoch_acc = []
num_updates_epochs = []
for epoch in range(1, num_epochs + 1):
    epoch_loss_list = []
    train_acc_list = []
    num_updates_epoch = 0
    for image, label in train_loader:
        optimizer.zero_grad()
        label_logits = crnn(image.to(device))
        label_pred = decode_predictions(label_logits.cpu())
        train_correct, check = compare_label(label, label_pred)
        train_iteration_accuracy = train_correct / check

        loss = compute_loss(label, label_logits)
        iteration_loss = loss.item()
        # If iteration loss is NaN or inf, ignore it
        if np.isnan(iteration_loss) or np.isinf(iteration_loss):
            continue

        num_updates_epoch += 1
        epoch_loss_list.append(iteration_loss)
        train_acc_list.append(train_iteration_accuracy)
        loss.backward()
        nn.utils.clip_grad_norm_(crnn.parameters(), clip_norm)
        optimizer.step()
    # Mean the iteration loss to 1 epoch loss
    epoch_loss = np.mean(epoch_loss_list)
    epoch_acc=np.mean(train_acc_list)
    print("Epoch:{}    Train loss:{}    Train acc:{}    NumUpdates:{}".format(epoch, epoch_loss, epoch_acc,num_updates_epoch))
    train_epoch_loss.append(epoch_loss)
    train_epoch_acc.append(epoch_acc)
    num_updates_epochs.append(num_updates_epoch)
    lr_scheduler.step(epoch_loss)

""" Test """
test_accuracy_list=[]
test_loss_list=[]
for epoch in range(1, num_epochs + 1):
    with torch.no_grad():
        test_correct = 0
        test_check = 0
        test_epoch_loss_list=[]
        test_acc_iteration_list = []
        for image, label in test_loader:
            test_check += 1
            label_logits = crnn(image.to(device))  # [width, batch_size, num_classes==num_features]
            label_pred = decode_predictions(label_logits.cpu())
            loss = compute_loss(label, label_logits)
            iteration_loss = loss.item()
            # print(label, label_pred)
            correct, check = compare_label(label, label_pred)
            test_correct += correct
            test_check += check
            test_accuracy = test_correct / test_check
            test_epoch_loss_list.append(iteration_loss)
            test_acc_iteration_list.append(test_accuracy)
        test_accuracy=np.mean(test_acc_iteration_list)
        test_loss=np.mean(test_epoch_loss_list)
        print("Epoch:{}    Test loss:{}    Test acc:{}".format(epoch, test_loss, test_accuracy))
        test_accuracy_list.append(test_accuracy)
        test_loss_list.append(test_loss)


if __name__ == '__main__':
    fig=plt.figure(figsize=(12,20))
    x=np.linspace(1,num_epochs,num_epochs)

    ax1=fig.add_subplot(2,1,1)
    ax1.plot(x,train_epoch_loss,label="Train loss")
    ax1.plot(x,test_loss_list,label="Test loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss")
    ax1.legend()

    ax2=fig.add_subplot(2,1,2)
    ax2.plot(x, train_epoch_acc,label="Train acc")
    ax2.plot(x, test_accuracy_list,label="Test acc")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy")
    ax2.legend()
    print("train_accuracy: ", train_epoch_acc[-1])
    print("test_accuracy: ", test_accuracy_list[-1])


    plt.show()
