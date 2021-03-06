# 超参数
hidden_dropout_prob = 0.3
num_labels = 2
learning_rate = 1e-5
weight_decay = 1e-2
epochs = 3
batch_size = 16
max_len = 256
import logging
import torch
from torch.utils.data import Dataset
import pandas as pd

class SentimentDataset(Dataset):
    def __init__(self, path_to_file):
        self.dataset = pd.read_csv(path_to_file, sep="\t", names=["text", "label"])
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        text = self.dataset.loc[idx, "text"]
        label = self.dataset.loc[idx, "label"]
        sample = {"text": text, "label": label}
        return sample
    
from transformers import ElectraConfig, ElectraForSequenceClassification

# 使用GPU
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
config = ElectraConfig.from_pretrained("hfl/chinese-electra-180g-base-discriminator", num_labels=num_labels, hidden_dropout_prob=hidden_dropout_prob)
model = ElectraForSequenceClassification.from_pretrained("hfl/chinese-electra-180g-base-discriminator", config=config)
model.to(device)


from torch.utils.data import DataLoader
data_path = "my_data/"
# 加载数据集
sentiment_train_set = SentimentDataset(data_path + "data_train1.csv")
sentiment_train_loader = DataLoader(sentiment_train_set, batch_size=batch_size, shuffle=True, num_workers=2)

sentiment_valid_set = SentimentDataset(data_path + "test_data_subelement.csv")
sentiment_valid_loader = DataLoader(sentiment_valid_set, batch_size=batch_size, shuffle=False, num_workers=2)

from transformers import ElectraTokenizer

tokenizer = ElectraTokenizer.from_pretrained('hfl/chinese-electra-180g-base-discriminator')

def convert_text_to_ids(tokenizer, text, max_len=max_len):
    if isinstance(text, str):
        tokenized_text = tokenizer.encode_plus(text, max_length=max_len, add_special_tokens=True)
        input_ids = tokenized_text["input_ids"]
        token_type_ids = tokenized_text["token_type_ids"]
    elif isinstance(text, list):
        input_ids = []
        token_type_ids = []
        for t in text:
            tokenized_text = tokenizer.encode_plus(t, max_length=max_len, add_special_tokens=True)
            input_ids.append(tokenized_text["input_ids"])
            token_type_ids.append(tokenized_text["token_type_ids"])
    else:
        logging.info("Unexpected input")
    return input_ids, token_type_ids

def seq_padding(tokenizer, X):
    pad_id = tokenizer.convert_tokens_to_ids("[PAD]")
    if len(X) <= 1:
        return torch.tensor(X)
    L = [len(x) for x in X]
    ML = max(L)
    X = torch.Tensor([x + [pad_id] * (ML - len(x)) if len(x) < ML else x for x in X])
    return X

import torch
import torch.nn as nn
from transformers import AdamW
# 定义优化器和损失函数
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
criterion = nn.CrossEntropyLoss()

def train(model, iterator, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    for i, batch in enumerate(iterator):
        label = batch["label"]
        text = batch["text"]
        input_ids, token_type_ids = convert_text_to_ids(tokenizer, text)
        input_ids = seq_padding(tokenizer, input_ids)
        token_type_ids = seq_padding(tokenizer, token_type_ids)
        # 标签形状为 (batch_size, 1) 
        label = label.unsqueeze(1)
        # 需要 LongTensor
        input_ids, token_type_ids, label = input_ids.long(), token_type_ids.long(), label.long()
        # 梯度清零
        optimizer.zero_grad()
        # 迁移到GPU
        input_ids, token_type_ids, label = input_ids.to(device), token_type_ids.to(device), label.to(device)
        output = model(input_ids=input_ids, token_type_ids=token_type_ids, labels=label)
        y_pred_prob = output[1]
        y_pred_label = y_pred_prob.argmax(dim=1)
        # 计算loss
        # 这个 loss 和 output[0] 是一样的
        loss = criterion(y_pred_prob.view(-1, 2), label.view(-1))
        #loss = output[0]
        # 计算acc
        acc = ((y_pred_label == label.view(-1)).sum()).item()
        # 反向传播
        loss.backward()
        optimizer.step()
        # epoch 中的 loss 和 acc 累加
        epoch_loss += loss.item()
        epoch_acc += acc
        if i % 200 == 0:
            print("current loss:", epoch_loss / (i+1), "\t", "current acc:", epoch_acc / ((i+1)*len(label)))
    return epoch_loss / len(iterator), epoch_acc / len(iterator.dataset.dataset)

def evaluate(model, iterator, criterion, device):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    with torch.no_grad():
        for _, batch in enumerate(iterator):
            label = batch["label"]
            text = batch["text"]
            input_ids, token_type_ids = convert_text_to_ids(tokenizer, text)
            input_ids = seq_padding(tokenizer, input_ids)
            token_type_ids = seq_padding(tokenizer, token_type_ids)
            label = label.unsqueeze(1)
            input_ids, token_type_ids, label = input_ids.long(), token_type_ids.long(), label.long()
            input_ids, token_type_ids, label = input_ids.to(device), token_type_ids.to(device), label.to(device)
            output = model(input_ids=input_ids, token_type_ids=token_type_ids, labels=label)
            y_pred_label = output[1].argmax(dim=1)
            loss = output[0]
            acc = ((y_pred_label == label.view(-1)).sum()).item()
            epoch_loss += loss.item()
            epoch_acc += acc
    return epoch_loss / len(iterator), epoch_acc / len(iterator.dataset.dataset)

    # 再测试
for i in range(epochs):
    train_loss, train_acc = train(model, sentiment_train_loader, optimizer, criterion, device)
#     model.save_pretrained('electra_pytorch')
    print("train loss: ", train_loss, "\t", "train acc:", train_acc)
    valid_loss, valid_acc = evaluate(model, sentiment_valid_loader, criterion, device)
    print("valid loss: ", valid_loss, "\t", "valid acc:", valid_acc)
model.save_pretrained('electra_pytorch12')
