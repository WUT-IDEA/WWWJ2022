# 超参数
hidden_dropout_prob = 0.3
num_labels = 2
learning_rate = 1e-5
weight_decay = 1e-2
epochs = 1
batch_size = 16
max_len = 300
save_path = "QA/model_1"
valid_file = "final_test_step1.csv"

import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import ElectraConfig, ElectraForQuestionAnswering


# 使用GPU
# 通过model.to(device)的方式使用
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config = ElectraConfig.from_pretrained(save_path, num_labels=num_labels, hidden_dropout_prob=hidden_dropout_prob)
model = ElectraForQuestionAnswering.from_pretrained(save_path, config=config)
model.to(device)



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
        print("Unexpected input")
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
# 定义优化器
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
criterion = nn.CrossEntropyLoss()

from torch.utils.data import DataLoader
data_path = "my_data/"

class LegalDataset(Dataset):
    def __init__(self, path_to_file):
        self.dataset = pd.read_csv(path_to_file, sep="\t", names=["text", "label", "charge", "pre_label"])
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        text = self.dataset.loc[idx, "text"]
        label = self.dataset.loc[idx, "label"]
        charge = self.dataset.loc[idx, "charge"]
        pre_label = self.dataset.loc[idx, "pre_label"]
        sample = {"text": text, "label": label, "charge": charge, "pre_label": pre_label}
        return sample
    
# 加载数据集
valid_set = LegalDataset(data_path + valid_file)
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=4)

import numpy as np

def evaluate(model, iterator, criterion, device):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    with torch.no_grad():
        for _, batch in enumerate(iterator):
            label = batch["label"]
            text = batch["text"]
            start_positions = batch["label"]
            end_positions = batch["label"]
            input_ids, token_type_ids = convert_text_to_ids(tokenizer, text)
            input_ids = seq_padding(tokenizer, input_ids)
            token_type_ids = seq_padding(tokenizer, token_type_ids)
            # 标签形状为 (batch_size, 1) 
            start_positions = start_positions.unsqueeze(1)
            end_positions = end_positions.unsqueeze(1)
            label = label.unsqueeze(1)
            # 需要 LongTensor
            input_ids, token_type_ids, start_positions, end_positions, label = input_ids.long(), token_type_ids.long(), start_positions.long(), end_positions.long(), label.long()
            # 迁移到GPU
            input_ids, token_type_ids, start_positions, end_positions, label = input_ids.to(device), token_type_ids.to(device), start_positions.to(device), end_positions.to(device), label.to(device)
            output = model(input_ids=input_ids, token_type_ids=token_type_ids, labels=label, start_positions=start_positions, end_positions=end_positions)
            y_pred_label = output[1].argmax(dim=1)
            loss1 = (output[0]-output[2])/6 #total_loss, logits2, loss2
            loss = loss1+output[2]*5/6
            acc = ((y_pred_label == label.view(-1)).sum()).item()

            file=open(save_path + "/test_result.txt",'a')
            file.write(str(y_pred_label)+'\n')

            epoch_loss += loss.item()
            epoch_acc += acc
    return epoch_loss / len(iterator), epoch_acc / len(iterator.dataset.dataset)


# 训练
for i in range(epochs):
    valid_loss, valid_acc = evaluate(model, valid_loader, criterion, device)
    # print("valid loss: ", valid_loss, "\t", "valid acc:", valid_acc)

print("success")
