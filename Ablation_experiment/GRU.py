import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from keras.preprocessing.text import Tokenizer
import pandas as pd
import numpy as np
import random
from sklearn.metrics import confusion_matrix, accuracy_score, matthews_corrcoef, roc_auc_score


# 定义一个函数来设置numpy、random和torch的随机种子以确保可重复性
def set_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"随机种子设置为 {seed}")


# 设置随机种子以确保可重复性
set_seed(0)

# 从CSV文件读取训练和测试数据
df_train = pd.read_csv('./dataset/mix-true-train_RNA.csv')
df_test = pd.read_csv('./dataset/mix-true-test_RNA.csv')


# 定义一个函数来生成给定序列的输入序列
def seq_fun(seq, K=1):
    seq_list = []
    for x in range(len(seq) - K + 1):
        seq_list.append(seq[x:x + K].lower())
    return seq_list


# 定义一个函数来用零填充序列到最大长度
def pad_seq(X, maxlen, mode='constant'):
    padded_seqs = []
    for i in range(len(X)):
        pad_width = maxlen - len(X[i])
        padded_seqs.append(np.pad(X[i], pad_width=(0, pad_width), mode=mode, constant_values=0))
    return np.array(padded_seqs)


# 从训练数据生成输入序列并找到这些序列的最大长度
input_seqs_train = df_train['seq'].apply(lambda x: seq_fun(x, 1))
max_len = max(input_seqs_train.apply(len))

# 使用Keras Tokenizer对输入序列进行标记化
tokenizer = Tokenizer(num_words=None)
tokenizer.fit_on_texts(input_seqs_train)
sequences_train = tokenizer.texts_to_sequences(input_seqs_train)
sequences_train = pad_seq(sequences_train, maxlen=max_len)

# 从测试数据生成输入序列并使用与上述相同的Tokenizer进行标记化
input_seqs_test = df_test['seq'].apply(lambda x: seq_fun(x, 1))
sequences_test = tokenizer.texts_to_sequences(input_seqs_test)
sequences_test = pad_seq(sequences_test, maxlen=max_len)

# 将标签与输入数据分离以进行训练和测试
y_train = df_train['label']
y_test = df_test['label']
x_train = sequences_train
x_test = sequences_test

# 打乱训练数据
length = len(x_train)
permutation = np.random.permutation(length)
x_train = x_train[permutation]
y_train = y_train[permutation]

# 定义批量大小并为训练和测试创建DataLoader对象
batch_size = 512

train_data = TensorDataset(torch.from_numpy(x_train), torch.Tensor(y_train))
test_data = TensorDataset(torch.from_numpy(x_test), torch.Tensor(y_test))
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)

# 检查GPU是否可用并相应地设置设备
device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")


# 定义一个神经网络模型类，包括嵌入层和用于分类的线性层
class MyNet(nn.Module):
    # 初始化神经网络
    def __init__(self, vocab_size):
        super(MyNet, self).__init__()

        # 定义一些超参数
        self.n_layers = n_layers = 2
        self.hidden_dim = hidden_dim = 512
        embedding_dim = 600
        drop_prob = 0.3

        # 定义嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # 定义单向GRU层
        self.gru = nn.GRU(embedding_dim,
                          hidden_dim,
                          n_layers,
                          dropout=drop_prob,
                          bidirectional=False,
                          batch_first=True)

        # 定义全连接层
        self.fc = nn.Linear(in_features=hidden_dim,
                            out_features=1)

        # 定义sigmoid激活函数
        self.sigmoid = nn.Sigmoid()

        # 定义dropout层
        self.dropout = nn.Dropout(drop_prob)

    # 神经网络的前向传播
    def forward(self, x, hidden):
        batch_size = x.shape[0]

        # 将输入张量转换为long类型张量
        x = x.long()

        # 将输入张量通过嵌入层
        embeds = self.embedding(x)

        # 将嵌入张量通过单向GRU层
        gru_out, hidden = self.gru(embeds, hidden)

        # 对输出张量应用dropout
        out = self.dropout(gru_out)

        # 将输出张量通过全连接层
        out = self.fc(out)

        # 对输出张量应用sigmoid激活函数
        out = out.view(batch_size, -1)
        out = out[:, -1]
        out = self.sigmoid(out)

        return out, hidden

    # 初始化GRU的隐藏状态
    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)
        return hidden


# 创建神经网络实例
model = MyNet(12)

# 将模型移动到指定设备（例如GPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 定义损失函数
criterion = nn.BCELoss()

# 定义学习率
lr = 0.001

# 定义优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

# 定义训练的轮数
epochs = 100
losses = []

# 训练循环
for i in range(epochs):
    model.to(device)
    for inputs, labels in train_loader:
        # 初始化模型的隐藏状态
        h = model.init_hidden(len(inputs))

        # 设置模型为训练模式
        model.train()

        # 将隐藏状态移动到指定设备（例如GPU）
        h = h.to(device)

        # 将输入和标签移动到指定设备（例如GPU）
        inputs, labels = inputs.to(device), labels.to(device)

        # 清除所有优化变量的梯度
        model.zero_grad()

        # 前向传播：计算给定输入和隐藏状态的模型输出
        output, h = model(inputs, h)

        # 计算预测输出与真实标签之间的损失
        loss = criterion(output, labels.float())

        # 反向传播：计算所有优化变量相对于损失的梯度
        loss.backward()

        # 裁剪梯度以防止梯度爆炸
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)

        # 根据计算出的梯度更新优化变量
        optimizer.step()

    # 打印当前轮次的损失
    losses.append(loss.item())
    print("Epoch: {}/{}   ".format(i + 1, epochs), "Loss: {:.6f}   ".format(loss.item()))

# 将训练好的模型保存到文件
torch.save(model.cpu(), 'model.pth')

# 加载保存的模型
new_model = torch.load('model.pth')

# 设置模型为评估模式
new_model.eval()

# 初始化模型的隐藏状态
h = new_model.init_hidden(len(x_test))

# 对测试数据进行预测
output, h = new_model(torch.Tensor(x_test).long(), h)

# 将预测结果四舍五入为最接近的整数（0或1）
y_pred = torch.round(output).detach()

# 从混淆矩阵计算真阴性（tn）、假阳性（fp）、假阴性（fn）和真阳性（tp）
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

# 计算灵敏度、特异度、准确性和马修斯相关系数（MCC）
sen = tp / (tp + fn)
spe = tn / (tn + fp)
acc = accuracy_score(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)

# 计算受试者工作特征（ROC）曲线下面积（AUROC）
y_score = output.detach()  # 模型预测的正概率
auroc = roc_auc_score(y_test, y_score)

# 打印计算出的指标
print("Sensitivity: ", sen)
print("Specificity: ", spe)
print("Accuracy: ", acc)
print("MCC: ", mcc)
print("AUROC: ", auroc)
