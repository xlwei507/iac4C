import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from keras.preprocessing.text import Tokenizer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.metrics import confusion_matrix, accuracy_score, matthews_corrcoef, roc_auc_score


# Define a function to set random seeds for numpy, random, and torch to ensure reproducibility
def set_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set as {seed}")


# Set random seeds for reproducibility
set_seed(0)

# Read the training and testing data from CSV files
df_train = pd.read_csv('./dataset/mix-true-train_RNA.csv')
df_test = pd.read_csv('./dataset/mix-true-test_RNA.csv')


# Define a function to generate input sequences from a given sequence
def seq_fun(seq, K=1):
    seq_list = []
    for x in range(len(seq) - K + 1):
        seq_list.append(seq[x:x + K].lower())
    return seq_list


# Define a function to pad sequences with zeros to a maximum length
def pad_seq(X, maxlen, mode='constant'):
    padded_seqs = []
    for i in range(len(X)):
        pad_width = maxlen - len(X[i])
        padded_seqs.append(np.pad(X[i], pad_width=(0, pad_width), mode=mode, constant_values=0))
    return np.array(padded_seqs)


# Generate input sequences from the training data and find the maximum length of these sequences
input_seqs_train = df_train['seq'].apply(lambda x: seq_fun(x, 1))
max_len = max(input_seqs_train.apply(len))

# Tokenize the input sequences using the Keras Tokenizer
tokenizer = Tokenizer(num_words=None)
tokenizer.fit_on_texts(input_seqs_train)
sequences_train = tokenizer.texts_to_sequences(input_seqs_train)
sequences_train = pad_seq(sequences_train, maxlen=max_len)

# Generate input sequences from the testing data and tokenize them using the same Tokenizer as above
input_seqs_test = df_test['seq'].apply(lambda x: seq_fun(x, 1))
sequences_test = tokenizer.texts_to_sequences(input_seqs_test)
sequences_test = pad_seq(sequences_test, maxlen=max_len)

# Separate the labels from the input data for training and testing
y_train = df_train['label']
y_test = df_test['label']
x_train = sequences_train
x_test = sequences_test

# Shuffle the training data
length = len(x_train)
permutation = np.random.permutation(length)
x_train = x_train[permutation]
y_train = y_train[permutation]

# Define the batch size and create DataLoader objects for training and testing
batch_size = 512

train_data = TensorDataset(torch.from_numpy(x_train), torch.Tensor(y_train))
test_data = TensorDataset(torch.from_numpy(x_test), torch.Tensor(y_test))
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)

# Check if GPU is available and set device accordingly
device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")


# Define a neural network model class that includes an embedding layer, an LSTM layer, and a linear layer for classification
class MyNet_NoAttention(nn.Module):
    def __init__(self, vocab_size):
        super(MyNet_NoAttention, self).__init__()

        # Define some hyperparameters
        self.n_layers = n_layers = 2
        self.hidden_dim = hidden_dim = 512
        embedding_dim = 600
        drop_prob = 0.3

        # Define the embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Define the BiGRU layer
        self.gru = nn.GRU(embedding_dim,
                          hidden_dim,
                          n_layers,
                          dropout=drop_prob,
                          bidirectional=True,
                          batch_first=True)

        # Remove the multi-head attention layer

        # Define the fully connected layer
        self.fc = nn.Linear(in_features=hidden_dim * 2,
                            out_features=1)

        # Define the sigmoid activation function
        self.sigmoid = nn.Sigmoid()

        # Define the dropout layer
        self.dropout = nn.Dropout(drop_prob)

    # Forward pass of the neural network
    def forward(self, x, hidden):
        batch_size = x.shape[0]

        # Convert the input tensor to a long tensor
        x = x.long()

        # Pass the input tensor through the embedding layer
        embeds = self.embedding(x)

        # Pass the embedded tensor through the GRU layer
        hidden = torch.cat((hidden[0], hidden[1]), dim=0)
        gru_out, hidden = self.gru(embeds, hidden)

        # Apply dropout to the output tensor
        out = self.dropout(gru_out)

        # Pass the output tensor through the fully connected layer
        out = self.fc(out)

        # Apply the sigmoid activation function to the output tensor
        out = self.sigmoid(out)

        # Reshape the output tensor
        out = out.view(batch_size, -1)
        out = out[:, -1]
        return out, hidden

    # Initialize the hidden state of the GRU
    def init_hidden(self, batch_size):
        hidden = (torch.zeros(self.n_layers, batch_size, self.hidden_dim),
                  torch.zeros(self.n_layers, batch_size, self.hidden_dim))
        return hidden


# Create an instance of the neural network without attention
model_no_attention = MyNet_NoAttention(12)

# Move the model to the specified device (e.g., GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_no_attention.to(device)

# Define the loss function
criterion = nn.BCELoss()

# Define the learning rate
lr = 0.001

# Define the optimizer
optimizer = torch.optim.AdamW(model_no_attention.parameters(), lr=lr, weight_decay=1e-4)

# Define the number of epochs
epochs = 100
losses = []

# Training loop
for i in range(epochs):
    model_no_attention.to(device)
    for inputs, labels in train_loader:
        # Initialize the hidden state of the model
        h = model_no_attention.init_hidden(len(inputs))

        model_no_attention.train()
        inputs, labels = inputs.to(device), labels.to(device)
        h = tuple([e.data.to(device) for e in h])
        model_no_attention.zero_grad()
        output, h = model_no_attention(inputs, h)
        loss = criterion(output, labels.float())
        loss.backward()
        nn.utils.clip_grad_norm_(model_no_attention.parameters(), max_norm=5)
        optimizer.step()
    losses.append(loss.item())
    print("Epoch: {}/{}   ".format(i + 1, epochs), "Loss: {:.6f}   ".format(loss.item()))

# Save the trained model without attention
torch.save(model_no_attention.cpu(), 'model_no_attention.pth')

# Load the saved model without attention
new_model_no_attention = torch.load('model_no_attention.pth')

# Set the model to evaluation mode
new_model_no_attention.eval()

# Initialize the hidden state of the model
h = new_model_no_attention.init_hidden(len(x_test))

# Make predictions on the test data
output, h = new_model_no_attention(torch.Tensor(x_test).long(), h)

# Round the predictions to the nearest integer (0 or 1)
y_pred = torch.round(output).detach()

# Calculate metrics
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
sen = tp / (tp + fn)
spe = tn / (tn + fp)
acc = accuracy_score(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)
auroc = roc_auc_score(y_test, output.detach())

# Print the calculated metrics
print("Sensitivity: ", sen)
print("Specificity: ", spe)
print("Accuracy: ", acc)
print("MCC: ", mcc)
print("AUROC: ", auroc)
