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


# Define a neural network model class that includes an embedding layer and a linear layer for classification
class MyNet(nn.Module):
    # Initialize the neural network
    def __init__(self, vocab_size):
        super(MyNet, self).__init__()

        # Define some hyperparameters
        embedding_dim = 600
        drop_prob = 0.3

        # Define the embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Define the multi-head attention layer
        self.attention = nn.MultiheadAttention(embed_dim=embedding_dim,
                                               num_heads=1,
                                               batch_first=True,
                                               dropout=drop_prob)

        # Define the fully connected layer
        self.fc = nn.Linear(in_features=embedding_dim,
                            out_features=1)

        # Define the sigmoid activation function
        self.sigmoid = nn.Sigmoid()

        # Define the dropout layer
        self.dropout = nn.Dropout(drop_prob)

    # Forward pass of the neural network
    def forward(self, x):
        batch_size = x.shape[0]

        # Convert the input tensor to a long tensor
        x = x.long()

        # Pass the input tensor through the embedding layer
        embeds = self.embedding(x)

        # Pass the embedded tensor through the multi-head attention layer
        attn_out, _ = self.attention(embeds, embeds, embeds)

        # Apply dropout to the output tensor
        out = self.dropout(attn_out)

        # Pass the output tensor through the fully connected layer
        out = self.fc(out)

        # Apply the sigmoid activation function to the output tensor
        out = out.view(batch_size, -1)
        out = out[:, -1]
        out = self.sigmoid(out)

        return out


# Create an instance of the neural network
model = MyNet(12)

# Move the model to the specified device (e.g., GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the loss function
criterion = nn.BCELoss()

# Define the learning rate
lr = 0.001

# Define the optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

# Define the number of epochs
epochs = 100
losses = []

# Training loop
for i in range(epochs):
    model.to(device)
    for inputs, labels in train_loader:
        # Set the model to training mode
        model.train()

        # Move the inputs and labels to the specified device (e.g., GPU)
        inputs, labels = inputs.to(device), labels.to(device)

        # Clear the gradients of all optimized variables
        model.zero_grad()

        # Forward pass: compute the output of the model given the inputs
        output = model(inputs)

        # Compute the loss between the predicted output and the true labels
        loss = criterion(output, labels.float())

        # Backward pass: compute the gradients of all optimized variables with respect to the loss
        loss.backward()

        # Clip the gradients to prevent exploding gradients
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)

        # Update the optimized variables based on the computed gradients
        optimizer.step()

    # Print the loss for the current epoch
    losses.append(loss.item())
    print("Epoch: {}/{}   ".format(i + 1, epochs), "Loss: {:.6f}   ".format(loss.item()))

# Save the trained model to a file
torch.save(model.cpu(), 'model.pth')

# Load the saved model
new_model = torch.load('model.pth')

# Set the model to evaluation mode
new_model.eval()

# Make predictions on the test data
output = new_model(torch.Tensor(x_test).long())

# Round the predictions to the nearest integer (0 or 1)
y_pred = torch.round(output).detach()

# Calculate the true negatives (tn), false positives (fp), false negatives (fn), and true positives (tp) from the confusion matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

# Calculate the sensitivity, specificity, accuracy, and Matthews correlation coefficient (MCC)
sen = tp / (tp + fn)
spe = tn / (tn + fp)
acc = accuracy_score(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)

# Calculate the area under the receiver operating characteristic (ROC) curve (AUROC)
y_score = output.detach()  # The positive probability predicted by the model
auroc = roc_auc_score(y_test, y_score)

# Print the calculated metrics
print("Sensitivity: ", sen)
print("Specificity: ", spe)
print("Accuracy: ", acc)
print("MCC: ", mcc)
print("AUROC: ", auroc)
