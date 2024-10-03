import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from keras.preprocessing.text import Tokenizer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from ROC2 import roc
from PR import PR
from Recall import recall_curve
from sklearn.metrics import auc


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

# Read the training and testing data from CSV files·············
df_train = pd.read_csv('./dataset/mix-true-train_RNA.csv')
df_test = pd.read_csv('./dataset/mix-true-test_RNA.csv')


# Define a function to generate inputseq from a given sequence
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


# Generate inputseq from the training data and find the maximum length of these sequences
input_seqs_train = df_train['seq'].apply(lambda x: seq_fun(x, 1))
max_len = max(input_seqs_train.apply(len))

# Tokenize the inputseq using the Keras Tokenizer
tokenizer = Tokenizer(num_words=None)
tokenizer.fit_on_texts(input_seqs_train)
sequences_train = tokenizer.texts_to_sequences(input_seqs_train)
sequences_train = pad_seq(sequences_train, maxlen=max_len)

# Generate inputseq from the testing data and tokenize them using the same Tokenizer as above
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


# Define a neural network model class that includes an embedding layer, an GRU layer, an attention layer,
# and a linear layer for classification
class MyNet(nn.Module):

    # Initialize the neural network
    def __init__(self, vocab_size):
        super(MyNet, self).__init__()

        # Define some hyperparameters
        self.n_layers = n_layers = 2  # Number of GRU layers
        self.hidden_dim = hidden_dim = 512  # Hidden layer dimension
        embedding_dim = 600  # Embedding dimension
        drop_prob = 0.3

        # Define the embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Define the GRU layer
        self.gru = nn.GRU(embedding_dim,
                          hidden_dim,
                          n_layers,
                          dropout=drop_prob,
                          bidirectional=False,  # Change to single directional GRU
                          batch_first=True
                          )

        # Define the fully connected layer
        self.fc = nn.Linear(in_features=hidden_dim,  # Change to single directional GRU output dimension
                            out_features=1
                            )

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
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return hidden.to(device)


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

# Define the number of epochs for training
for i in range(epochs):
    # Move the model to the specified device (e.g. GPU)
    model.to(device)

    # Iterate through the training data loader, which returns batches of inputs and labels
    for inputs, labels in train_loader:
        # Initialize the hidden state of the model
        h = model.init_hidden(len(inputs))

        # Set the model to training mode
        model.train()

        # Move the hidden state to the specified device (e.g. GPU)
        h = h.to(device)

        # Move the inputs and labels to the specified device (e.g. GPU)
        inputs, labels = inputs.to(device), labels.to(device)

        # Clear the gradients of all optimized variables
        model.zero_grad()

        # Forward pass: compute the output of the model given the inputs and hidden state
        output, h = model(inputs, h)

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

    # Iterate through the test data loader (not used in this code)
    for inputs, labels in test_loader:
        pass

# Save the trained model to a file
torch.save(model.cpu(), 'model.pth')

# Import necessary libraries for calculating metrics
from sklearn.metrics import confusion_matrix, accuracy_score, matthews_corrcoef, roc_auc_score

# Load the saved model
new_model = torch.load('model.pth')

# Set the model to evaluation mode
new_model.eval()
new_model.to(device)  # Ensure the model is on the correct device

# Initialize the hidden state of the model
h = new_model.init_hidden(len(x_test))

# Make predictions on the test data
x_test_tensor = torch.Tensor(x_test).long().to(device)  # Move input data to device
output, h = new_model(x_test_tensor, h)

# Round the predictions to the nearest integer (0 or 1)
y_pred = torch.round(output).detach().cpu()  # Move predictions to CPU for evaluation

# Calculate the true negatives (tn), false positives (fp), false negatives (fn), and true positives (tp) from the confusion matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

# Calculate the sensitivity, specificity, accuracy, and Matthews correlation coefficient (MCC)
sen = tp / (tp + fn)
spe = tn / (tn + fp)
acc = accuracy_score(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)

# Calculate the area under the receiver operating characteristic (ROC) curve (AUROC)
y_score = output.detach().cpu()  # Move score to CPU for evaluation
auroc = roc_auc_score(y_test, y_score)

# Print the calculated metrics
print("Sensitivity: ", sen)
print("Specificity: ", spe)
print("Accuracy: ", acc)
print("MCC: ", mcc)
print("AUROC: ", auroc)
