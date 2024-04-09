import torch
import torch.nn as nn
import numpy as np
import pandas as pd 
from Bio import SeqIO
import json
import argparse
import sys

# Construct a model class for neural networks that encompasses an embedding layer and a GRU layer 

class MyNet(nn.Module):
    
    # Commence the initialization process of the neural network
    def __init__(self, vocab_size):
        super(MyNet, self).__init__()
        # Specify a few parameters that are highly influential in the performance of the paper deduplication system
        self.n_layers = n_layers = 2 
        self.hidden_dim = hidden_dim = 512
        embedding_dim = 400
        drop_prob=0.5
        
        # Define the embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim) 
        
        # Define the GRU layer
        self.GRU = nn.GRU(embedding_dim,
                            hidden_dim, 
                            n_layers, 
                            dropout=drop_prob, 
                            bidirectional=True,
                            batch_first = True 
                           )
        
        # Define the multi-head attention layer
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim,
                                                num_heads=1,
                                                batch_first=True,
                                                dropout=drop_prob
                                               )
        
        # Define the fully connected layer
        self.fc = nn.Linear(in_features=hidden_dim, 
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
        
        hidden = torch.cat((hidden[0],hidden[1]),dim=0)
        # Pass the GRU output tensor through the multi-head attention layer
        gru_out, hidden = self.gru(embeds, hidden)
        gru_out, _ =self.attention(gru_out,gru_out,gru_out)
                
        # Apply dropout to the output tensor
        out = self.dropout(gru_out)
        
        # Pass the output tensor through the fully connected layer
        out = self.fc(out)
        # Apply the sigmoid activation function to the output tensor
        out = self.sigmoid(out)
        # Reshape the output tensor
        out = out.view(batch_size, -1)       
        out = out[:,-1]        
        return out, hidden     
    
    # Initialize the hidden state of the GRU
    def init_hidden(self, batch_size):
        hidden = (torch.zeros(self.n_layers, batch_size, self.hidden_dim),
                  torch.zeros(self.n_layers, batch_size, self.hidden_dim)
                 )
        return hidden

def rna_to_token(dna_seq, token_len=1):
    base_map = {'A': 3, 'C': 2, 'G': 1, 'U': 4}
    num_seq = [base_map[base] for base in dna_seq]
    
    tokens = []
    for i in range(0, len(num_seq), token_len):
        token = num_seq[i:i+token_len]
        token_str = ''.join([str(num) for num in token])
        tokens.append(int(token_str))

    return tokens

def Csv_to_json(result_csv,result_json):

    csvData = pd.read_csv(result_csv, header = 0)  
    columns = csvData.columns.tolist()
    dimensionoutPut = {}
    for index in range(len(csvData)):
        outPut = {}
        for col in columns:
            outPut[col] = str(csvData.loc[index, col])
        dimensionoutPut[str(index+1)] = outPut
    jsonData = json.dumps(dimensionoutPut,indent=4) 
    with open(args.result_json, 'w') as jsonFile:   
        jsonFile.write(jsonData)


def main():
    seq_filename = args.input_fa   # Read the file directly

    # fasta sequences were read from the input file and screened for sequences of length 201 with a C at the 101st position.
    detected_seq = {"name": [], "seq": []}
    for fa in SeqIO.parse(seq_filename, "fasta"):
        seq = fa.seq.upper()
        if len(seq) == 201 and seq[100] == "C":
            detected_seq["name"].append(fa.name)
            detected_seq["seq"].append(rna_to_token(seq))

    if len(detected_seq["seq"]) > 0:

        new_model = torch.load(args.module)
        #new_model = MyNet(6)
        input_seqs = detected_seq["seq"]
        h = new_model.init_hidden(len(input_seqs))
        new_model.eval()
        output, h = new_model(torch.Tensor(input_seqs).long(),h)


        predict = output
        ac4Cs = np.array(torch.round(output).detach())
        ac4C = ["Yes" if x == 1 else "No" for x in ac4Cs]
        results = dict()
        results["Name"] = np.array(detected_seq["name"])
        results["Position"] = [101] * len(results["Name"])
        results["Base"] = ['C'] * len(results["Name"])
        results["ac4C Probability"] = np.array([format(x,".2%") for x in predict])
        results["ac4C"] = ac4C

        # creates a dataframe from the results and writes it to a csv file named results.csv in the result_path directory.
        df = pd.DataFrame(results)
        df.to_csv(args.result_csv,index=False,encoding="utf_8_sig")

        # calls the Csv_to_json function to convert the csv results to json format and write it to the result_path directory.
        Csv_to_json(args.result_csv,args.result_json)

    # If no sequences are filtered, it writes an empty dataframe to the csv and json files.    
    else:
        results = dict()
        results["Name"] =  []
        results["Position"] =  []
        results["Residue"] =  []
        results["ac4C Probability"] =  []
        results["ac4C"] =  []
        df = pd.DataFrame(results)
        df.to_csv(args.result_csv,index=False,encoding="utf_8_sig")
        Csv_to_json(args.result_csv,args.result_json)

if __name__ == "__main__":

    # sets up the command line argument parser and assigns default values to the input_path and result_path.
    parser = argparse.ArgumentParser(description="ac4C")
    parser.add_argument('--input_fa',type=str,default='./seq.fa',help='input.fa')
    parser.add_argument('--result_csv',type=str,default='./result.csv',help='result.csv file') 
    parser.add_argument('--result_json',type=str,default='./result.json',help='result.json file')
    parser.add_argument('--module',type=str,default='./mix_module.pth',help='module.pth file')
    args = parser.parse_args()    

    # starts the main function when the script is run.
    main()
