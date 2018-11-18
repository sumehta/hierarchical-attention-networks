An Efficient implementation of Hierarchical Attention Networks model proposed in the paper https://www.cs.cmu.edu/~hovy/papers/16HLT-hierarchical-attention-networks.pdf

Includes all the supporting files:
data.py - Reads data from pandas dataframe containing fields, 'text' and 'label' respectively. Expects the data to be split into train, test and val each into separate csv files.
util.py - Contains batcher methods, embedding methods etc.
train.py - Main training loop
main.py - Specify data paths, embedding files and other settings here.
model.py - Contains model broken down into different parts.
