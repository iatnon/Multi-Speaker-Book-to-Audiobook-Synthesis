import torch
import torch.nn as nn

# Speaker embedding similarity model
class SESModel(nn.Module):
    """
    The Speaking Embedding Similarity (SES) model takes two embeddings and calculates their similarity based on how
    Similar they should sound
    """    
    def __init__(self, embedding_dim=128, hidden_dim=256):
        super(SESModel, self).__init__()
        self.fc1 = nn.Linear(embedding_dim * 2, hidden_dim)  # Expect concatenated embeddings.
        self.fc4 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.3)


    def forward(self, emb1, emb2):
        # Ensure inputs are float32
        emb1 = emb1.to(torch.float32)
        emb2 = emb2.to(torch.float32)
        
        # Concatenate along feature dimension
        x = torch.cat([emb1, emb2], dim=1)
        
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc4(x)
        x = self.sigmoid(x)
        return x

# Description to speaker embedding model
class DtSEModel(nn.Module):
    """
    The Description To Speaker Embedding (DtSE) model takes a character description containingn an sex, age and 5 words
    converted to GloVe word embeddings and generates a speaker embedding that should sound similar to the character description 
    """    
    def __init__(self, max_adjectives=5, embedding_dim=50, hidden_size=256, output_size=128, dropout = 0.3):
        super(DtSEModel, self).__init__()
        input_size = 2 + max_adjectives * embedding_dim
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc4(x))
        x = self.dropout(x)
        x = self.fc5(x)
        x = torch.clip(x, -10, 10)
        return x