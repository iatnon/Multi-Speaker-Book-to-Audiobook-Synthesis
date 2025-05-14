import itertools
import math
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
from typing import List, Dict
import torch.nn.functional as F
import json

class SESDataset(Dataset):
    """
    The SES dataset class for the Speaker Embedding Similarity model.
    """    
    def __init__(self, celebrity_descriptions, celebrity_speaker_embeddings,get_word_embedding):
        """_summary_

        Args:
            celebrity_descriptions (Dict): Contains celebrities with their corresponding descriptions (sex, age and 5 words describing their voice).
            celebrity_speaker_embeddings (Dict): Contains celebrities with their corresponding speaker embeddings.
            get_word_embedding (Function): A function to get a GloVe word embedding for a given word.
        """        

        # Set up the class variables
        self.celebrity_descriptions = celebrity_descriptions
        self.names = list(celebrity_descriptions.keys())

        # the GLoVe model has 50 dimensions
        self.glove_dim = 50
        self.pairs = []
        self.targets = []

        # Precompute average description embeddings
        self.avg_desc_emb = {}

        # Loop through each celebrity and compute the mean of their word embeddings
        for name in self.names:
            words = celebrity_descriptions[name][2:] 
            if not words:
                self.avg_desc_emb[name] = torch.zeros(self.glove_dim, dtype=torch.float32)
            else:
                embeddings = [torch.from_numpy(get_word_embedding(word)).to(torch.float32) for word in words if get_word_embedding(word) is not None]
                if embeddings:
                    self.avg_desc_emb[name] = torch.mean(torch.stack(embeddings), dim=0)
                else:
                    self.avg_desc_emb[name] = torch.zeros(self.glove_dim, dtype=torch.float32)

        # Create pairs of celebrity embeddings and compute target similarity
        for name1, name2 in itertools.combinations(self.names, 2):

            # Ensure the embeddings are Float32 and squeeze out 1 dimensions
            emb1 = celebrity_speaker_embeddings[name1].to(torch.float32).squeeze()
            emb2 = celebrity_speaker_embeddings[name2].to(torch.float32).squeeze()
            
            # Get the age and sex of the celebrities
            sex1, age1 = celebrity_descriptions[name1][:2]
            sex2, age2 = celebrity_descriptions[name2][:2]

            # Check if the sex is the same, if not the target similarity is 0
            if sex1 == sex2:

                # Get the difference in age
                age_diff = abs(age1 - age2)

                # Calculate the age factor using an exponential decay function
                age_factor = math.exp(-age_diff / 10)
                desc_emb1 = self.avg_desc_emb[name1]
                desc_emb2 = self.avg_desc_emb[name2]

                # Handle zero-norm cases
                if desc_emb1.norm() == 0 or desc_emb2.norm() == 0:
                    desc_sim = 0.0
                else:
                    # Calculate cosine similarity between the average description embeddings
                    desc_sim = F.cosine_similarity(desc_emb1, desc_emb2, dim=0).item()

                    # Ensure non-negative
                    desc_sim = max(0, desc_sim)  
                target_similarity = (age_factor + desc_sim) / 2

                # Ensure target similarity is between 0 and 1
                if target_similarity > 1:
                    target_similarity = 1.0
            else:
                target_similarity = 0.0
            
            # Add the pair and their target to the dataset
            self.pairs.append((emb1, emb2))
            self.targets.append(target_similarity)
            
            # Add random pair to ensure the model doesn't overfit to the celebrity pairs
            random_tensor = torch.rand(128) * 20 - 10
            self.pairs.append((random_tensor, emb2))
            self.targets.append(0)

    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        """Get an item from the dataset

        Returns:
            Zonos Speaker Embedding: The first speaker embedding of the pair.
            Zonos Speaker Embedding: The second speaker embedding of the pair.
            float: The target similarity between the two speaker embeddings.
        """        
        emb1, emb2 = self.pairs[idx]

        # Ensure the embeddings are Float32
        emb1 = emb1.to(torch.float32)
        emb2 = emb2.to(torch.float32)
        target = self.targets[idx]
        return emb1, emb2, torch.tensor([target], dtype=torch.float32)

class DtSEDataset(Dataset):
    """
    The dataset for the DtSE model.
    """
    def __init__(self, data, get_word_embedding, celebrity_embeddings):
        """Initialize the dataset with the data, word embedding function and celebrity embeddings.

        Args:
            data (list): A list of synthesized data containing a fictional character description (sex, age and 5 words describing their voice) and a list of 5 celebrities that the large language model thinks sounds similar to the fictional character.
            get_word_embedding (Function): Function to get a GloVe word embedding for a given word
            celebrity_embeddings (Dict): A dictionary of celebrity names and their corresponding speaker embeddings.
        """        

        # Set up the class variables
        self.data = data
        self.get_word_embedding = get_word_embedding
        self.celebrity_embeddings = celebrity_embeddings  # Dict: {name: tensor(128)}
        self.max_adjectives = 5
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset

        Args:
            idx (int): index

        Returns:
            input_tensor: The input for the model
            targets: The target celebrities for the model
        """        

        # Get an item and normalize the age and binarize the sex
        sample = self.data[idx]
        age = sample['age'] / 100.0
        male_sex = 1.0 if sample['sex'].lower() == 'male' else 0.0

        # Get the adjectives and their word embeddings
        adjectives = sample['adjectives'][:self.max_adjectives]
        adj_embeddings = [self.get_word_embedding(word) for word in adjectives if self.get_word_embedding(word) is not None]
       
        # Pad with zero's if neccesary
        if not adj_embeddings:
            adj_embeddings = [np.zeros(50)]
        num_adjs = len(adj_embeddings)

        if num_adjs < self.max_adjectives:
            adj_embeddings.extend([np.zeros(50)] * (self.max_adjectives - num_adjs))
        adj_vector = np.concatenate(adj_embeddings[:self.max_adjectives])

        # Concatenate input features and convert them to a tensor
        input_features = np.concatenate(([age], [male_sex], adj_vector))
        input_tensor = torch.tensor(input_features, dtype=torch.float32)

        # Get the target celebrities
        top_celebrities = sample['top_celebrities']
        target_embeddings = []
        for x, celeb in enumerate(top_celebrities):
            if celeb not in self.celebrity_embeddings:
                target_embeddings.append(torch.zeros(128))
            else:
                target_embeddings.append(self.celebrity_embeddings[celeb])
        targets = torch.stack(target_embeddings)
        
        # Return the input and targets
        return input_tensor, targets

def load_DtSE_data_from_csv(csv_data_path: str) -> List[Dict]:
    """Loads the data from a csv for DtSE data 

    Args:
        csv_data_path (str): The path to the data for the DtSE model.

    Returns:
        List[Dict]: The data for the DtSE dataset contains sex, age, 5 adjectives and 5 names of similar sounding celebirites
    """    

    # Read the csv
    df = pd.read_csv(csv_data_path)

    # Count the males and females
    male_count = (df["sex"] == "male").sum()
    female_count = (df["sex"] == "female").sum()

    # Find the smaller group size 
    min_count = min(male_count, female_count)
    data = []

    male_count, female_count = 0, 0

    # Loop thought the data and format it
    for _, row in df.iterrows():            
        adjectives = [row["adj1"], row["adj2"], row["adj3"], row["adj4"], row["adj5"]]
        sample = {
                    "age": row["age"],
                    "sex": row["sex"],
                    "adjectives": adjectives,
                    "top_celebrities": [
                        row["celeb1_name"],
                        row["celeb2_name"],
                        row["celeb3_name"],
                        row["celeb4_name"],
                        row["celeb5_name"]
                    ]
                }
        
        # Check if the males are not overrepresented at this point, if not add the sample to the data
        if row["sex"] == "male" and male_count < min_count:
            data.append(sample)
            male_count += 1

        # Check if the females are not overrepresented at this point, if not add the sample to the data
        elif row["sex"] == "female" and female_count < min_count:
            data.append(sample)
            female_count += 1
    
    return data

class DataProcessor():
    """
    This class is used to process the data for both the SES and DtSE models
    """    

    def __init__(self, components, SE_path, DTSC_train_path, DTSC_test_path , CD_train_path, CD_test_path):
        """Initialize the data proccesor by providing the paths the available data is located

        Args:
            components (Object): Component (The get word embedding function)
            SE_path (str): The path to the Speaker Embeddings (SE)
            DTSC_train_path (str): The path to the Description To Similar Celebrities (DTSC) training data
            DTSC_test_path (str): The path to the Description To Similar Celebrities (DTSC) testing data
            CD_train_path (str): The path to the Celebrity Descriptions (CD) training data
            CD_test_path (str): The path to the Celebrity Descriptionss (CD) testing data
        """             

        # Set the class variables and load the data
        self.components = components
        self.training_data = load_DtSE_data_from_csv(DTSC_train_path)
        self.test_data = load_DtSE_data_from_csv(DTSC_test_path)
        speaker_embeddings = torch.load(SE_path, map_location="cpu")
        self.celebrity_speaker_embeddings = {name: torch.squeeze(emb, dim=(0, 1)) for name, emb in speaker_embeddings.items()}
        with open(CD_train_path) as json_file:
            self.training_celebrity_descriptions = json.load(json_file)
        with open(CD_test_path) as json_file:
            self.test_celebrity_descriptions = json.load(json_file)

    def get_SES_loaders(self, batch_size=32896, test_size=0.5):
        """Gets the loaders for the SES data

        Args:
            batch_size (int, optional): The batch size. Defaults to 32896.
            test_size (float, optional): The test size (Should probably be 0.5 to evenly divide the number of celebrities). Defaults to 0.5.

        Returns:
            DataLoader: The dataloader for the SES model which is ready to be used for training
        """        

        # Create SES datasets
        train_dataset = SESDataset(self.training_celebrity_descriptions, self.celebrity_speaker_embeddings, self.components.get_word_embedding)
        val_test_dataset = SESDataset(self.test_celebrity_descriptions, self.celebrity_speaker_embeddings, self.components.get_word_embedding)
        return self.get_loaders(train_dataset, val_test_dataset, batch_size, test_size)

    def get_DtSE_loaders(self, batch_size=32, test_size=0.5):
        """Gets the loaders for the DtSE data

        Args:
            batch_size (int, optional): The batch size. Defaults to 32896.
            test_size (float, optional): The test size (Should probably be 0.5 to evenly divide the number of celebrities). Defaults to 0.5.

        Returns:
            DataLoader: The dataloader for the DtSE model which is ready to be used for training
        """      

        # Create DtSE datasets
        train_dataset = DtSEDataset(self.training_data,self.components.get_word_embedding,self.celebrity_speaker_embeddings)
        val_test_dataset = DtSEDataset(self.test_data,self.components.get_word_embedding,self.celebrity_speaker_embeddings)
        return self.get_loaders(train_dataset, val_test_dataset, batch_size, test_size)

    def get_loaders(self, train_dataset, val_test_dataset, batch_size, test_size):
        """Gets data loaders for a training and validation/test dataset

        Args:
            train_dataset (DataSet): The training dataset
            val_test_dataset (DataSet): The validation/testing dataset
            batch_size (int): The batch size
            test_size (float): The test size

        Returns:
            DataLoader: Dataloaders for training, testing and validation 
        """        

        # Divide into validation and test sets
        test_size = int((1 - test_size) * len(val_test_dataset))
        val_size = len(val_test_dataset) - test_size
        test_dataset, val_dataset = random_split(val_test_dataset, [test_size, val_size])

        # Create loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        return train_loader, val_loader, test_loader