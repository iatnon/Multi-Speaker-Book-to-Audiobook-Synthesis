import torch
import pandas as pd
from typing import List
from components import llm_call

class SyntheticDataGenerator:
    """
    The SyntheticDataGenerator generates fictional characters using a LLM. All characters have a character description
    and 5 celebrities that should should similar to the character.
    """    
    def __init__(self, n_similar_celebrities=5):
        """Initialize the SyntheticDataGenerator

        Args:
            n_similar_celebrities (int, optional): The number of similar sounding celebrities to be synthesized. Defaults to 5.
        """        
        self.n_similar_celebrities = n_similar_celebrities
        
    # Simulated LLM response (replace with actual LLM call)
    def synthezise_DTSC_data(self, celebrities_embeddings_path, output_path, n) -> str:
        """Generates the synthesized data neccesary for the training of the DtSE model

        Args:
            celebrities_embeddings_path (str): The path where the celebrity speaker embeddings are stored
            output_path (str): The path where the synthetic data is to be saved
            n (int): The total number of data samples require 

        
        """        

        # Get the list of celebrities to be inserted into the prompt (Between which similar sounding celebrities the LLM can choose)
        celebrity_speaker_embeddings = torch.load(celebrities_embeddings_path, map_location="cpu")  # Load on CPU
        celebrities = list(celebrity_speaker_embeddings.keys())
        
        # Construct the prompt to the LLM specifying the required data
        prompt = f"""
        Generate {self.n_similar_celebrities} fictional characters including the following:

        For each character:
        1. Randomly assign an age between 25 and 80 (integer).
        2. Randomly assign a sex (male or female).
        3. Provide a 5-word description of the character's voice using extremely varied adjectives (for example arrogant) for complex characters. Do not repeat the same adjectives too often, as this prompt will be asked and varied characters are necessary.
        4. List the top 5 celebrities from the provided list underneath whose voices you estimate to most closely match the character's voice. Ensure all celebrities you use are in this list:
        - List of celebrities: {celebrities}
        Return the response in JSON format as a list of {self.n_similar_celebrities} objects, each with:
        {{
        "age": integer,
        "sex": "male or female",
        "voice_description": ["adjective1", "adjective2", "adjective3", "adjective4", "adjective5"],
        "top_celebrities": ["celebrity1", "celebrity2","celebrity3","celebrity4","celebrity5",]
        ]
        }}
        """

        # Call the LLM and save the response data for the amount of data samples required
        for i in range(int(n/self.n_similar_celebrities)):
            parsed_response = llm_call(prompt, data_type=list)
            self.save_llm_data_to_csv(parsed_response, output_path, celebrities, append=True)

    # Function to parse LLM response and save to CSV
    def save_llm_data_to_csv(self, parsed_response: str, output_csv: str, celebrity_list: List[str], append: bool = True):
        """Saves the parsed LLM response to the specified CSV

        Args:
            parsed_response (str): The parsed response from the LLM
            output_csv (str): The csv path to which the response should be added
            celebrity_list (List[str]): The list of celebrities
            append (bool, optional): Controls when to append to the CSV or create it. Defaults to True.
        """        
        
        data_rows = []
        for sample in parsed_response:
            # Validate structure
            if (not all(key in sample for key in ["age", "sex", "voice_description", "top_celebrities"]) or
                len(sample["voice_description"]) != 5 or 
                len(sample["top_celebrities"]) != 5 
                ):
                print(f"Invalid sample: {sample}")
                continue

            # Count how many celebrities are not in the celebrity list, if this is higher than 2 he sample is discarded
            count = 0
            for c in sample["top_celebrities"]:
                if c not in celebrity_list:
                    count += 1
                    print(f"Invalid celebrity name: {c}")
            if count >= 2:
                print('rejected')
                continue

            # Flatten into a row
            row = {
                "age": sample["age"],
                "sex": sample["sex"],
                "adj1": sample["voice_description"][0],
                "adj2": sample["voice_description"][1],
                "adj3": sample["voice_description"][2],
                "adj4": sample["voice_description"][3],
                "adj5": sample["voice_description"][4],
                "celeb1_name": sample["top_celebrities"][0],
                "celeb2_name": sample["top_celebrities"][1],
                "celeb3_name": sample["top_celebrities"][2],
                "celeb4_name": sample["top_celebrities"][3],
                "celeb5_name": sample["top_celebrities"][4],
            }

            # Add to the data
            data_rows.append(row)
        
        # Save to CSV
        if data_rows:
            df = pd.DataFrame(data_rows)
            mode = 'a' if append and pd.io.common.file_exists(output_csv) else 'w'
            header = not (append and pd.io.common.file_exists(output_csv))
            df.to_csv(output_csv, mode=mode, header=header, index=False)
            print(f"Saved {len(data_rows)} records to {output_csv}")
        else:
            print("No valid samples to save")
        