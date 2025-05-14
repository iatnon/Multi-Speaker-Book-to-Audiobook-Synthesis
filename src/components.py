import torch
import gensim.downloader as api
import pickle
import os
import json
import re
import requests
import numpy as np

def extract_json_from_markdown(response: str) -> str:
    """Extracts JSON content from a markdown llm response.

    Args:
        response (str): LLM response containing JSON in markdown format.

    Returns:
        str: A clean string of JSON content.
    """    

    # Look for ```json ... ```
    response = str(response)
    match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
    if match:
        return match.group(1)
    else:
        # If no markdown block is found, assume the entire response is JSON
        return response.strip()
    
def parse_llm_response(llm_response: str, data_type):
    """Parses the LLM response to extract JSON data.

    Args:
        llm_response (str): The LLM response containing JSON data.
        data_type (type): The type of data expected (e.g., dict, list).

    Returns:
        Data: The parsed data in the specified format.
    """    

    # Extract JSON content
    json_content = extract_json_from_markdown(llm_response)
    
    # Attempt to parse the JSON
    try:
        parsed_data = json.loads(json_content)
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON: {e}")
        print(f"Invalid JSON content: {json_content}")
        return None
    
    # Ensure the parsed data is a list (assuming that's the expected structure)
    if not isinstance(parsed_data, data_type):
        print(f"Expected a dict, got: {type(parsed_data)}")
        return None
    
    return parsed_data

def llm_call(prompt, data_type=dict):
    """Call to a Large Language Model (LLM) to generate a response for the given prompt. 

    Args:
        prompt (str): The prompt to the LLM.
        data_type (type, optional): _description_. Defaults to dict, the expected type of data you expect.

    Returns:
        str: The parsed response from the LLM in the specified data structure.
    """    

 
    response = "llm_response"  # Placeholder for actual LLM call
    
    parsed_response = parse_llm_response(response, data_type)
    if parsed_response is None:
        print("Failed to parse LLM response.")
        return None
    return parsed_response
    
def view_SEED(SEED_path):
    """Gets a Speaker Embedded Expressive Dialogue (SEED) file and prints the contents in a readable format.

    Args:
        SEED_path (str): The path to the SEED file
    """    

    # Load the SEED file
    with open(SEED_path, 'rb') as f:
        seed = pickle.load(f)

    # Print the contents of the SEED file
    for i in seed:
        print('---------------------')
        print(f'Character: {i[0]}, {i[5]}, speaking rate: {i[6]}')
        print(f'Dialogue text: {i[1]}')
        print(f'Happiness: {round(i[2][0]*100)}%, Sadness: {round(i[2][1]*100)}%, Disgust: {round(i[2][2]*100)}%, Fear: {round(i[2][3]*100)}%, Surprise: {round(i[2][4]*100)}%, Anger: {round(i[2][5]*100)}%, Other: {round(i[2][6]*100)}%, Neutral: {round(i[2][7]*100)}%')
        print(f'Context: {i[4]}')

def generate_k_lambda_list(steepness=1.0, n=5):
    """Generates a list of `n` numbers from high to low with a total sum of 1.
    The `steepness` parameter controls how steep the drop is (higher = steeper).

    Args:
        steepness (float, optional): _description_. Defaults to 1.0, The steepness in which the k_lambda list decays.
        n (int, optional): _description_. Defaults to 5, The lenght of the k_lambdas list (The number of speaker embedding targets).

    Returns:
        list: A list of `n` float numbers that sum to 1, the list decays (First number weighted highest).
    """

    # Generate exponentially decaying values
    raw = np.exp(-steepness * np.arange(n))

    # Normalize to sum to 1
    normalized = raw / np.sum(raw)
    
    return list(normalized)


class Components():
    """
    The Components class is used to load the GloVe model and get word embeddings.
    """    
    def __init__(self, glove_model_path = "glove_model.pkl"):
        """Takes a path to a GloVe model and loads it. If the model is not found, it will download the model from the internet.

        Args:
            glove_model_path (str): _description_. Defaults to "glove_model.pkl", The path where the glove model is located or will be located.
        """ 

        # Check if the GloVe model exists locally
        if os.path.exists(glove_model_path):

            # Load the model from local file
            with open(glove_model_path, "rb") as f:
                self.glove_model = pickle.load(f)
            print("Loaded GloVe model from local file.")
        else:
            # Download the model and save it locally
            print("Downloading GloVe model...")
            self.glove_model = api.load("glove-wiki-gigaword-50")
            
            with open(glove_model_path, "wb") as f:
                pickle.dump(self.glove_model, f)
            print("Downloaded and saved GloVe model.")   
             
    def get_word_embedding(self, word):
        """
        Returns the embedding for a given word using the GloVe model.
        
        Parameters:
        word (str): The word to get the embedding for.
        
        Returns:
        numpy.array or None: The embedding vector if the word is in the model's vocabulary, otherwise None.
        """
        word.lower()
        if word in self.glove_model:
            return self.glove_model[word]
        else:
            return None