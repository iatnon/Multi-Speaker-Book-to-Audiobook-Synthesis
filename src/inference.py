import numpy as np
import torch
from data_proccesing import *
from components import *
from train import *
import difflib

def get_closest_match(query, data_dict, characters):
    """Get the closest matching character to the character in the dialogue

    Args:
        query (str): The character in the dialogue
        character_embeddings (Dict): The characters (str) and their corresponding speaker embedding
        characters (List): A list of all characters in a book

    Returns:
        torch.Tensor: 128-dimensional speaker embedding for the character
    """    

    # Get a list of all characters (str)
    keys = list(data_dict.keys())
    closest_match = difflib.get_close_matches(query, keys, n=1, cutoff=0.1)  # Adjust cutoff as needed
    if closest_match:
        print(closest_match[0])
        for char in characters:
            if closest_match[0] == char['name']:
              
                speaker_embedding = data_dict[closest_match[0]].unsqueeze(0).unsqueeze(0)
                character_description =  str(char['age']) + " "  + char['sex'] + " " + char['voice'][0] + " " + char['voice'][1] + " " + char['voice'][2] + " " + char['voice'][3] + " " + char['voice'][4]
                return speaker_embedding, character_description
        return data_dict[closest_match[0]]
    return None  # No close match found

# Inference function
def get_speaker_embedding(age: int, sex: str, adjectives: list[str], get_word_embedding, model) -> torch.Tensor:
    """
    Generate a speaker embedding from age, sex, and five adjectives.
    
    Args:
        age (int): Age of the speaker
        sex (str): 'male' or 'female'
        adjectives (list[str]): Exactly five adjectives describing the voice
        glove_model: Loaded GloVe model
        model: The DtSE model
    
    Returns:
        torch.Tensor: 128-dimensional speaker embedding
    """

    # Validate input
    if len(adjectives) != 5:
        raise ValueError("Exactly 5 adjectives are required.")
    
    # Normalize age (assuming max age of 100 for simplicity)
    age_normalized = (age / 100.0)
    
    # Convert sex to float
    male_sex = 1.0 if sex.lower() == 'male' else 0.0

    # Convert adjectives to GloVe embeddings
    adj_embeddings = [get_word_embedding(word) for word in adjectives if get_word_embedding(word) is not None]
    while len(adj_embeddings) < 5:
        adj_embeddings.append(np.zeros(50))

    # Concatenate all inputs into a single array
    input_features = np.concatenate(([age_normalized], [male_sex], *adj_embeddings))

    input_tensor = torch.tensor(input_features, dtype=torch.float32)
   
    # Perform on the DtSE model using the input
    with torch.no_grad():
        speaker_embedding = model(input_tensor)
    return speaker_embedding

# Book To Speaker Embedded Emotional Dialogue
def BTSEED(components, model_path, book_path,SEED_path):
    """Book To Speaker Embedded Expressive Dialogue (DtSEED). This function takes a book and analyses it using a LLM
    The LLM will respond with a list of character descriptions that are interpretable for the DtSE model to make
    speaker embeddings from. Secondly the LLM provides a quote from dialogue for each character together with it's
    emotional tone, speaking rate and context.

    Args:
        components (Object): The get word embedding component
        model_path (str): The path to the DtSE model
        book_path (str): The path to the book (txt)
        SEED_path (str): The path where the Speaker Embedded Expressive Dialogue (SEED) will be stored

    """ 

    # Load the DtSE model
    model = DtSEModel()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    # Load the book
    story = ''
    with open(book_path, 'r', encoding='utf8') as file:
        lines = file.readlines()
        for line in lines:
            line.replace('\n', '')
            if line != '':
                story += line

    # Construct the prompt in which an LLM will be required to generate the data detailed in the function description
    prompt = f"""
    Given the following story
    
    Please provide:
    1.(characters) A list of characters (excluding any under 20), each with their age (integer between 0 and 100), sex ('male' or 'female'), and five adjectives describing their voice (e.g., deep, soft, booming) do NEVER assign female associated adjectives such as gentle to males or the other way around.
    2. (story_breakdown) For each of the main characters find exactly 1 fragment of speech dialogue about 6-8 words long, find one that is important to the plot. Also return it as such. Also make sure the speech is a full spoken part, so find one for every main character as the speaker.I expect a good number of different speakers. specify the speaker and provide an emotion vector (a list of eight values between 0 and 1 representing Happiness, Sadness, Disgust, Fear, Surprise, Anger, Other, Neutral) also provide a context in each dialogue that describes the context around what is being spoken use the key 'context' for it. Also provide a key 'speaking_rate' this should be an integer between 9 and 15 determining how quickly the character speeks in this exact dialogue 9 being the lowest, 12 average and 15 quickets.
    Present the character list as a JSON array of objects, and the story breakdown as a JSON array of objects with fields 'type', 'text', and for dialogue, 'speaker' and 'emotion'.
    place both of them in the same JSON object. Don't forget about the context of the dialogue, it is important to understand the characters and their emotions. Also make sure to not inclue any character more than once.

    Story
    {story}
    """

    # Call LLM and parse response
    data = llm_call(prompt)

    # Get the characters
    characters = data["characters"]

    # Get the speaker embedding for every character
    character_embeddings = {}
    for char in characters:
        embedding = get_speaker_embedding(char["age"], char["sex"], char["voice"], components.get_word_embedding, model)
        character_embeddings[char["name"]] = embedding

    # Loop through the dialogue and assign the correct speaker embedding to the Speaker Embedded Expressive Dialogue (SEED)
    SEED = []
    for i in data['story_breakdown']:
        speaker_embedding, character_description = get_closest_match(i['speaker'], character_embeddings, characters)
        SEED.append([i['speaker'], i['text'], i['emotion'], speaker_embedding, i['context'], character_description, i['speaking_rate']])
    
    # Save the SEED
    with open(SEED_path, 'wb') as f:
        pickle.dump(SEED, f)

# Book To Speaker Embedded Emotional Audiobook
def BTSEEB(components, model_path, book_path,SEEB_path):
    """Book To Speaker Embedded Expressive Book (DtSEEB). This function takes a book and analyses it using a LLM
    The LLM will respond with a list of character descriptions that are interpretable for the DtSE model to make
    speaker embeddings from. Secondly the LLM provides a full ordered story breakdown seperating the text spoken by the narrator and characters together with their respective
    emotional tone and speaking rate.

    Args:
        components (Object): The get word embedding component
        model_path (str): The path to the DtSE model
        book_path (str): The path to the book (txt)
        SEEB_path (str): The path where the Speaker Embedded Expressive Book (SEEB) will be stored

    """ 

    # Load the DtSE model
    model = DtSEModel()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    # Load the book
    story = ''
    with open(book_path, 'r', encoding='utf8') as file:
        lines = file.readlines()
        for line in lines:
            line.replace('\n', '')
            if line != '':
                story += line

    # Construct the prompt in which an LLM will be asked to generate the data detailed in the function description
    prompt = f"""
    Given the following story.
    
    Please provide a dictionary with two keys:

    1. (characters) A list of all characters with dialogue, and a narrator voice (The main character if not clear), each with their age (integer between 0 and 100), sex ('male', 'female'),and five adjectives describing their voice (e.g., deep, soft, booming) do NEVER assign female associated adjectives such as gentle to males or the other way around, use the key 'voice.
    2. (story_breakdown) A full ordered breakdown of the story into a sequence narration and dialogue. Every quote of a non-narrating character should be a seperate segment of just the quote. The narration and narrator quote's should be together in a segment, segements should not be limited to 1 sentence, if spoken by the same person or narrator it should be as long as it is. So if there is a sentence like. He said "Hello", this should be 2 segements one of the narrator "He said", and one of the speaker "Hello". For each segment specify the speaker and an emotion vector (a list of eight values between 0 and 1 representing Happiness, Sadness, Disgust, Fear, Surprise, Anger, Other, Neutral), also add a speaking rate this must be and integer from 1 to 5, 1 being slow and 5 being fast.
    Remember to always seperate speech and dialogue something like "charcter said "hi"" should be 2 segments, one of the narrator and one of the character. Also make sure to not inclue any character more than once.
    Present the character list as a JSON array of objects, and the story breakdown as a JSON array of objects with fields  'text', 'speaker', 'emotion' and 'speaking_rate'.
    Story:
    {story}
    """

    # Call LLM and parse response
    data = llm_call(prompt)

    # Get the characters
    characters = data["characters"]

    # Get the speaker embedding for every character
    character_embeddings = {}
    for char in characters:
        embedding = get_speaker_embedding(char["age"], char["sex"], char["voice"], components.get_word_embedding, model)
        character_embeddings[char["name"]] = embedding

    # Loop through the dialogue and assign the correct speaker embedding to the Speaker Embedded Expressive Dialogue (SEED)
    SEEB = []

    # Combine the dialogue of the same speaker into one segment
    running_dialogue = ''
    for x,i in enumerate(data['story_breakdown']):
        speaker_embedding, character_description = get_closest_match(i['speaker'], character_embeddings, characters)
        if x+2 <= len(data['story_breakdown']) and i['speaker'] == data['story_breakdown'][x+1]['speaker']:
            running_dialogue += i['text']+'. ' 
        else:
            running_dialogue +=  i['text']
            SEEB.append([i['speaker'], running_dialogue, i['emotion'], speaker_embedding, character_description, i['speaking_rate']])
            running_dialogue = ''

    # Save the SEEB
    with open(SEEB_path, 'wb') as f:
        pickle.dump(SEEB, f)
