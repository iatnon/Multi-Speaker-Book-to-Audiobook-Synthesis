import os
import wave
import csv
from pydub import AudioSegment
import torch
from components import *

def get_wav_duration(file_path):
    """Calculate the duration of a WAV file in seconds.

    Args:
        file_path (str): The path to a wav file

    Returns:
        float: The duration of a wav
    """ 
    with wave.open(file_path, 'r') as wav_file:
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
        return frames / float(rate)

def concatenate_and_trim_wavs(wav_files, output_path, max_duration=45):
    """Concatenate a list of WAV files and trim to a maximum duration in seconds.

    Args:
        wav_files (List of WAVS): A list of WAV files to be concatinated and trimmed
        output_path (str): The output path in which the final concatinated and trimmed verion will be stored
        max_duration (int, optional): The maximum duration of a final WAV. Defaults to 45.
    """    

    combined = AudioSegment.empty()

    # Loop through the wavs
    for wav_file in wav_files:

        # Get it's segment and add it to the total
        segment = AudioSegment.from_wav(wav_file)
        combined += segment

    # Trim to max_duration seconds if longer (pydub works in milliseconds)
    if len(combined) > max_duration * 1000:
        combined = combined[:max_duration * 1000]

    # Save the final WAV file
    combined.export(output_path, format="wav")

def annotate_and_process_wavs(voxceleb_ids_path, voxceleb_wavs_path, output_dir_celebrity_wavs):
    """Annotates the WAV files with the corresponsing celebrity and trims 

    Args:
        voxceleb_ids_path (str): The path to the mapping file from Youtube ID to Celebrity name
        voxceleb_wavs_path (str): The path to the VoxCeleb dataset containing all the WAVS
        output_dir_celebrity_wavs (str): The output directory where all annotated and processed WAVS are to be stored
    """    

    # The CSV is expected to have a "uri" field formatted as "Speaker/YouTubeID"
    ytid_to_speaker = {}

    # Open the mapping file
    with open(voxceleb_ids_path, newline="") as csvfile:
        reader = csv.DictReader(csvfile)

        # Loop through every Youtube ID to celebrity name map
        for row in reader:

            # Get the URI
            uri = row["uri"].strip()
            parts = uri.split('/')

            # skip malformed rows
            if len(parts) != 2:
                continue 
            speaker, ytid = parts

            # In case the same YouTube id appears multiple times, they should all map to the same speaker.
            ytid_to_speaker[ytid] = speaker

    # Make a new directory at the output directory
    os.makedirs(output_dir_celebrity_wavs, exist_ok=True)

    min_duration = 15  # seconds: skip speakers with less than this total duration
    max_duration = 45  # seconds: final output is trimmed to this length

    # We'll scan every id folder, and for each subfolder (named as YouTube id) that is in our CSV mapping,
    # We'll accumulate its .wav files.
    speaker_to_wav_files = {}

    for id_folder in os.listdir(voxceleb_wavs_path):
        id_folder_path = os.path.join(voxceleb_wavs_path, id_folder)
        if not os.path.isdir(id_folder_path):
            continue

        # Process each subfolder, which should be named as a YouTube id
        for ytid_folder in os.listdir(id_folder_path):
            ytid_folder_path = os.path.join(id_folder_path, ytid_folder)
            if not os.path.isdir(ytid_folder_path):
                continue
            
            # Check if this folder's name (assumed to be a YouTube id) exists in our mapping
            if ytid_folder not in ytid_to_speaker:
                continue  # skip folders that are not in the CSV
            
            speaker = ytid_to_speaker[ytid_folder]

            # List all WAV files in this YouTube folder
            wav_files = [os.path.join(ytid_folder_path, f)
                        for f in os.listdir(ytid_folder_path)
                        if f.lower().endswith(".wav")]
            if not wav_files:
                continue
            
            if speaker not in speaker_to_wav_files:
                speaker_to_wav_files[speaker] = []
            speaker_to_wav_files[speaker].extend(wav_files)

    # Process each speaker: concatenate and trim if total duration meets threshold
    for speaker, files in speaker_to_wav_files.items():
        if not files:
            continue
        total_duration = sum(get_wav_duration(f) for f in files)
        if total_duration < min_duration:
            print(f"Skipping {speaker}: total duration {total_duration:.2f}s is less than {min_duration}s")
            continue

        output_path = os.path.join(output_dir_celebrity_wavs, f"{speaker}.wav")
        concatenate_and_trim_wavs(files, output_path, max_duration)
        print(f"Saved {output_path} (concatenated from {len(files)} files, total duration {total_duration:.2f}s)")
    print("Processing complete.")

def get_celebrity_descriptions(SE_path, CD_output_path):
    """Get descriptions for all celebrities (sex, age and 5 words describing their voice)

    Args:
        SE_path (str): The path with each celebrity name
        CD_output_path (str): The output path where the Celebritiy Descriptions (CD) will be stored
    """       

    # Load the file and get the celebirity names
    celebrity_speaking_embeddings = torch.load(SE_path)
    celebrities = list(celebrity_speaking_embeddings.keys())

    # Format the prompt to get the desired response
    prompt = f"""
    A list of celebrities will be provided, it is your task to assign each one of them a sex (M/F) an age (int) and a 5-word description of their voice.
    Please resond in valid JSON format which each key being a celebrity and the 7 items a list.
    Here is an example: "Stephen Fry": [
        "M",
        58,
        "rich",
        "warm",
        "articulate",
        "witty",
        "resonant"
    ],
    Do this for all the following celebrities: {celebrities}
    """

    # Call to an LLM (replace this function or the logic inside it with your own LLM call)
    celebrity_descriptions = llm_call(prompt, data_type=list)

    # Save the Celebrity descriptions
    with open(CD_output_path, 'w') as f:
        json.dump(celebrity_descriptions, f, indent=4)