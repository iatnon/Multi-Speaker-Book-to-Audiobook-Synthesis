
import time
import librosa
import torch
import pickle
import torchaudio
import os
from pydub import AudioSegment
from Zonos.zonos.model import Zonos
from Zonos.zonos.conditioning import make_cond_dict

class Zonos:
    """
    The Zonos model has the capability to clone voices and create speaker embeddings for them. These speaker embeddings can be used to generate speech.
    Run setup.py to download the model. The model can only be ran using CUDA on a Linux machine.
    """    
    def __init__(self, model_version="Zyphra/Zonos-v0.1-transformer"):
        """Initialize the Zonos model

        Args:
            model_version (str, optional): The zonos model vesion. Defaults to "Zyphra/Zonos-v0.1-transformer".
        """           

        # Get the device and check if it's cuda as Zonos can only run on Cuda
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        if device == "cpu":
            print("The Zonos model is not capable of running on CPU. Please use a GPU.")
        else:
            # Load model
            self.model = Zonos.from_pretrained(model_version, device=device)
        
    def generate_speech(self, speaker_embedding, text, language="en-us", emotion = [0.3077, 0.0256, 0.0256, 0.0256, 0.0256, 0.0256, 0.2564, 0.3077], pitch_std=20, output_file='output.wav', speaking_rate=15):
        """Generates speech from a speaker embedding, text, emotion, pitch standard deviation and speaking rate

        Args:
            text (str): The text to be spoken out
            speaker_embedding (Torch.tensor): A speaker embedding of size 1.1.128.
            language (str, optional): The language the text is spoken in. Defaults to "en-us".
            emotion (list, optional): The emotion controlling the prevelance of the following emotion in order: Happiness, Sadness, Disgust, Fear, Surprise, Anger, Other, Neutral. Defaults to [0.3077, 0.0256, 0.0256, 0.0256, 0.0256, 0.0256, 0.2564, 0.3077].
            pitch_std (int, optional): The standard deviation of the pitch, if changed it could distrubt the speech. Defaults to 20.
            output_file (str, optional): The path to which the generated speech is to be saved. Defaults to 'output.wav'.
            speaking_rate (int, optional): The speaking rate, the lower this is the slower the speech is. Defaults to 15.
        """           

        # Display the text that is being generated
        print(f"Generating: {text}")

        # Create conditioning for the speech
        cond_dict = make_cond_dict(
            text=text,
            speaker=speaker_embedding,
            language=language,
            emotion=emotion,
            pitch_std=pitch_std,
            speaking_rate=speaking_rate
        )
        conditioning = self.model.prepare_conditioning(cond_dict)

        # Generate audio
        codes = self.model.generate(conditioning)
        wavs = self.model.autoencoder.decode(codes).cpu()

        # Save the speech audio to the specified path
        torchaudio.save(output_file, wavs[0], self.model.autoencoder.sampling_rate)

    def SEEDTSD(self, SEED_path, book_name, concat=True):
        """Speaker Embedded Emotion Dialogue To Spoken Dialogue ()

        Args:
            SEED_path (str): The path to the Speaker Embedded Emotion Dialogue (SEED)
            book_name (str): The name of the book that is being processed
        """        

        # Load the SEED
        with open(SEED_path, 'rb') as f:
            SEED = pickle.load(f)

        if concat:
            output_file_paths = []
            silence = AudioSegment.silent(duration=1000)

        # Generate the specified seech for each entry in the SEED list
        for x,i in enumerate(SEED):
            if concat:
                output_file_path = f"output/{book_name}_{x}.wav"
                output_file_paths.append(output_file_path)
            else:
                output_file_path = f"output/{book_name}_{i[0]}.wav"
            self.generate_speech(
                i[3].bfloat16(),
                i[1],
                language="en-us",
                emotion=i[2],
                pitch_std=20,
                speaking_rate=i[6],
                output_file=output_file_path
            )
        if concat:
            # Wait for all the files to be proccesed
            time.sleep(5)
            # Concatenate all the generated speech files into one
            combined = AudioSegment.empty()
            for file_path in output_file_paths:
                segment = AudioSegment.from_wav(file_path)
                # Add silence between segments
                combined += segment + silence
            combined.export(f"output/{book_name}_audiobook.wav", format="wav")
            print(f"Generated audiobook saved to: output/{book_name}.wav")

    def WAVDtSE(self, wav_path, SE_path, SE_train_path, SE_test_path):
        """WAV Directory To Speaker Embeddings takes a directory path with wav files and turn them into Zonos speaker embeddigns

        Args:
            wav_path (str): The path to a directory of WAV files (without noise and 15-45 seconds long)
            SE_path (str): The path where all Speaker Embeddings (SE) are to be stored
            SE_train_path (str): The path where the training Speaker Embeddings (SE) are to be stored
            SE_test_path (str): The path where the testing Speaker Embeddings (SE) are to be stored
        """        

        speaker_embeddings = {}
        files = os.listdir(wav_path)

        # Loop through all the files in the directory and convert them into speaker embeddings
        for file in files:
            file_path = os.path.join(wav_path, file)
            if os.path.isfile(file_path):
                extension = os.path.splitext(file)[1].lower()
                if extension == '.wav':
                    file_name = os.path.splitext(file)[0]
                    wav, sampling_rate = librosa.load(file_name)
                    wav = torch.from_numpy(wav)
                    speaker_embedding = self.model.make_speaker_embedding(wav, sampling_rate)
                    speaker_embeddings[file_name] = speaker_embedding
                
        # Split the data into testing and training
        keys = list(speaker_embeddings.keys())
        midpoint = len(keys) // 2 
        train_keys = keys[:midpoint]
        test_keys = keys[midpoint:]

        # Create two dictionaries
        speaker_embeddings_train = {k: speaker_embeddings[k] for k in train_keys}
        speaker_embeddings_test = {k: speaker_embeddings[k] for k in test_keys}   

        # Save the data
        torch.save(speaker_embeddings, SE_path)
        torch.save(speaker_embeddings_train, SE_train_path)
        torch.save(speaker_embeddings_test, SE_test_path)

