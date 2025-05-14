from analysis import DtSE_test, analyze_survey_results, correlation_analysis
from data import annotate_and_process_wavs, get_celebrity_descriptions
from inference import BTSEEB, BTSEED
from data_proccesing import *
from components import *
from train import *
from synthesize import SyntheticDataGenerator
from zonos import Zonos

if __name__ == "__main__":

    # Define the paths to all components that will be used
    analysis_image_path = "data/analysis.png"             # Path to the embedding analysis image
    glove_model_path = "data/glove_model.pkl"             # GloVe model path
    DtSE_model_path = "data/DtSE_model.pth"               # Description To Speaker Embedding model
    SES_model_path = "data/SES_model.pth"                 # Speaker Embedding Similarity model
    DTSC_train_path = "data/DTSC_train.csv"               # Description to Similar Celebrities training data
    DTSC_test_path = "data/DTSC_test.csv"                 # Description to Similar Celebrities testing data
    CD_train_path = "data/CD_train.json"                  # Celebrities Descriptions training data
    CD_test_path = "data/CD_test.json"                    # Celebrities Descriptions testing data
    SE_path = "data/SE.pt"                                # Speaker Embeddings data
    SE_train_path="data/SE_train.pt"                      # Speaker Embeddings training data
    SE_test_path = "data/SE_test.pt"                      # Speaker Embeddings testing data
    voxceleb_ids_path = "data/voxceleb1.csv"              # Voxceleb youtube IDs to name mapping https://www.kaggle.com/datasets/yosrahashem/voxceleb/data
    voxceleb_wavs_path = "vox1_dev_wav/wav"               # Voxceleb dataset from kaggle https://www.kaggle.com/datasets/namhocayai/voxceleb1?resource=download-directory&select=vox1_dev_wav
    output_dir_celebrity_wavs = "data/celebrity_wavs"     # Directory in which 15-45 second celebrity voice snipet wavs will be saved
    book_path = "data/hard_times.txt"                     # The path to the book
    story_path = "data/story.txt"                         # The path to the story
    SEED_path = book_path.split('.')[0] + '_SEED.pkl'     # The path to the output SEED (Speaker Embedded Expressive Dialogue)
    SEEB_path = book_path.split('.')[0] + '_SEEB.pkl'     # The path to the output SEEB (Speaker Embedded Expressive Audiobook)
    survey_results_path = "data/survey_results.csv"       # The path to the survey results
    DtSE_test_fig_output_path = "data/DtSE_test_plot.png" # The path to the DtSE test figure is saved
    k_lambda_steepness_search = 4                         # The number of steepness values that will be searched during hyperparameter tuning
    learnings_rates = [0.001, 0.0005, 0.0001]             # The learning rates that will be searched during hyperparameter tuning
    num_epochs_SES = 20                                   # The number of epochs for training the SES models
    num_epochs_DtSE = 60                                  # The number of epochs for training the DtSE models
    normality_plot_path = "data/normality_plot.png"       # The path to the normality plot
    
    # Initialize components (Will load GloVe model)
    components = Components(glove_model_path=glove_model_path)

    # Annotate and process the VoxCeleb dataset
    annotate_and_process_wavs(voxceleb_ids_path, voxceleb_wavs_path, output_dir_celebrity_wavs)

    # Initialize the Zonos model
    zonos = Zonos()

    # # Generate speaker embeddings from the celebrity voice snippet wavs and split them into training and testing sets
    zonos.WAVDtSE(wav_path=voxceleb_wavs_path, SE_path=SE_path, SE_train_path=SE_train_path, SE_test_path=SE_test_path, CD_train_path=CD_train_path, CD_test_path=CD_test_path, voxceleb_ids_path=voxceleb_ids_path, output_dir_celebrity_wavs=output_dir_celebrity_wavs)
    
    # Get description (sex, age, 5 words describing their voice) for each celebrity in the training and testing sets
    get_celebrity_descriptions(SE_train_path, CD_train_path)
    get_celebrity_descriptions(SE_test_path, CD_test_path)

    # Generate synthetic data for the DTSC model training and testing. 
    # This data consists of of a fictional character with (sex, age, 5 words describing their voice),
    # and a list of 5 celebrities that the large langue model thinks sounds similar to the fictional character.
    synthetic_data_generator = SyntheticDataGenerator(n_similar_celebrities=10)
    synthetic_data_generator.synthezise_DTSC_data(SE_train_path, DTSC_train_path, n=200)
    synthetic_data_generator.synthezise_DTSC_data(SE_test_path, DTSC_test_path, n=40)

    # Analyses the correlation between sex in speaker embeddings from the celebrity training data
    correlation_analysis(SE_train_path, CD_train_path, analysis_image_path, normality_plot_path)
    
    # Initialize Data proccesor 
    data_proccesor = DataProcessor(components, SE_path,DTSC_train_path, DTSC_test_path, CD_train_path, CD_test_path)

    # Get the dataloaders for model training
    DtSE_train_loader, DtSE_val_loader, DtSE_test_loader = data_proccesor.get_DtSE_loaders()
    SES_train_loader, SES_val_loader, SES_test_loader = data_proccesor.get_SES_loaders()
    
    # Initialize the model training class
    model_training = ModelTraining(components)
    
    # Train the Speaker Embedding Similarity model (SES), this model is trained on the Celebrity Descriptions (CD) dataset.
    # Each celebrity is paired to each other similarity, and the model is optimizted to minimize the difference
    # the output similarity score of embeddings pairs with the expected output similarity score.
    # This score is calculated using the difference in the celebrity descriptions using the sex, age and word embeddings from the 5 desccriptive words
    model_training.SES_train(SES_train_loader, SES_val_loader, SES_test_loader,SES_model_path, num_epochs=num_epochs_SES)
    
    # The Description To Speaker Embedding (DtSE) model is trained on the synthetic data generated above.
    # For each character description an speaker embedding of size 128 will be generated. This output will then be
    # Compared to the speaker embeddings of the 5 celebrities that the LLM thinks sounds similar to the character.
    # How well these embeddings match is calculated using both the SES model and cosine similarity.
    # Hyper parameter tuning is done for both the learning rate and k_lamdas steepness.
    # The k_lambdas is a list of 5 numbers that sum to 1, and are used to weight the similarity scores of the 5 celebrities.
    # The score of the best match will be weighted higher the second lowers and so onwards. The steepness hyperparameter controls how steep the drop is.
    model_training.DtSE_train(components, DtSE_train_loader, DtSE_val_loader, DtSE_test_loader, num_epochs_DtSE,DtSE_model_path, SES_model_path, k_lambda_steepness_search, learnings_rates, DtSE_test ,SE_test_path, DTSC_test_path, DtSE_test_fig_output_path)
    
    # Calcute the relative DtSE score for the final model.
    DtSE_score = DtSE_test(components.get_word_embedding, model_path=DtSE_model_path, SES_model_path=SES_model_path, speaker_embeddings_test_path=SE_test_path, DTSC_test_path=DTSC_test_path, fig_output_path=DtSE_test_fig_output_path)
    
    # Book To Speaker Embedded Expressive Dialogue (BTSEED) takes a book and the trained DtSE model and
    # Generates a Speaker Embedded Expressive Dialogue (SEED) file. Using a llm is engineered to take a book and return
    # Characters with a sex, age and 5 words describing their voice. The DtSE model is then used to generate a speaker embedding for each character.
    # These will then be integrated with the dialogue with corresponding emotion chosen by the llm 
    BTSEED(components, DtSE_model_path, book_path, SEED_path)

    # Book To Speaker Embedded Expressive Book (BTSEEB) takes a book and the trained DtSE model and
    # Generates a Speaker Embedded Expressive Book (SEEB) file. Using a llm is engineered to take a book and return
    # Characters with a sex, age and 5 words describing their voice. The DtSE model is then used to generate a speaker embedding for each character.
    # The book will then be segmented into one character spoken dialogue or narration and assign emotion a emotion vector and speaking rate
    BTSEEB(components, DtSE_model_path, story_path, SEEB_path)

    # A function that takes the SEED file and display's it's readable contents.
    view_SEED(path=SEED_path)

    # The Speaker Embedded Expressive Dialogue To Spoken Dialogue (SEEDTSD) function takes the SEED file and the Zonos model and generates a spoken dialogue used for the survey.
    zonos.SEEDTSD(SEED_path, book_name=book_path.split('.')[0].split('/')[-1], concat=False)

    # The Speaker Embedded Expressive Book (SEEB) function takes the SEEB file and the Zonos model and generates an audiobook used for the demonstration.
    zonos.SEEDTSD(SEEB_path, book_name=book_path.split('.')[0].split('/')[-1], concat=True)

    # The SEED file is used to design a survey that will be analyzed and tested for three hypotheses
    analyze_survey_results(survey_results_path)