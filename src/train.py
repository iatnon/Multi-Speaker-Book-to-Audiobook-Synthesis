import torch
import torch.nn as nn
import random
from components import generate_k_lambda_list
from models import DtSEModel, SESModel
import torch.nn.functional as F
import os
class ModelTraining():
    """
    The model training class can be used to train both the SES and DtSE model
    """    
    def __init__(self, components, lambda_age=0.5):
        """Initialize the ModelTraining class

        Args:
            components (Object): Includes the function to get a GloVe word embedding
            lambda_age (float, optional): Determines the weight given to the age in the SES model. Defaults to 0.5.
        """        

        # Initialie the components used in both the SES and DtSE models
        self.components = components
        self.DtSE_model = DtSEModel()
        self.SES_model = SESModel()
        self.criterion_sex = nn.BCELoss()
        self.criterion_age = nn.MSELoss()
        self.criterion = nn.MSELoss()
        self.lambda_age = lambda_age

    def SES_train(self, train_loader, val_loader, test_loader,saved_model_path, num_epochs=20, learning_rate=0.001):
        """Trains the Speaker Embedding Similarity (SES) model

        Args:
            train_loader (DataLoader): The training loader
            val_loader (DataLoader): The validation loader
            test_loader (DataLoader): The testing loader
            saved_model_path (str): The path to which the modle should be saved
            num_epochs (int, optional): The number of epochs. Defaults to 20.
            learning_rate (float, optional): The learning rate. Defaults to 0.001.
        """        

        # Initialize the optimizer
        optimizer = torch.optim.Adam(self.SES_model.parameters(), lr=learning_rate)
        best_val_loss = float('inf')

        # Train the model and save the model with the best validation score
        for epoch in range(num_epochs):
            self.SES_model.train()
            running_loss = 0.0
            for emb1, emb2, targets in train_loader:
                optimizer.zero_grad()
                outputs = self.SES_model(emb1, emb2)
                loss = self.criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                
            train_loss = running_loss / len(train_loader)
            
            # Validation step
            self.SES_model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for emb1, emb2, targets in val_loader:
                    outputs = self.SES_model(emb1, emb2)
                    loss = self.criterion(outputs, targets)
                    val_loss += loss.item()
            val_loss = val_loss / len(val_loader)
            
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save model if validation loss improves
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.SES_model.state_dict(), saved_model_path)
                print(f"New best model saved at epoch {epoch+1} with val loss {val_loss:.4f}")

        # Load best model
        self.SES_model.load_state_dict(torch.load(saved_model_path))
        self.SES_model.eval()

        # Test the model
        test_loss = 0.0
        predictions = []
        targets_list = []
        with torch.no_grad():
            for emb1, emb2, targets in test_loader:
                outputs = self.SES_model(emb1, emb2)
                loss = self.criterion(outputs, targets)
                test_loss += loss.item()
                predictions.extend(outputs.squeeze().tolist())
                targets_list.extend(targets.squeeze().tolist())


        test_loss = test_loss / len(test_loader)
        print(f"\nTest Loss: {test_loss:.4f}")


    def DtSE_train(self, components, train_loader, val_loader,  test_loader,num_epochs, saved_model_path, SES_model_path, k_lambda_steepness_search, learnings_rates, DtSE_test, SE_test_path, DTSC_test_path, fig_output_path):
        """Train the Description To Speaker Embedding (DtSE) model

        Args:
            components (Object): Includes the function to get a GloVe word embedding
            train_loader (DataLoader): The Data Loader for the training data
            val_loader (DataLoader): The Data Loader for the validation data
            test_loader (DataLoader): The Data Loader for the testing data
            num_epochs (int): The number of epochs
            saved_model_path (str): The path to which the model is to be saved
            SES_model_path (str): The path to the trained Speaker Embedding Similarity (SES) model
            k_lambda_steepness_search (int): The number of k_lambda_steepness values to be searched during hyperparameter tuning
            learnings_rates (List[float]): A list of learning rates to be searched during hyperparameter tuning
            DtSE_test (Function): A test for the DtSE model to test it's relative performance (The maximum target rank between all embeddings)
            SE_test_path (str): The path to the Speaker Embeddings (SE)
            DTSC_test_path (str): The path Description To Similar Celebrities (DTSC) test path for the DtSE_test (Also used to create the validation and test loaders)
        """           
        # Load the SES model
        SES_model = SESModel()
        SES_model.load_state_dict(torch.load(SES_model_path))
        SES_model.eval()
        # Perfrom hyperparameter training over learning rates and k_lambdas with the DtSE test as performce metric
        best_DtSE_score = float('inf')
        
        for steepness in range(k_lambda_steepness_search):
            print(f"testing steepness: {((steepness+1)/k_lambda_steepness_search)+0.5}")
            # Generate a k lambdas list for the steepness value to be generated
            k_lambdas = generate_k_lambda_list(steepness=((steepness+1)/k_lambda_steepness_search)+0.5, n=5)
            print(f"k_lambdas: {k_lambdas}")
            for learning_rate in learnings_rates:

                # Initialze the optimizer with the learning rate being tested
                optimizer = torch.optim.Adam(self.DtSE_model.parameters(), lr=learning_rate)
                best_val_loss = float('inf')    
                for epoch in range(num_epochs):
                    # Training phase
                    self.DtSE_model.train()
                    train_loss = 0.0
                    num_batches_train = 0
                    
                    for input_batch, targets_batch in train_loader:
                        output = self.DtSE_model(input_batch)
                        if output.shape[0] == targets_batch.shape[0]:
                            loss = self.DtSE_loss(output, targets_batch, SES_model, k_lambdas)
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                            train_loss += loss.item() # Accumulate loss
                            num_batches_train += 1
                        train_loss /= num_batches_train  # Average over training samples

                    # Validation phase
                    self.DtSE_model.eval()
                    val_loss = 0.0
                    num_batches_val = 0
                    with torch.no_grad():
                        for input_batch, targets_batch in val_loader:
                            if output.shape[0] == targets_batch.shape[0]:
                                loss = self.DtSE_loss(output, targets_batch, SES_model, k_lambdas)
                    
                            val_loss += loss.item() # Accumulate loss
                            num_batches_val += 1
                    val_loss /= num_batches_val  # Average over validation samples

                    print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

                    # Save best model based on validation loss
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss

                        # Save the model in a temporary path
                        torch.save(self.DtSE_model.state_dict(), saved_model_path.split('.')[0] + f"_test.pth")
                
                # Load best model for testing
                self.DtSE_model.load_state_dict(torch.load(saved_model_path.split('.')[0] + f"_test.pth"))
                self.DtSE_model.eval()

                # Test phase
                test_loss = 0.0 
                num_batches_test = 0
                with torch.no_grad():
                    for input_batch, targets_batch in test_loader:
                        output = self.DtSE_model(input_batch)
                        loss = self.DtSE_loss(output, targets_batch,  SES_model, k_lambdas)
                
                        test_loss += loss.item() 
                        num_batches_test += 1

                # Average over test samples
                test_loss /= num_batches_test  
                print(f"Test Loss: {test_loss:.4f}")
                DtSE_score = DtSE_test(components.get_word_embedding, saved_model_path.split('.')[0] + f"_test.pth", SES_model_path,SE_test_path, DTSC_test_path, fig_output_path.split('.')[0] + f"_temp.png")
                # DtSE_score = DtSE_test(components.get_word_embedding, model_path=saved_model_path.split('.')[0] + f"_test.pth", SES_model_path=SES_model_path, speaker_embeddings_test_path=SE_test_path, DTSC_test_path=DTSC_test_path)
                
                print(f"DtSE score: {DtSE_score:.4f}")
                print(f"Best DtSE score: {best_DtSE_score:.4f} with steepness: {(steepness+1)/k_lambda_steepness_search:.2f} and learning rate: {learning_rate:.4f}")
                if best_DtSE_score > DtSE_score:
                    best_DtSE_score = DtSE_score
                    best_steepness = ((steepness+1)/k_lambda_steepness_search)+0.5
                    best_learning_rate = learning_rate
                    os.replace(fig_output_path.split('.')[0] + f"_temp.png", fig_output_path)
                    # Save the current best performing model based on the DtSE test to the model output location
                    torch.save(self.DtSE_model.state_dict(), saved_model_path)
                else:
                    os.remove(fig_output_path.split('.')[0] + f"_temp.png")
                    os.remove(saved_model_path.split('.')[0] + f"_test.pth")

        # Display  the best hyperparameters
        print(f"Best DtSE score: {best_DtSE_score:.4f} with steepness: {best_steepness:.2f} and learning rate: {best_learning_rate:.4f}")
    def DtSE_loss(self, output, targets_batch, SES_model, k_lamdas, verbosity = 0):
        """_summary_

        Args:
            output (List[Torch.tensor]): The 128 dimensional speaker embedding outputted from the model (In a batch)
            targets_batch (List[List[Torch.tensor]]): The 5 target speaker embeddings the output should be similar too (In a batch) 
            SES_model (Model): The Speaker Embedding Simmilarity model used to calculate the similarity between speaker embedding trained on how they sound
            k_lamdas (List): A list of lambdas giving weight to the ranked similarities (Give the most similar score more weight)
            verbosity (int, optional): The verbosity (If set to 1 the loss is printed). Defaults to 0.

        Returns:
            float: The loss of the batch
        """        

        # Repeat the output
        output_repeated = output.unsqueeze(1).repeat(1, 5, 1)  # [64, 5, 128]

        # Get the batch size
        batch_size = output.shape[0]

        # Compute Speaker Embedding Similarity (SES)
        similarities = SES_model(
            output_repeated.view(-1, 128), 
            targets_batch.view(-1, 128)
        ).view(batch_size, 5) 

        # Get the maximum off the SES similarites
        max_similarity, best_target_idx = similarities.max(dim=1)

        # Compute similarity with random tensor to counter overfitting
        batch_random_tensor = torch.rand(batch_size, 128) * 20 - 10  # [64, 128]
        SES_similarity_random = SES_model(output, batch_random_tensor).squeeze()  # [64]

        # Get the cosine similarity between the output and the targets
        similarities = F.cosine_similarity(output.unsqueeze(1), targets_batch, dim=2)

        # Get a batch of random tensors again to do the same so that the model doesn't overfit
        batch_random_tensor_cos = torch.rand(batch_size,5, 128) * 20 - 10  # [64, 128]
        similarities_random = F.cosine_similarity(batch_random_tensor_cos, targets_batch, dim=2)

        # Rank the similarities from most similar to least similar
        top_values, top_indices = torch.topk(similarities, k=5, dim=1)

        # Weight the similarity scores by the k_lambdas (weight the most heighest similarity score the most to optimize the model towards finding at least one suiting voice)
        similarities = k_lamdas[0] * top_values[:, 0] + k_lamdas[1] * top_values[:, 1] + k_lamdas[2] * top_values[:, 2]  + k_lamdas[3] * top_values[:, 2]  + k_lamdas[4] * top_values[:, 2] 

        # Do the same for the random embeddings
        top_values, top_indices = torch.topk(similarities_random, k=5, dim=1)  # Shapes: (16, 2), (16, 2)
        similarities_random = k_lamdas[0] * top_values[:, 0] + k_lamdas[1] * top_values[:, 1] + k_lamdas[2] * top_values[:, 2]  + k_lamdas[3] * top_values[:, 2]  + k_lamdas[4] * top_values[:, 2] 
        
        # Define the loss
        loss = (self.criterion(max_similarity, torch.ones_like(max_similarity)) - 
                self.criterion(SES_similarity_random, torch.ones_like(SES_similarity_random)) + 1) + -similarities.mean() + 1

        # Print the loss if the verbosity is 1 or above
        if verbosity >= 1:
            print(f"Loss: {loss}")

        # Return the loss
        return loss

    def DtSE_train(self, components, train_loader, val_loader,  test_loader,num_epochs, saved_model_path, SES_model_path, k_lambda_steepness_search, learnings_rates, DtSE_test, SE_test_path, DTSC_test_path, fig_output_path):
        """Train the Description To Speaker Embedding (DtSE) model

        Args:
            components (Object): Includes the function to get a GloVe word embedding
            train_loader (DataLoader): The Data Loader for the training data
            val_loader (DataLoader): The Data Loader for the validation data
            test_loader (DataLoader): The Data Loader for the testing data
            num_epochs (int): The number of epochs
            saved_model_path (str): The path to which the model is to be saved
            SES_model_path (str): The path to the trained Speaker Embedding Similarity (SES) model
            k_lambda_steepness_search (int): The number of k_lambda_steepness values to be searched during hyperparameter tuning
            learnings_rates (List[float]): A list of learning rates to be searched during hyperparameter tuning
            DtSE_test (Function): A test for the DtSE model to test it's relative performance (The maximum target rank between all embeddings)
            SE_test_path (str): The path to the Speaker Embeddings (SE)
            DTSC_test_path (str): The path Description To Similar Celebrities (DTSC) test path for the DtSE_test (Also used to create the validation and test loaders)
        """           
        # Load the SES model
        SES_model = SESModel()
        SES_model.load_state_dict(torch.load(SES_model_path))
        SES_model.eval()
        # Perfrom hyperparameter training over learning rates and k_lambdas with the DtSE test as performce metric
        best_DtSE_score = float('inf')
        
        for steepness in range(k_lambda_steepness_search):
            print(f"Steepness: {(steepness+1)/k_lambda_steepness_search}")
            # Generate a k lambdas list for the steepness value to be generated
            k_lambdas = generate_k_lambda_list(steepness=(steepness+1)/k_lambda_steepness_search, n=5)
            print(f"k_lambdas: {k_lambdas}")
            for learning_rate in learnings_rates:

                # Initialze the optimizer with the learning rate being tested
                optimizer = torch.optim.Adam(self.DtSE_model.parameters(), lr=learning_rate)
                best_val_loss = float('inf')    
                for epoch in range(num_epochs):
                    # Training phase
                    self.DtSE_model.train()
                    train_loss = 0.0
                    num_batches_train = 0
                    
                    for input_batch, targets_batch in train_loader:
                        output = self.DtSE_model(input_batch)
                        if output.shape[0] == targets_batch.shape[0]:
                            loss = self.DtSE_loss(output, targets_batch, SES_model, k_lambdas)
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                            train_loss += loss.item() # Accumulate loss
                            num_batches_train += 1
                        train_loss /= num_batches_train  # Average over training samples

                    # Validation phase
                    self.DtSE_model.eval()
                    val_loss = 0.0
                    num_batches_val = 0
                    with torch.no_grad():
                        for input_batch, targets_batch in val_loader:
                            if output.shape[0] == targets_batch.shape[0]:
                                loss = self.DtSE_loss(output, targets_batch, SES_model, k_lambdas)
                    
                            val_loss += loss.item() # Accumulate loss
                            num_batches_val += 1
                    val_loss /= num_batches_val  # Average over validation samples

                    print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

                    # Save best model based on validation loss
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss

                        # Save the model in a temporary path
                        torch.save(self.DtSE_model.state_dict(), saved_model_path.split('.')[0] + f"_test.pth")
                
                # Load best model for testing
                self.DtSE_model.load_state_dict(torch.load(saved_model_path.split('.')[0] + f"_test.pth"))
                self.DtSE_model.eval()

                # Test phase
                test_loss = 0.0 
                num_batches_test = 0
                with torch.no_grad():
                    for input_batch, targets_batch in test_loader:
                        output = self.DtSE_model(input_batch)
                        loss = self.DtSE_loss(output, targets_batch,  SES_model, k_lambdas)
                
                        test_loss += loss.item() 
                        num_batches_test += 1

                # Average over test samples
                test_loss /= num_batches_test  
                print(f"Test Loss: {test_loss:.4f}")

                DtSE_score = DtSE_test(components.get_word_embedding, saved_model_path.split('.')[0] + f"_test.pth", SES_model_path,SE_test_path, DTSC_test_path, fig_output_path.split('.')[0] + f"_temp.png")
                print(f"DtSE score: {DtSE_score:.4f}")
                print(f"Best DtSE score: {best_DtSE_score:.4f} with steepness: {(steepness+1)/k_lambda_steepness_search:.2f} and learning rate: {learning_rate:.4f}")
                if best_DtSE_score > DtSE_score:
                    best_DtSE_score = DtSE_score
                    best_steepness = ((steepness+1)/k_lambda_steepness_search)
                    best_learning_rate = learning_rate
                    os.replace(fig_output_path.split('.')[0] + f"_temp.png", fig_output_path)

                    # Save the current best performing model based on the DtSE test to the model output location
                    torch.save(self.DtSE_model.state_dict(), saved_model_path)
                else:
                    os.remove(fig_output_path.split('.')[0] + f"_temp.png")
                    os.remove(saved_model_path.split('.')[0] + f"_test.pth")

        # Display  the best hyperparameters
        print(f"Best DtSE score: {best_DtSE_score:.4f} with steepness: {best_steepness:.2f} and learning rate: {best_learning_rate:.4f}")