
import pandas as pd
import torch
from components import Components
from inference import get_speaker_embedding
from models import DtSEModel, SESModel
import itertools
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import ttest_ind, shapiro, mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

def correlation_analysis(SE_train_path, CD_train_path, saved_image_path, normality_plot_path):
    """Analyze the celebrity speaker embeddings for correlation in sex

    Args:
        SE_train_path (string): Path to the speaker embeddings
        CD_train_path (string): Path to celebrities descriptions
        saved_image_path (string): Path where the box plot will be saved
        normality_plot_path (string): Path where normality plots (histograms, Q-Q) will be saved
    """    
    # Load the speaker embeddings and celebrity descriptions
    SE_train = torch.load(SE_train_path, map_location="cpu")
    with open(CD_train_path) as json_file:
        CD_train = json.load(json_file)

    results = []

    # Get the names of all the celebrities
    names = list(CD_train.keys())

    for name1, name2 in itertools.combinations(names, 2):
        emb1 = SE_train[name1].squeeze().to(torch.float32).numpy().reshape(1, -1)
        emb2 = SE_train[name2].squeeze().to(torch.float32).numpy().reshape(1, -1)
        
        # Normalize embeddings to ensure cosine similarity is between -1 and 1
        emb1 = emb1 / np.linalg.norm(emb1)
        emb2 = emb2 / np.linalg.norm(emb2)
        sim = cosine_similarity(emb1, emb2)[0][0] 
        
        same_sex = CD_train[name1][0] == CD_train[name2][0]
        
        results.append({
            "pair": (name1, name2),
            "cosine_similarity": sim,
            "same_sex": same_sex
        })

    df = pd.DataFrame(results)

    # Separate cosine similarities into same-sex and different-sex groups
    same_sex_sim = df[df["same_sex"] == True]["cosine_similarity"]
    diff_sex_sim = df[df["same_sex"] == False]["cosine_similarity"]

    # Perform an independent t-test
    t_stat, t_p_value = ttest_ind(same_sex_sim, diff_sex_sim)

    # Perform Mann-Whitney U test
    u_stat, u_p_value = mannwhitneyu(same_sex_sim, diff_sex_sim, alternative="two-sided")

    # Calculate degrees of freedom for t-test
    degrees_of_freedom = len(same_sex_sim) + len(diff_sex_sim) - 2

    # Test for normality using Shapiro-Wilk on a subsample (due to large sample size)
    sample_size = 5000  # Subsample size for Shapiro-Wilk test
    same_sex_sample = np.random.choice(same_sex_sim, size=sample_size, replace=False)
    diff_sex_sample = np.random.choice(diff_sex_sim, size=sample_size, replace=False)

    shapiro_same_stat, shapiro_same_p = shapiro(same_sex_sample)
    shapiro_diff_stat, shapiro_diff_p = shapiro(diff_sex_sample)

    # Print results
    print(f"T-Test Result: t={t_stat:.3f}, p={t_p_value:.3g}, df={degrees_of_freedom}")
    print(f"Mann-Whitney U Test: U={u_stat:.3f}, p={u_p_value:.3g}")
    print(f"Shapiro-Wilk Test (Same-sex sample): W={shapiro_same_stat:.3f}, p={shapiro_same_p:.3g}")
    print(f"Shapiro-Wilk Test (Different-sex sample): W={shapiro_diff_stat:.3f}, p={shapiro_diff_p:.3g}")


def DtSE_test(get_word_embedding, model_path, SES_model_path, speaker_embeddings_test_path, DTSC_test_path, fig_output_path):
    """Test a DtSE model on the test set. In this test the model will create a speaker embedding that will be compared
    to all other speaker embeddings using cosine similarity and the SES model. The rank of the highest similarity will
    then be averaged over all test samples and returned as the result which should be as low as possible.

    Args:
        get_word_embedding (function): get a GLoVe word embedding for a word
        model_path (str): _description_. Defaults to "DtSE_model.pth". This is the path to the DtSE model.
        SES_model_path (str): _description_. Defaults to "SES_model.pth". This is the path to the SES model.
        speaker_embeddings_test_path (str): _description_. Defaults to "CD_test.pt". This is the path to the Celebrity descriptions.
        DTSC_test_path (str): _description_. Defaults to "DTSC_test.csv". This is the path to the test set.

    Returns:
        tuple: (avg_best_rank_cosine, avg_best_rank_ses, best_ranks, best_SES_ranks)
            - avg_best_rank_cosine (float): Average best rank for cosine similarity
            - avg_best_rank_ses (float): Average best rank for SES model
            - best_ranks (list): List of best ranks for cosine similarity
            - best_SES_ranks (list): List of best ranks for SES model
    """    
    # Load celebrity embeddings
    embeddings = torch.load(speaker_embeddings_test_path, map_location="cpu")
    
    # Load and set DtSE model to evaluation mode
    model = DtSEModel()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    # Load and set SES model to evaluation mode
    SES_model = SESModel()
    SES_model.load_state_dict(torch.load(SES_model_path, map_location=torch.device('cpu')))
    SES_model.eval()
    
    # Load the CSV file
    df = pd.read_csv(DTSC_test_path)

    # Lists to store metrics for each description
    best_ranks = []
    best_SES_ranks = []
    average_similarities = []

    # Iterate over each row in the CSV
    for _, row in df.iterrows():
        # Extract description details
        age = row['age']
        sex = row['sex']
        adjectives = [row['adj1'], row['adj2'], row['adj3'], row['adj4'], row['adj5']]
        listed_celebs = [row['celeb1_name'], row['celeb2_name'], row['celeb3_name'], 
                        row['celeb4_name'], row['celeb5_name']]
        
        # Generate embedding for the description
        embedding = get_speaker_embedding(age, sex, adjectives, get_word_embedding, model)
        
        # Compute similarities with all celebrity embeddings
        similarities = {}
        speaker_embedding_similarities = {}
        for name, c_embedding in embeddings.items():
            c_embedding = c_embedding.squeeze()
            similarity = torch.nn.functional.cosine_similarity(embedding, c_embedding, dim=0).item()
            speaker_embedding_similarity = SES_model(embedding.unsqueeze(0), c_embedding.unsqueeze(0))
            similarities[name] = similarity
            speaker_embedding_similarities[name] = speaker_embedding_similarity
        
        # Cosine Similarity Ranks
        # Sort celebrities by similarity (highest first)
        sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        sorted_names = [name for name, _ in sorted_similarities]
        
        # Assign ranks (1 is best, i.e., highest similarity)
        ranks = {name: idx + 1 for idx, name in enumerate(sorted_names)}
        
        # Get ranks of the 5 listed celebrities
        listed_ranks = [ranks[celeb] for celeb in listed_celebs if celeb in ranks]
        if listed_ranks:
            # Find the best rank (smallest number = highest similarity)
            best_rank = min(listed_ranks)
            best_ranks.append(best_rank)
        
        # Get similarities of the 5 listed celebrities
        listed_similarities = [similarities[celeb] for celeb in listed_celebs if celeb in similarities]
        if listed_similarities:
            # Compute average similarity for the listed celebrities
            avg_sim = sum(listed_similarities) / len(listed_similarities)
            average_similarities.append(avg_sim)

        # SES Ranks
        # Sort celebrities by similarity (highest first)
        sorted_similarities = sorted(speaker_embedding_similarities.items(), key=lambda x: x[1], reverse=True)
        sorted_names = [name for name, _ in sorted_similarities]
        
        # Assign ranks (1 is best, i.e., highest similarity)
        ranks = {name: idx + 1 for idx, name in enumerate(sorted_names)}
        
        # Get ranks of the 5 listed celebrities
        listed_ranks = [ranks[celeb] for celeb in listed_celebs if celeb in ranks]
        if listed_ranks:
            # Find the best rank (smallest number = highest similarity)
            best_rank = min(listed_ranks)
            best_SES_ranks.append(best_rank)

    # Compute average and median best ranks
    avg_best_rank_cosine = sum(best_ranks) / len(best_ranks) if best_ranks else float('inf')
    avg_best_rank_ses = sum(best_SES_ranks) / len(best_SES_ranks) if best_SES_ranks else float('inf')
    best_ranks_np = np.array(best_ranks) 
    median_best_rank = np.median(best_ranks_np)
    
    # Print the metrics
    print(f"Average best rank: {avg_best_rank_cosine:.2f}")
    print(f"Average best SES rank: {avg_best_rank_ses:.2f}")
    print(f"Median best rank: {median_best_rank:.2f}")
    # Generate CDF Plot
    # Set a clean style suitable for academic papers
    sns.set_style('whitegrid')

    # Function to compute the empirical CDF
    def ecdf(data):
        sorted_data = np.sort(data)
        n = len(data)
        y = np.arange(1, n + 1) / n  # Cumulative proportion
        return sorted_data, y

    # Compute ECDF for cosine similarity and SES
    x_cosine, y_cosine = ecdf(best_ranks)
    x_ses, y_ses = ecdf(best_SES_ranks)

    # Create the plot
    plt.figure(figsize=(10, 6))  # Width and height suitable for a paper
    plt.plot(x_cosine, y_cosine, 'k-', label='Cosine Similarity')  # Black solid line
    plt.plot(x_ses, y_ses, 'k--', label='SES Model')              # Black dashed line

    # Customize the plot
    plt.xlabel('Best Rank', fontsize=14)
    plt.ylabel('Cumulative Proportion', fontsize=14)
    plt.title('Cumulative Distribution of Best Ranks', fontsize=16)
    plt.legend(fontsize=12)
    plt.xticks(np.arange(0, 601, 50), fontsize=12)  # Ticks every 50 up to 600
    plt.yticks(np.arange(0, 1.1, 0.1), fontsize=12) # Proportion from 0 to 1
    plt.grid(True)

    # Save the plot as a high-quality PDF
    plt.savefig(fig_output_path, bbox_inches='tight')
    plt.close()  # Close the figure to free memory

    # Return average best ranks and the rank lists
    return avg_best_rank_cosine

def DtSE_test(get_word_embedding, model_path, SES_model_path, speaker_embeddings_test_path, DTSC_test_path, fig_output_path):
    """Test a DtSE model on the test set. In this test the model will create a speaker embedding that will be compared
    to all other speaker embeddings using cosine similarity and the SES model. The rank of the highest similarity will
    than be averaged over all test samples and returned as the result which should be as low as possible.

    Args:
        get_word_embedding (function): get a GLoVe word embedding for a word
        model_path (str): _description_. Defaults to "DtSE_model.pth". This is the path to the DtSE model.
        SES_model_path (str): _description_. Defaults to "SES_model.pth". This is the path to the SES model.
        speaker_embeddings_test_path (str): _description_. Defaults to "CD_test.pt. This is the path to the Celebrity descriptions".
        DTSC_test_path (str): _description_. Defaults to "DTSC_test.csv. This is the path to the test set".

    Returns:
        float: score of the test, the lower the better.
    """    
    
    embeddings = torch.load(speaker_embeddings_test_path, map_location="cpu")
    model = DtSEModel()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    SES_model = SESModel()
    SES_model.load_state_dict(torch.load(SES_model_path, map_location=torch.device('cpu')))
    SES_model.eval()
    
    # Load the CSV file
    df = pd.read_csv(DTSC_test_path)

    # Lists to store metrics for each description
    best_ranks = []
    best_SES_ranks = []

    average_similarities = []

    # Iterate over each row in the CSV
    for _, row in df.iterrows():
        # Extract description details
        age = row['age']
        sex = row['sex']
        adjectives = [row['adj1'], row['adj2'], row['adj3'], row['adj4'], row['adj5']]
        listed_celebs = [row['celeb1_name'], row['celeb2_name'], row['celeb3_name'], 
                        row['celeb4_name'], row['celeb5_name']]
        
        # Generate embedding for the description
        embedding = get_speaker_embedding(age, sex, adjectives, get_word_embedding, model)
        
        # Compute similarities with all celebrity embeddings
        similarities = {}
        speaker_embedding_similarities = {}
        for name, c_embedding in embeddings.items():
            c_embedding = c_embedding.squeeze()
            similarity = torch.nn.functional.cosine_similarity(embedding, c_embedding, dim=0).item()
            speaker_embedding_similarity = SES_model(embedding.unsqueeze(0), c_embedding.unsqueeze(0))
            similarities[name] = similarity
            speaker_embedding_similarities[name] = speaker_embedding_similarity
        
        # Sort celebrities by similarity (highest first)
        sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        sorted_names = [name for name, _ in sorted_similarities]
        
        # Assign ranks (1 is best, i.e., highest similarity)
        ranks = {name: idx + 1 for idx, name in enumerate(sorted_names)}
        
        # Get ranks of the 5 listed celebrities
        listed_ranks = [ranks[celeb] for celeb in listed_celebs if celeb in ranks]
        if listed_ranks:
            # Find the best rank (smallest number = highest similarity)

            best_rank = min(listed_ranks)
            # print(best_rank)
            best_ranks.append(best_rank)
        
        # Get similarities of the 5 listed celebrities
        listed_similarities = [similarities[celeb] for celeb in listed_celebs if celeb in similarities]
        if listed_similarities:
            # Compute average similarity for the listed celebrities
            avg_sim = sum(listed_similarities) / len(listed_similarities)
            average_similarities.append(avg_sim)

        # SES
        # Sort celebrities by similarity (highest first)
        sorted_similarities = sorted(speaker_embedding_similarities.items(), key=lambda x: x[1], reverse=True)
        sorted_names = [name for name, _ in sorted_similarities]
        
        # Assign ranks (1 is best, i.e., highest similarity)
        ranks = {name: idx + 1 for idx, name in enumerate(sorted_names)}
        
        # Get ranks of the 5 listed celebrities
        listed_ranks = [ranks[celeb] for celeb in listed_celebs if celeb in ranks]
        # print(listed_ranks)
        if listed_ranks:
            # Find the best rank (smallest number = highest similarity)
            best_rank = min(listed_ranks)
            best_SES_ranks.append(best_rank)
    
    # Compute the averages across all descriptions
    avg_best_rank = sum(best_ranks) / len(best_ranks) if best_ranks else float('nan')
    avg_best_SES_rank = sum(best_SES_ranks) / len(best_SES_ranks) if best_SES_ranks else float('nan')

    # Print the two metrics
    print(f"Average best rank: {avg_best_rank:.2f}")
    print(f"Average best SES rank: {avg_best_SES_rank:.2f}")

    # Generate CDF Plot
    # Set a clean style suitable for academic papers
    sns.set_style('whitegrid')

    # Function to compute the empirical CDF
    def ecdf(data):
        sorted_data = np.sort(data)
        n = len(data)
        y = np.arange(1, n + 1) / n  # Cumulative proportion
        return sorted_data, y

    # Compute ECDF for cosine similarity and SES
    x_cosine, y_cosine = ecdf(best_ranks)
    x_ses, y_ses = ecdf(best_SES_ranks)

    # Create the plot
    plt.figure(figsize=(10, 6))  # Width and height suitable for a paper
    plt.plot(x_cosine, y_cosine, 'k-', label='Cosine Similarity')  # Black solid line
    plt.plot(x_ses, y_ses, 'k--', label='SES Model')              # Black dashed line

    # Customize the plot
    plt.xlabel('Best Rank', fontsize=14)
    plt.ylabel('Cumulative Proportion', fontsize=14)
    plt.title('Cumulative Distribution of Best Ranks', fontsize=16)
    plt.legend(fontsize=12)
    plt.xticks(np.arange(0, 601, 50), fontsize=12)  # Ticks every 50 up to 600
    plt.yticks(np.arange(0, 1.1, 0.1), fontsize=12) # Proportion from 0 to 1
    plt.grid(True)

    # Save the plot as a high-quality PDF
    plt.savefig(fig_output_path, bbox_inches='tight')
    plt.close()  # Close the figure to free memory

    return avg_best_rank

import pandas as pd
import numpy as np
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt

def analyze_survey_results(survey_results_path):
    """
    Analyzes the survey results and performs hypothesis tests on the data.
    """
    
    # Read the CSV
    df = pd.read_csv(survey_results_path)

    # Compute mean ratings
    df.columns = ['character', 'type', 'count_1', 'count_2', 'count_3', 'count_4', 'count_5', 'same_sex_narration', 'mapping']

    # Calculate total ratings and mean rating per row
    df['total_ratings'] = df['count_1'] + df['count_2'] + df['count_3'] + df['count_4'] + df['count_5']
    df['mean_rating'] = (1 * df['count_1'] + 2 * df['count_2'] + 3 * df['count_3'] + 
                         4 * df['count_4'] + 5 * df['count_5']) / df['total_ratings']
    
    # Restructure the data
    # Split by audio type
    df['type'] = df['type'].str.strip()  # Remove extra spaces
    df['type'] = df['type'].str.title()  # Standardize capitalization

    # Filter the DataFrame
    system_df = df[df['type'] == 'System'][['character', 'mean_rating']]
    audiobook_df = df[df['type'] == 'Audiobook'][['character', 'mean_rating', 'same_sex_narration']]
    random_df = df[df['type'] == 'Random'][['character', 'mean_rating']]

    # Merge into one DataFrame per character
    merged_df = pd.merge(system_df, audiobook_df, on='character', suffixes=('_System', '_Audiobook'))
    merged_df = pd.merge(merged_df, random_df, on='character')
    merged_df.rename(columns={'mean_rating': 'mean_Random'}, inplace=True)

    # Clean same_sex_narration
    merged_df['same_sex_narration'] = merged_df['same_sex_narration'].str.strip().str.title()

    # Define function to perform hypothesis tests
    def perform_tests(df, label):
        if df.empty or len(df) < 2:
            print(f"\n{label}: Not enough data (n={len(df)} characters)")
            return
        print(f"\n{label} (n={len(df)} characters):")
        # Calculate means and standard deviations
        avg_system = df['mean_rating_System'].mean()
        std_system = df['mean_rating_System'].std()
        avg_audiobook = df['mean_rating_Audiobook'].mean()
        std_audiobook = df['mean_rating_Audiobook'].std()
        avg_random = df['mean_Random'].mean()
        std_random = df['mean_Random'].std()
        print(f"Average mean rating for System: {avg_system:.2f} (STD: {std_system:.2f})")
        print(f"Average mean rating for Audiobook: {avg_audiobook:.2f} (STD: {std_audiobook:.2f})")
        print(f"Average mean rating for Random: {avg_random:.2f} (STD: {std_random:.2f})")
        
        # H1: System > Random (one-tailed test)
        t_stat_h1, p_value_h1 = ttest_rel(df['mean_rating_System'], df['mean_Random'], alternative='greater')
        print(f"H1: System > Random, t-stat={t_stat_h1:.3f}, p-value={p_value_h1:.3f}")
        
        # H2: Audiobook > System (one-tailed test, corrected from two-sided)
        t_stat_h2, p_value_h2 = ttest_rel(df['mean_rating_System'], df['mean_rating_Audiobook'], alternative='less')
        print(f"H2: Audiobook > System, t-stat={t_stat_h2:.3f}, p-value={p_value_h2:.3f}")
        
        # H3: System > Audiobook (one-tailed test)
        t_stat_h3, p_value_h3 = ttest_rel(df['mean_rating_System'], df['mean_rating_Audiobook'], alternative='greater')
        print(f"H3: System > Audiobook, t-stat={t_stat_h3:.3f}, p-value={p_value_h3:.3f}")

    # Define function to create bar plot
    def create_bar_plot(df, label, filename):

        # Skip empty or single-character datasets
        if df.empty or len(df) < 2:
            return  
        
        # Data for plotting
        groups = ['System', 'Audiobook', 'Random']
        means = [
            df['mean_rating_System'].mean(),
            df['mean_rating_Audiobook'].mean(),
            df['mean_Random'].mean()
        ]
        stds = [
            df['mean_rating_System'].std(),
            df['mean_rating_Audiobook'].std(),
            df['mean_Random'].std()
        ]

        # Create figure
        plt.figure(figsize=(8, 6))
        bars = plt.bar(groups, means, yerr=stds, capsize=5, color=['#1f77b4', '#ff7f0e', '#2ca02c'], edgecolor='black')
        
        # Customize plot
        plt.ylabel('Mean Rating', fontsize=12)
        plt.title(f'Mean Ratings by Audio Type ({label})', fontsize=14, pad=15)
        plt.ylim(0, 5.5)  # Assuming ratings are 1-5
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tick_params(axis='both', labelsize=10)
        
        # Add value labels on top of bars
        for i, bar in enumerate(bars):
            yval = bar.get_height()
            std = stds[i]

            # Place label above the std bar (mean + std + small offset)
            label_y = yval + std + 0.2
            plt.text(bar.get_x() + bar.get_width()/2, label_y, f'{yval:.2f}', 
                     ha='center', va='bottom', fontsize=15)
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Save plot
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

    # Perform tests and create plots
    perform_tests(merged_df, "Overall Analysis")
    create_bar_plot(merged_df, "Overall", "data/overall_ratings_plot.png")
    
    matching_df = merged_df[merged_df['same_sex_narration'] == 'Yes']
    perform_tests(matching_df, "Matching Sex Narration")
    create_bar_plot(matching_df, "Matching Sex Narration", "data/matching_sex_ratings_plot.png")

    non_matching_df = merged_df[merged_df['same_sex_narration'] == 'No']
    perform_tests(non_matching_df, "Non-Matching Sex Narration")
    create_bar_plot(non_matching_df, "Non-Matching Sex Narration", "data/non_matching_sex_ratings_plot.png")