import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import argparse

def load_sst_data(true_file, predicted_file):
    # Load true sentiments
    true_sentiments_df = pd.read_csv(true_file, delimiter='\t', quoting=3)
    true_sentiments = true_sentiments_df['sentiment'].values

    # Load predicted sentiments
    with open(predicted_file, 'r') as file:
        lines = file.readlines()
        predicted_sentiments = []
        for line in lines[1:]:  # Skip header
            parts = line.split(',')
            if len(parts) > 1:
                predicted_sentiment = parts[-1].strip()
                predicted_sentiments.append(predicted_sentiment)
            else:
                print(f"Skipping line: {line}")

    predicted_sentiments = [int(x) for x in predicted_sentiments]
    return true_sentiments, predicted_sentiments

def load_para_data(true_file, predicted_file):
    # Load true is_duplicate values
    true_data_df = pd.read_csv(true_file, delimiter='\t')
    true_is_duplicate = true_data_df['is_duplicate'].values

    # Load predicted is_duplicate values
    with open(predicted_file, 'r') as file:
        lines = file.readlines()
        predicted_is_duplicate = []
        for line in lines[1:]:  # Skip header
            parts = line.split(',')
            if len(parts) > 1:
                is_duplicate = parts[-1].strip()
                predicted_is_duplicate.append(is_duplicate)
            else:
                print(f"Skipping line: {line}")

    predicted_is_duplicate = [float(x) for x in predicted_is_duplicate]
    return true_is_duplicate, predicted_is_duplicate

def load_sts_data(true_file, predicted_file):
    # Load true similarity values
    true_data_df = pd.read_csv(true_file, delimiter='\t')
    true_similarity = true_data_df['similarity'].values

    # Load predicted similarity values
    with open(predicted_file, 'r') as file:
        lines = file.readlines()
        predicted_similarity = []
        for line in lines[1:]:  # Skip header
            parts = line.split(',')
            if len(parts) > 1:
                similarity = parts[-1].strip()
                predicted_similarity.append(similarity)
            else:
                print(f"Skipping line: {line}")

    predicted_similarity = [float(x) for x in predicted_similarity]
    return true_similarity, predicted_similarity

def evaluate_continuous(true_values, predicted_values, output_file):
    mse = mean_squared_error(true_values, predicted_values)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_values, predicted_values)
    r2 = r2_score(true_values, predicted_values)

    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"R² Score: {r2}")

    # Residual plot
    residuals = true_values - predicted_values
    plt.scatter(predicted_values, residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot for STS Task')

    # Save the plot to a file if output_file is provided
    if output_file:
        plt.savefig(output_file)
        print(f'Residual plot saved to {output_file}')
    else:
        plt.show()

def main(task, true_file, predicted_file, output_file):
    if task == 'sst':
        true_labels, predicted_labels = load_sst_data(true_file, predicted_file)
        # Generate the confusion matrix
        cm = confusion_matrix(true_labels, predicted_labels)
        # Display the confusion matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix for SST Task')
    elif task == 'para':
        true_labels, predicted_labels = load_para_data(true_file, predicted_file)
        # Generate the confusion matrix
        cm = confusion_matrix(true_labels, predicted_labels)
        # Display the confusion matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix for Paraphrase Task')
    elif task == 'sts':
        true_labels, predicted_labels = load_sts_data(true_file, predicted_file)
        evaluate_continuous(true_labels, predicted_labels, output_file)
        return
    else:
        raise ValueError(f"Unknown task: {task}")

    # Save the figure to a file if output_file is provided
    if output_file:
        plt.savefig(output_file)
        print(f'Confusion matrix saved to {output_file}')
    else:
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a confusion matrix from true and predicted CSV files for different tasks.')
    parser.add_argument('task', type=str, choices=['sst', 'para', 'sts'], help='Task type: sst, para, or sts')
    parser.add_argument('--true_file', type=str, help='Path to the CSV file containing true labels', required=True)
    parser.add_argument('--predicted_file', type=str, help='Path to the CSV file containing predicted labels', required=True)
    parser.add_argument('--output_file', type=str, help='Path to save the output image', default=None)

    args = parser.parse_args()

    main(args.task, args.true_file, args.predicted_file, args.output_file)
