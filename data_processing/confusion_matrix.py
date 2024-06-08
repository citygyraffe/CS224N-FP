import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
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

def main(task, true_file, predicted_file, output_file):
    if task == 'sst':
        true_labels, predicted_labels = load_sst_data(true_file, predicted_file)
    elif task == 'para':
        true_labels, predicted_labels = load_para_data(true_file, predicted_file)
    else:
        raise ValueError(f"Unknown task: {task}")

    # Generate the confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)

    # Display the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix for {task}')

    # Save the figure to a file if output_file is provided
    if output_file:
        plt.savefig(output_file)
        print(f'Confusion matrix saved to {output_file}')
    else:
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a confusion matrix from true and predicted CSV files for different tasks.')
    parser.add_argument('task', type=str, choices=['sst', 'para'], help='Task type: sst or para')
    parser.add_argument('--true_file', type=str, help='Path to the CSV file containing true labels', required=True)
    parser.add_argument('--predicted_file', type=str, help='Path to the CSV file containing predicted labels', required=True)
    parser.add_argument('--output_file', type=str, help='Path to save the output confusion matrix image', default=None)

    args = parser.parse_args()

    main(args.task, args.true_file, args.predicted_file, args.output_file)
