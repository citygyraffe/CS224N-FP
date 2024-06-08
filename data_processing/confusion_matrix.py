import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import argparse

def main(true_file, predicted_file, output_file):
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


    # Generate the confusion matrix
    cm = confusion_matrix(true_sentiments, predicted_sentiments)

    # Display the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')

    # Save the figure to a file if output_file is provided
    if output_file:
        plt.savefig(output_file)
        print(f'Confusion matrix saved to {output_file}')
    else:
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a confusion matrix from true and predicted sentiment CSV files.')
    parser.add_argument('--true_file', type=str, help='Path to the CSV file containing true sentiments')
    parser.add_argument('--predicted_file', type=str, help='Path to the CSV file containing predicted sentiments')
    parser.add_argument('--output_file', type=str, help='Path to save the output confusion matrix image', default=None)

    args = parser.parse_args()

    main(args.true_file, args.predicted_file, args.output_file)
