import tensorflow as tf
import csv
import argparse
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Set up argument parsing
parser = argparse.ArgumentParser(description='Extract scalar data from TensorBoard tfevents files.')
parser.add_argument('--tfevents_path', type=str, help='Path to the tfevents file.')
parser.add_argument('--output_tag', type=str, default='', help='Extra tag to append to the output filename.')
args = parser.parse_args()

# Fail if tf events file is not provided
if not args.tfevents_path:
    parser.error("Please provide the path to the tfevents file.")

# Initialize an EventAccumulator object
event_acc = EventAccumulator(args.tfevents_path)
event_acc.Reload()  # Load the data from the file

# Get scalar data
scalars = event_acc.Tags()['scalars']

# Iterate over all scalars and create a CSV file for each
for tag in scalars:
    # Extract scalar events for the given tag
    scalar_events = event_acc.Scalars(tag)

    # Define CSV file name with optional extra tag
    csv_file_name = f"{tag.replace('/', '_')}_{args.output_tag}.csv"

    # Open a CSV file for writing
    with open(csv_file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write header
        writer.writerow(['Step', 'Wall time', 'Value'])

        # Write data rows
        for event in scalar_events:
            writer.writerow([event.step, event.wall_time, event.value])

    print(f"Data for tag '{tag}' has been written to {csv_file_name}")