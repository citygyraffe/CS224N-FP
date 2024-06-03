import pandas as pd
import matplotlib.pyplot as plt
import argparse

def load_and_plot_data(file_paths, x_label, y_label, title, legend_labels, use_time, outfile='training_metrics.png'):
    # Setting up the plot - can be customized as needed
    plt.figure(figsize=(10, 5))

    for file_path, label in zip(file_paths, legend_labels):
        # Load the data from CSV file
        df = pd.read_csv(file_path)

        if use_time:
            # Convert Wall time to a readable format
            df['Wall time'] = pd.to_datetime(df['Wall time'], unit='s')
            # Calculate relative time in seconds from the first timestamp
            start_time = df['Wall time'].min()
            df['Relative Time'] = (df['Wall time'] - start_time).dt.total_seconds()
            x_data = df['Relative Time']
            if x_label == 'Training Step':
                x_label = 'Time (seconds since start)'
        else:
            x_data = df['Step']

        # Plotting
        plt.plot(x_data, df['Value'], marker='o', linestyle='-', label=label)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Save the plot to a file
    plt.savefig(outfile, format='png', dpi=300)
    # Show the plot
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Plot TensorBoard metrics from one or more CSV files.")
    parser.add_argument('--files', nargs='+', help="Paths to the CSV files containing the metrics data.")
    parser.add_argument('--x_label', type=str, default='Training Step', help="Label for the X-axis.")
    parser.add_argument('--y_label', type=str, default='Metric Value', help="Label for the Y-axis.")
    parser.add_argument('--title', type=str, default='TensorBoard Metrics Over Training Steps', help="Title of the plot.")
    parser.add_argument('--legends', type=str, nargs='+', help="Custom legends for each file. Number of legends must match the number of files.")
    parser.add_argument('--use_time', action='store_true', help="Use 'Time (Seconds)' as the X-axis instead of 'Step'.")
    parser.add_argument('--outfile', type=str, default='training_metrics.png', help="Output file name for the plot.")

    args = parser.parse_args()

    # Check if the number of legends matches the number of files
    if args.legends and len(args.legends) != len(args.files):
        parser.error("The number of legend labels must match the number of files.")

    # Set default X-axis label based on mode
    if args.use_time and args.x_label == 'Training Step':
        args.x_label = 'Time (Seconds)'

    # Load and plot data from the specified files
    load_and_plot_data(args.files, args.x_label, args.y_label, args.title, args.legends if args.legends else args.files, args.use_time, outfile=args.outfile)

if __name__ == "__main__":
    main()