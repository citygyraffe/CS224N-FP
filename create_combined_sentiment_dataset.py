import os
import csv

def combine_test_datasets(sst_dataset_filepath, cfimdb_dataset_filepath):
    ids = {}
    sentiment_data = []
    counter = 1
    
    with open(sst_dataset_filepath, 'r') as fp:
        for record in csv.DictReader(fp,delimiter = '\t'):
            sent = record['sentence']
            sent_id = record['id']

            if sent_id in ids:
                print(f"Duplicate id found: {sent_id} skipping...")
                continue
        
            sentiment_data.append((counter, float(counter -1), sent_id, sent))
            counter += 1
    
    with open(cfimdb_dataset_filepath, 'r') as fp:
        for record in csv.DictReader(fp,delimiter = '\t'):
            sent = record['sentence']
            sent_id = record['id']
            
            if sent_id in ids:
                print(f"Duplicate id found: {sent_id} skipping...")
                continue
        
            sentiment_data.append((counter, float(counter -1), sent_id, sent))
            counter += 1
    
    return sentiment_data

def combine_dev_train_datasets(sst_dataset_filepath, cfimdb_dataset_filepath):
    ids = {}
    sentiment_data = []
    counter = 0

    # we want to map cfimdb from 0-1 to 0-4 to match SST dataset
    cfimdb_label_scale = 4

    with open(sst_dataset_filepath, 'r') as fp:
        for record in csv.DictReader(fp,delimiter = '\t'):
            sent = record['sentence']
            sent_id = record['id']
            label = record['sentiment']

            if sent_id in ids:
                print(f"Duplicate id found: {sent_id} skipping...")
                continue

            sentiment_data.append((counter, sent_id, sent, label))
            counter += 1
            
    with open(cfimdb_dataset_filepath, 'r') as fp:
        for record in csv.DictReader(fp,delimiter = '\t'):
            sent = record['sentence']
            sent_id = record['id']
            label = record['sentiment']
            
            if sent_id in ids:
                print(f"Duplicate id found: {sent_id} skipping...")
                continue
        
            sentiment_data.append((counter, sent_id, sent, int(label)*cfimdb_label_scale))
            counter += 1
    
    return sentiment_data


def combine_sst_cfimdb_datasets(sst_dataset_filepath, cfimdb_dataset_filepath, split):
    sentiment_data = []
    
    if split == 'test-student':
        sentiment_data = combine_test_datasets(sst_dataset_filepath, cfimdb_dataset_filepath)
    else:
        sentiment_data = combine_dev_train_datasets(sst_dataset_filepath, cfimdb_dataset_filepath)

    filename = f'data/ids-sentiment-combined-{split}.csv'

    if os.path.exists(os.path.join(os.getcwd(),filename)):
        print(f"File {filename} already exists. Skipping...")
        return

    with open(filename, 'w') as fp:
        writer = csv.writer(fp, delimiter = '\t')
        
        if split == 'test-student':
            writer.writerow(['0', '', 'id', 'sentence'])
        else:
            writer.writerow(['', 'id', 'sentence', 'sentiment'])
        
        writer.writerows(sentiment_data)

def main():

    print("Combining sentiment dev datasets...")
    sst_dataset_filepath = 'data/ids-sst-dev.csv'
    cfimdb_dataset_filepath = 'data/ids-cfimdb-dev.csv'
    combine_sst_cfimdb_datasets(sst_dataset_filepath, cfimdb_dataset_filepath, 'dev')

    print("Combining sentiment train datasets...")
    sst_dataset_filepath = 'data/ids-sst-train.csv'
    cfimdb_dataset_filepath = 'data/ids-cfimdb-train.csv'
    combine_sst_cfimdb_datasets(sst_dataset_filepath, cfimdb_dataset_filepath, 'train')

    print("Combining sentiment test-student datasets...")
    sst_dataset_filepath = 'data/ids-sst-test-student.csv'
    cfimdb_dataset_filepath = 'data/ids-cfimdb-test-student.csv'
    combine_sst_cfimdb_datasets(sst_dataset_filepath, cfimdb_dataset_filepath, 'test-student')
    

if __name__ == "__main__":
    print("Running combined sentiment dataset creation...")
    main()
