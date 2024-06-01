'''
Multitask BERT class, starter training code, evaluation, and test code.

Of note are:
* class MultitaskBERT: Your implementation of multitask BERT.
* function train_multitask: Training procedure for MultitaskBERT. Starter code
    copies training procedure from `classifier.py` (single-task SST).
* function test_multitask: Test procedure for MultitaskBERT. This function generates
    the required files for submission.

Running `python multitask_classifier.py` trains and tests your MultitaskBERT and
writes all required submission files.
'''

import random, numpy as np, argparse
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bert import BertModel
from optimizer import AdamW
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import datetime
import shutil

from datasets import (
    SentenceClassificationDataset,
    SentenceClassificationTestDataset,
    SentencePairDataset,
    SentencePairTestDataset,
    load_multitask_data
)

# Custom imports
from task_scheduler import TaskScheduler

from evaluation import model_eval_sst, model_eval_multitask, model_eval_test_multitask

# ADDED INCLUDES
from tokenizer import BertTokenizer
from lora_layer import LoRALayer, LinearWithLoRALayer


TQDM_DISABLE=False

# CUSTOM SETTINGS
DEBUG_OUTPUT = False
USE_COMBINED_SST_DATASET = False
STS_TRAINING_SCALING_FACTOR = 1

# Fix the random seed.
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5

scaler = GradScaler()

# Summary Writer global object
writer = None

class MultitaskBERT(nn.Module):
    '''
    This module should use BERT for 3 tasks:

    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    '''
    def __init__(self, config):
        super(MultitaskBERT, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        # last-linear-layer mode does not require updating BERT paramters.
        assert config.fine_tune_mode in ["last-linear-layer", "full-model"]
        print(f"Fine-tune mode: {config.fine_tune_mode}")
        for param in self.bert.parameters():
            if config.fine_tune_mode == 'last-linear-layer':
                param.requires_grad = False
            elif config.fine_tune_mode == 'full-model':
                param.requires_grad = True

        # You will want to add layers here to perform the downstream tasks.
        ### TODO
        if DEBUG_OUTPUT:
            print(config)

        # Common Resources
        self.device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # Sentiment analysis
        self.sentiment_dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.sentiment_classifier = torch.nn.Linear(config.hidden_size, 5)

        # Paraphrase detection
        self.paraphrase_dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.paraphrase_classifier = torch.nn.Linear(config.hidden_size, 1)

        # Semantical similarity
        self.similarity_dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.similarity_classifier = torch.nn.Linear(config.hidden_size, 1)

         # Add a LoRA layer to linear layers in the self-attention components
        # by replacing the default linear laryer with a LinearWithLoRALayer
        # Disable gradient updates for BERT parameters if using LoRA so that
        # only LoRA parameters will be updated for better efficiency
        if config.lora:
            print("Using LoRA in BERT layers")
            # for param in self.bert.bert_layers.parameters():
            #     param.requires_grad = False # Freeze the original BERT layers
            for layer in self.bert.bert_layers:

                # Add lora layers to all of the self attention components
                # for param in layer.self_attention.parameters():
                #     param.requires_grad = False
                layer.self_attention.query = LinearWithLoRALayer(layer.self_attention.query, config.lora_rank, config.lora_alpha)
                layer.self_attention.key = LinearWithLoRALayer(layer.self_attention.key, config.lora_rank, config.lora_alpha)
                layer.self_attention.value = LinearWithLoRALayer(layer.self_attention.value, config.lora_rank, config.lora_alpha)

                # Add lora layers to feed forward components
                # for param in layer.interm_dense.parameters():
                #     param.requires_grad = False
                layer.interm_dense = LinearWithLoRALayer(layer.interm_dense, config.lora_rank, config.lora_alpha)

                # for param in layer.out_dense.parameters():
                #     param.requires_grad = False
                layer.out_dense = LinearWithLoRALayer(layer.out_dense, config.lora_rank, config.lora_alpha)

            self.sentiment_classifier = LinearWithLoRALayer(self.sentiment_classifier, config.lora_rank, config.lora_alpha)
            self.similarity_classifier = LinearWithLoRALayer(self.similarity_classifier, config.lora_rank, config.lora_alpha)
            self.paraphrase_classifier = LinearWithLoRALayer(self.paraphrase_classifier, config.lora_rank, config.lora_alpha)


    def forward(self, input_ids, attention_mask):
        'Takes a batch of sentences and produces embeddings for them.'
        # The final BERT embedding is the hidden state of [CLS] token (the first token)
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).

        # Retrieve BERT outputs
        outputs = self.bert(input_ids, attention_mask)
        # Fetch pooled output (pooled representation of each sentence)
        pooled_output = outputs['pooler_output']
        return pooled_output


    def predict_sentiment(self, input_ids, attention_mask):
        '''Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        '''
        ### TODO
        pooled_output = self.forward(input_ids, attention_mask)
        pooled_output = self.sentiment_dropout(pooled_output)
        logits = self.sentiment_classifier(pooled_output)
        return logits


    def predict_paraphrase(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation.
        '''
        ### TODO
        if DEBUG_OUTPUT:
            print("input_ids_1: ", input_ids_1)
            print("input_ids_2: ", input_ids_2)
            print("attention_mask_1: ", attention_mask_1)
            print("attention_mask_2: ", attention_mask_2)

        # PER BERT PAPER SEP TOKEN IS USED TO SEPERATE SENTENCES
        self.seperator_tokens = torch.full((input_ids_1.shape[0], 1), self.tokenizer.sep_token_id).to(self.device)

        if DEBUG_OUTPUT:
            print("seperator_tokens shape: ", self.seperator_tokens.shape)
            print("input_ids_1 shape: ", input_ids_1.shape)
            print("input_ids_2 shape: ", input_ids_2.shape)

        # ADD SEP TOKENS TO INPUTS
        inputs = torch.cat((input_ids_1, self.seperator_tokens, input_ids_2), dim=1)

        # ADD ATTENTION MASKS
        attentions = torch.cat((attention_mask_1, torch.ones_like(self.seperator_tokens), attention_mask_2), dim=1)

        pooled_output = self.forward(inputs, attentions)
        pooled_output = self.paraphrase_dropout(pooled_output)
        logit = self.paraphrase_classifier(pooled_output)
        return logit


    def predict_similarity(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Note that your output should be unnormalized (a logit).
        '''
        ### TODO
        if DEBUG_OUTPUT:
            print("input_ids_1: ", input_ids_1)
            print("input_ids_2: ", input_ids_2)
            print("attention_mask_1: ", attention_mask_1)
            print("attention_mask_2: ", attention_mask_2)

        # PER BERT PAPER SEP TOKEN IS USED TO SEPERATE SENTENCES
        self.seperator_tokens = torch.full((input_ids_1.shape[0], 1), self.tokenizer.sep_token_id).to(self.device)

        if DEBUG_OUTPUT:
            print("seperator_tokens shape: ", self.seperator_tokens.shape)
            print("input_ids_1 shape: ", input_ids_1.shape)
            print("input_ids_2 shape: ", input_ids_2.shape)

        # ADD SEP TOKENS TO INPUTS
        inputs = torch.cat((input_ids_1, self.seperator_tokens, input_ids_2), dim=1)

        # ADD ATTENTION MASKS
        attentions = torch.cat((attention_mask_1, torch.ones_like(self.seperator_tokens), attention_mask_2), dim=1)

        pooled_output = self.forward(inputs, attentions)
        pooled_output = self.similarity_dropout(pooled_output)
        logit = self.similarity_classifier(pooled_output)
        return logit

def save_model(model, optimizer, args, config, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")

class BatchProcessor:
    def __init__(self, model, optimizer, args, config):
        self.device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        self.model = model
        self.optimizer = optimizer
        self.args = args
        self.config = config
        self.scaler = GradScaler()

    def process_batch(self, task, batch):
        self.optimizer.zero_grad()

        if task == "sst":
            b_ids, b_mask, b_labels = (batch['token_ids'].to(self.device),
                                       batch['attention_mask'].to(self.device),
                                       batch['labels'].to(self.device))

            with autocast():
                logits = self.model.predict_sentiment(b_ids, b_mask)
                loss = F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / self.args.batch_size

        elif task == "para":
            b_ids1, b_mask1, b_ids2, b_mask2, b_labels = (
                batch['token_ids_1'].to(self.device),
                batch['attention_mask_1'].to(self.device),
                batch['token_ids_2'].to(self.device),
                batch['attention_mask_2'].to(self.device),
                batch['labels'].to(self.device)
            )

            with autocast():
                logits = self.model.predict_paraphrase(b_ids1, b_mask1, b_ids2, b_mask2)
                loss = F.binary_cross_entropy_with_logits(logits.view(-1), b_labels.view(-1).float(), reduction='sum') / self.args.batch_size

        elif task == "sts":
            for _ in range(STS_TRAINING_SCALING_FACTOR):
                b_ids1, b_mask1, b_ids2, b_mask2, b_labels = (
                    batch['token_ids_1'].to(self.device),
                    batch['attention_mask_1'].to(self.device),
                    batch['token_ids_2'].to(self.device),
                    batch['attention_mask_2'].to(self.device),
                    batch['labels'].to(self.device)
                )

                with autocast():
                    logits = self.model.predict_similarity(b_ids1, b_mask1, b_ids2, b_mask2).squeeze()
                    loss = F.mse_loss(logits, b_labels.view(-1).float(), reduction='sum') / self.args.batch_size

        else:
            raise ValueError(f"train_multitask::Unknown task: {task}")

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return loss.item()



def train_multitask(args):
    '''Train MultitaskBERT.

    Currently only trains on SST dataset. The way you incorporate training examples
    from other datasets into the training procedure is up to you. To begin, take a
    look at test_multitask below to see how you can use the custom torch `Dataset`s
    in datasets.py to load in examples from the Quora and SemEval datasets.
    '''
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    # Create the data and its corresponding datasets and dataloader.
    sst_train_data, num_labels,para_train_data, sts_train_data = load_multitask_data(args.sst_train,args.para_train,args.sts_train, split ='train')
    sst_dev_data, num_labels,para_dev_data, sts_dev_data = load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev, split ='train')

    # Sentiment data:
    sst_train_data = SentenceClassificationDataset(sst_train_data, args)
    sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

    sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sst_train_data.collate_fn, pin_memory=True)
    sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sst_dev_data.collate_fn, pin_memory=True)

    # Paraphrase data:
    para_train_data = SentencePairDataset(para_train_data, args)
    para_dev_data = SentencePairDataset(para_dev_data, args)

    para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size,
                                        collate_fn=para_train_data.collate_fn)
    para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=para_dev_data.collate_fn)

    # STS data:
    sts_train_data = SentencePairDataset(sts_train_data, args)
    sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression=True)

    sts_train_dataloader = DataLoader(sts_train_data, shuffle=True, batch_size=args.batch_size,
                                        collate_fn=sts_train_data.collate_fn)
    sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sts_dev_data.collate_fn)


    # A list to hold all the training data loaders
    training_data_loaders = {}
    training_data_loaders["sst"] = sst_train_dataloader
    training_data_loaders["para"] = para_train_dataloader
    training_data_loaders["sts"] = sts_train_dataloader

    # Init model.
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'num_labels': num_labels,
              'hidden_size': 768,
              'data_dir': '.',
              'fine_tune_mode': args.fine_tune_mode,
              'lora': args.lora,
              'lora_rank': args.lora_rank,
              'lora_alpha': args.lora_alpha}

    config = SimpleNamespace(**config)

    model = MultitaskBERT(config)
    model = model.to(device)

    print(model)

    # Get number of trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of trainable parameters: {trainable_params}")
    print(f"Total number of parameters: {total_params}")

    # for name, param in model.named_parameters():
    #     print(f"{name}: {param.requires_grad}")

    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr)
    best_dev_acc = 0

    # Create a task scheduler to determine which tasks to train on
    scheduler = TaskScheduler(args, ["sst", "para", "sts"], [sst_train_dataloader.__len__(), para_train_dataloader.__len__(), sts_train_dataloader.__len__()])

    # Create a batch processor to process the batches
    batch_processor = BatchProcessor(model, optimizer, args, config)

    # Run for the specified number of epochs.
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        step_loss = 0
        num_batches = 0

        scheduler.step_epoch()

        if args.scheduling_mode == 'epoch':
            print("Using epoch scheduling mode")
            task = scheduler.get_next_task()
            scheduler.print_task_distribution()
            data_loader = training_data_loaders[task]

            for batch in tqdm(data_loader, desc=f'{task}-train-{epoch}', disable=TQDM_DISABLE):
                step_loss = batch_processor.process_batch(task, batch)
                train_loss += step_loss
                num_batches += 1
        elif args.scheduling_mode == 'batch':
            assert args.num_batches_per_epoch > 0, "Number of batches per epoch must be greater than 0"
            print("Using batch scheduling mode")
            num_batches = args.num_batches_per_epoch
            for batch in tqdm(range(args.num_batches_per_epoch), desc=f'train-{epoch}', disable=TQDM_DISABLE):
                task = scheduler.get_next_task()
                step_loss = batch_processor.process_batch(task, next(iter(training_data_loaders[task])))
                train_loss += step_loss
            scheduler.print_task_distribution()

        train_loss = train_loss / (num_batches)
        writer.add_scalar('loss/train', train_loss, epoch)
        writer.add_scalar('loss/step', step_loss, epoch)

        if (epoch + 1) % args.eval_epochs == 0:
            dev_sentiment_accuracy, _, _, \
            dev_paraphrase_accuracy, _, _, \
            dev_sts_corr, _, _ = model_eval_multitask(sst_dev_dataloader,
                                                    para_dev_dataloader,
                                                    sts_dev_dataloader, model, device)

            total_accuracy = (dev_sentiment_accuracy + dev_paraphrase_accuracy + (0.5 + (0.5 * dev_sts_corr)))/3
            writer.add_scalar('accuracy/sst', dev_sentiment_accuracy, epoch)
            writer.add_scalar('accuracy/para', dev_paraphrase_accuracy, epoch)
            writer.add_scalar('accuracy/sts', dev_sts_corr, epoch)
            writer.add_scalar('accuracy/total', total_accuracy, epoch)

            if total_accuracy > best_dev_acc:
                best_dev_acc = total_accuracy
                save_model(model, optimizer, args, config, args.filepath)
                print("New Best Model!")

            print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, dev acc sentiment:: {dev_sentiment_accuracy :.3f}, dev acc paraphrase :: {dev_paraphrase_accuracy :.3f}, dev acc sts :: {dev_sts_corr :.3f},")

    return total_params, trainable_params

def test_multitask(args):
    '''Test and save predictions on the dev and test sets of all three tasks.'''
    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        saved = torch.load(args.filepath)
        config = saved['model_config']

        model = MultitaskBERT(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"Loaded model to test from {args.filepath}")

        sst_test_data, num_labels,para_test_data, sts_test_data = \
            load_multitask_data(args.sst_test,args.para_test, args.sts_test, split='test')

        sst_dev_data, num_labels,para_dev_data, sts_dev_data = \
            load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev,split='dev')

        sst_test_data = SentenceClassificationTestDataset(sst_test_data, args)
        sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

        sst_test_dataloader = DataLoader(sst_test_data, shuffle=True, batch_size=args.batch_size,
                                         collate_fn=sst_test_data.collate_fn)
        sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sst_dev_data.collate_fn)

        para_test_data = SentencePairTestDataset(para_test_data, args)
        para_dev_data = SentencePairDataset(para_dev_data, args)

        para_test_dataloader = DataLoader(para_test_data, shuffle=True, batch_size=args.batch_size,
                                          collate_fn=para_test_data.collate_fn)
        para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                         collate_fn=para_dev_data.collate_fn)

        sts_test_data = SentencePairTestDataset(sts_test_data, args)
        sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression=True)

        sts_test_dataloader = DataLoader(sts_test_data, shuffle=True, batch_size=args.batch_size,
                                         collate_fn=sts_test_data.collate_fn)
        sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sts_dev_data.collate_fn)

        dev_sentiment_accuracy,dev_sst_y_pred, dev_sst_sent_ids, \
            dev_paraphrase_accuracy, dev_para_y_pred, dev_para_sent_ids, \
            dev_sts_corr, dev_sts_y_pred, dev_sts_sent_ids = model_eval_multitask(sst_dev_dataloader,
                                                                    para_dev_dataloader,
                                                                    sts_dev_dataloader, model, device)

        test_sst_y_pred, \
            test_sst_sent_ids, test_para_y_pred, test_para_sent_ids, test_sts_y_pred, test_sts_sent_ids = \
                model_eval_test_multitask(sst_test_dataloader,
                                          para_test_dataloader,
                                          sts_test_dataloader, model, device)

        with open(args.sst_dev_out, "w+") as f:
            print(f"dev sentiment acc :: {dev_sentiment_accuracy :.3f}")
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(dev_sst_sent_ids, dev_sst_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sst_test_out, "w+") as f:
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(test_sst_sent_ids, test_sst_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.para_dev_out, "w+") as f:
            print(f"dev paraphrase acc :: {dev_paraphrase_accuracy :.3f}")
            f.write(f"id \t Predicted_Is_Paraphrase \n")
            for p, s in zip(dev_para_sent_ids, dev_para_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.para_test_out, "w+") as f:
            f.write(f"id \t Predicted_Is_Paraphrase \n")
            for p, s in zip(test_para_sent_ids, test_para_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sts_dev_out, "w+") as f:
            print(f"dev sts corr :: {dev_sts_corr :.3f}")
            f.write(f"id \t Predicted_Similiary \n")
            for p, s in zip(dev_sts_sent_ids, dev_sts_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sts_test_out, "w+") as f:
            f.write(f"id \t Predicted_Similiary \n")
            for p, s in zip(test_sts_sent_ids, test_sts_y_pred):
                f.write(f"{p} , {s} \n")

        return (dev_sentiment_accuracy,dev_sst_y_pred, dev_sst_sent_ids, \
            dev_paraphrase_accuracy, dev_para_y_pred, dev_para_sent_ids, \
            dev_sts_corr, dev_sts_y_pred, dev_sts_sent_ids)

def get_args():
    parser = argparse.ArgumentParser()

    if USE_COMBINED_SST_DATASET:
        parser.add_argument("--sst_train", type=str, default="data/ids-sentiment-combined-train.csv")
    else:
        parser.add_argument("--sst_train", type=str, default="data/ids-sst-train.csv")

    # Cant use combined dataset for dev and test
    parser.add_argument("--sst_dev", type=str, default="data/ids-sst-dev.csv")
    parser.add_argument("--sst_test", type=str, default="data/ids-sst-test-student.csv")

    parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")

    parser.add_argument("--sts_train", type=str, default="data/sts-train.csv")
    parser.add_argument("--sts_dev", type=str, default="data/sts-dev.csv")
    parser.add_argument("--sts_test", type=str, default="data/sts-test-student.csv")

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--fine-tune-mode", type=str,
                        help='last-linear-layer: the BERT parameters are frozen and the task specific head parameters are updated; full-model: BERT parameters are updated as well',
                        choices=('last-linear-layer', 'full-model'), default="full-model")
    parser.add_argument("--use_gpu", action='store_true')

    parser.add_argument("--sst_dev_out", type=str, default="predictions/sst-dev-output.csv")
    parser.add_argument("--sst_test_out", type=str, default="predictions/sst-test-output.csv")

    parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

    parser.add_argument("--sts_dev_out", type=str, default="predictions/sts-dev-output.csv")
    parser.add_argument("--sts_test_out", type=str, default="predictions/sts-test-output.csv")

    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)

    # Custom arguments
    parser.add_argument("--testOnly", action='store_true', help="Only run test on existing model")
    parser.add_argument("--model_path", type=str, help="Path to the model to load for testing")
    parser.add_argument("--force_task", type=str, choices=('', 'sst', 'para', 'sts'), default="", help="Force the task to train on (sst, para, sts)")
    parser.add_argument("--scheduling_policy", type=str, choices=('random', 'round-robin', 'annealed-sampling'), default="round-robin", help="Scheduling policy for task selection")
    parser.add_argument("--scheduling_mode", type=str, choices=('epoch', 'batch'), default='epoch', help="Scheduling mode for task selection (single task per epoch or per batch)")
    parser.add_argument("--num_batches_per_epoch", type=int, default=0, help="Number of batches per epoch (used by batch scheduling mode)")
    parser.add_argument("--eval_epochs", type=int, default=1, help="Run evaluation on dev set every eval_epochs")

    # Arguments for LoRA finetuning
    parser.add_argument("--lora", action='store_true', default=False, help="Use LoRA for training")
    parser.add_argument("--lora_rank", type=int, default=0, help="Rank of LoRA matrix. 0 to skip LoRA")
    parser.add_argument("--lora_alpha", type=float, default=2.0, help="Alpha value for LoRA")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)  # Fix the seed for reproducibility.
    dateStr = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    loraStr = f'lora_r{args.lora_rank}' if args.lora else "no-lora"
    outputPathName = f'logs/{dateStr}-{args.fine_tune_mode}-{args.epochs}-{args.lr}-{args.scheduling_policy}-{args.batch_size}-{loraStr}'
    args.filepath = f'{args.fine_tune_mode}-{args.epochs}-{args.lr}-multitask.pt' # Save path.
    args.outputPathName = outputPathName

    # Values to save in results.txt
    training_time = 0
    testing_time = 0
    total_params = "N/A"
    training_params = "N/A"

    # Write all args to args.txt
    layout = {
        "multitask": {
            "loss": ["Multiline", ["loss/train", "loss/step"]],
            "accuracy": ["Multiline", ["accuracy/sst", "accuracy/para", "accuracy/sts", "accuracy/total"]],
        },
    }
    writer = SummaryWriter(outputPathName)
    writer.add_custom_scalars(layout)

    # Save all arguments to a file
    with open(f'{outputPathName}/args.txt', 'w') as f:
        for arg in vars(args):
            f.write(f"{arg}: {getattr(args, arg)}\n")


    if(not args.testOnly):
        # Start timer to measure training time
        start_time = datetime.datetime.now()
        total_params, training_params = train_multitask(args)
        end_time = datetime.datetime.now()
        training_time = end_time - start_time
        print(f"Training time: {training_time}")
    else:
        print("Skipping training and running test only!")

    # Start timer to measure testing time
    start_time = datetime.datetime.now()
    # Test the model
    dev_sentiment_accuracy,_, _, \
            dev_paraphrase_accuracy, _, _, \
            dev_sts_corr, _, _ = test_multitask(args)
    end_time = datetime.datetime.now()
    testing_time = end_time - start_time

    # Write results to a file
    with open(f'{outputPathName}/results.txt', 'w') as f:
        f.write(f"dev sentiment acc :: {dev_sentiment_accuracy :.3f}\n")
        f.write(f"dev paraphrase acc :: {dev_paraphrase_accuracy :.3f}\n")
        f.write(f"dev sts corr :: {dev_sts_corr :.3f}\n")
        f.write(f"Training time: {training_time}\n")
        f.write(f"Testing time: {testing_time}\n")
        f.write(f"Total parameters: {total_params}\n")
        f.write(f"Tunable parameters: {training_params}\n")

    # Copy predictions output and model to the output directory
    try:
        shutil.copytree('predictions', f'{outputPathName}/predictions')
        shutil.copyfile(args.filepath, f'{outputPathName}/model.pt')
    except:
        print("Error copying model or predictions to output directory")

    writer.close()
