# CS 224N Default Final Project - Multitask BERT

This is the default final project for the Stanford CS 224N class. Please refer to the project handout on the course website for detailed instructions and an overview of the codebase.

This project comprises two parts. In the first part, you will implement some important components of the BERT model to better understand its architecture. 
In the second part, you will use the embeddings produced by your BERT model on three downstream tasks: sentiment classification, paraphrase detection, and semantic similarity. You will implement extensions to improve your model's performance on the three downstream tasks.

In broad strokes, Part 1 of this project targets:
* bert.py: Missing code blocks.
* classifier.py: Missing code blocks.
* optimizer.py: Missing code blocks.

And Part 2 targets:
* multitask_classifier.py: Missing code blocks.
* datasets.py: Possibly useful functions/classes for extensions.
* evaluation.py: Possibly useful functions/classes for extensions.

## Setup instructions

Follow `setup.sh` to properly setup a conda environment and install dependencies.


## Baseline implementation for multitask minBERT

Baseline configuration
* 30 epochs of fine-tuning with last linear layer or full model training mode
* Random selection of dataset (between SST, Quora, STS) for each epoch
* Batch size of 16 for SST and Quora, 8 for STS
* Learning rate of 1e-5

Below is the performance for each dataset's dev evaluation:
* SST (acc): 0.322 (LLL), 0.503 (full)
* Quora (acc): 0.688 (LLL), 0.881 (full)
* STS (correlation): 0.320 (LLL), 0.855 (full)

To run the job for fine-tuning and evaluating, use commands:
* Full model training mode
`python3 multitask_classifier.py --use_gpu --epochs 30 --batch_size 16 --fine-tune-mode full-model`
* Last linear layer training mode
`python3 multitask_classifier.py --use_gpu --epochs 30 --batch_size 16 --fine-tune-mode last-linear-layer`


## Acknowledgement

The BERT implementation part of the project was adapted from the "minbert" assignment developed at Carnegie Mellon University's [CS11-711 Advanced NLP](http://phontron.com/class/anlp2021/index.html),
created by Shuyan Zhou, Zhengbao Jiang, Ritam Dutt, Brendon Boldt, Aditya Veerubhotla, and Graham Neubig.

Parts of the code are from the [`transformers`](https://github.com/huggingface/transformers) library ([Apache License 2.0](./LICENSE)).
