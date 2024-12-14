from datasets import load_dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

# Load datasets
dataset = load_dataset('csv', data_files={
    'train': 'train.csv',
    'validation': 'validation.csv'
})

# Map labels to integers
label_names = [
    "Least squares method",
    "Distance calculation",
    "Linear transformation",
    "Matrix multiplication",
    "Determinant calculation",
    "Inverse calculation",
    "Eigenvalue calculation",
    "Reflection"
]
label_to_id = {label: idx for idx, label in enumerate(label_names)}
id_to_label = {idx: label for label, idx in label_to_id.items()}

def encode_labels(example):
    example['label'] = label_to_id[example['label']]
    return example

dataset = dataset.map(encode_labels)

# Load tokenizer and model
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=len(label_names),
    id2label=id_to_label,
    label2id=label_to_id
)

# Tokenize the datasets
def tokenize_function(example):
    return tokenizer(example['text'], padding='max_length', truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Set the format for PyTorch
tokenized_datasets.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

# Initialize data collator
from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Define compute metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    return {'accuracy': accuracy, 'f1': f1}

# Set up training arguments
training_args = TrainingArguments(
    output_dir='./model',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir='./logs',
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Save the model and tokenizer
trainer.save_model('./model')
tokenizer.save_pretrained('./model')