from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import pandas as pd
import evaluate
from datasets import Dataset

# Step 1: Load the dataset
data = {
    "job_description": [
        "Looking for a Senior Software Engineer",
        "We need a Digital Marketing Expert",
        "Hiring Sales Manager with 5+ years",
        "Frontend Developer role available",
        "Marketing Assistant needed"
    ],
    "category": [0, 1, 2, 0, 1]  # 0 = Engineering, 1 = Marketing, 2 = Sales
}

df = pd.DataFrame(data)

# Step 2: Split the data into training and test sets
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['job_description'], df['category'], test_size=0.3)

# Step 3: Load a pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

# Step 4: Tokenize the data
train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=64)
test_encodings = tokenizer(list(test_texts), truncation=True, padding=True, max_length=64)

# Convert to Hugging Face Dataset format
train_dataset = Dataset.from_dict({'input_ids': train_encodings['input_ids'], 'attention_mask': train_encodings['attention_mask'], 'labels': list(train_labels)})
test_dataset = Dataset.from_dict({'input_ids': test_encodings['input_ids'], 'attention_mask': test_encodings['attention_mask'], 'labels': list(test_labels)})

# Step 5: Define a function to compute accuracy using the `evaluate` library
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# Step 6: Set up training arguments
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # number of epochs
    per_device_train_batch_size=8,   # batch size
    per_device_eval_batch_size=16,   # evaluation batch size
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    evaluation_strategy="epoch",     # evaluation during training
    save_strategy="epoch",           # save the model at every epoch
    logging_dir='./logs',            # directory for storing logs
)

# Step 7: Create a Trainer object for fine-tuning the model
trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=test_dataset,           # evaluation dataset
    compute_metrics=compute_metrics      # function to compute accuracy
)

# Step 8: Fine-tune the model
trainer.train()

# Step 9: Save the fine-tuned model
model.save_pretrained("./fine_tuned_job_model")
tokenizer.save_pretrained("./fine_tuned_job_model")

# Step 10: Evaluate the model on the test dataset
results = trainer.evaluate()

print(f"Test Accuracy: {results['eval_accuracy']:.4f}")


# Step 11: Load the fine-tuned model for querying
model = BertForSequenceClassification.from_pretrained('./fine_tuned_job_model')
tokenizer = BertTokenizer.from_pretrained('./fine_tuned_job_model')

# Step 12: Query the model with a new job description
job_description = "Looking for a Senior Software Engineer"

# Tokenize the input
inputs = tokenizer(job_description, return_tensors="pt", truncation=True, padding=True)

# Predict the category
outputs = model(**inputs)
predictions = outputs.logits.argmax(dim=-1)

# Map predictions back to categories
categories = {0: "Engineering", 1: "Marketing", 2: "Sales"}
predicted_category = categories[predictions.item()]
print(f"Predicted category: {predicted_category}")
