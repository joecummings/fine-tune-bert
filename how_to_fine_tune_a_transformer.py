"""How to fine-tune BERT."""
from datasets import load_dataset, load_metric
import numpy as np
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    Trainer,
    TrainingArguments,
)

RANDOM_SEED = 42


def main():
    emo_dataset = load_dataset("emotion")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def tokenize_emotion(example):
        """Tokenize text from emotion dataset."""
        return tokenizer(example["text"], padding="max_length", truncation=True)

    tokenized_data = emo_dataset.map(tokenize_emotion, batched=True)

    training_size = 1000
    validation_size = 100

    small_train_dataset = (
        tokenized_data["train"].shuffle(seed=RANDOM_SEED).select(range(training_size))
    )
    small_val_dataset = (
        tokenized_data["validation"]
        .shuffle(seed=RANDOM_SEED)
        .select(range(validation_size))
    )

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=8
    )

    training_args = TrainingArguments(
        "bert_trainer",
        evaluation_strategy="epoch",
        num_train_epochs=5,
    )

    metric = load_metric("f1")

    def metrics(eval_pred):
        """Compute micro F1 score."""
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(
            predictions=predictions, references=labels, average="micro"
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_val_dataset,
        compute_metrics=metrics,
    )

    trainer.train()


if __name__ == "__main__":
    main()
