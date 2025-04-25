from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

def get_data_loader(train=True):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train" if train else "test")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Add a padding token if not already defined
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # Add padding token
        tokenizer.model_max_length = 512  # Ensure max length is set

    def tokenize_function(examples):
        # Tokenize the text and create labels by shifting input tokens
        tokenized = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
        tokenized["labels"] = tokenized["input_ids"].copy()  # Labels are the same as input_ids

        # Ensure input_ids are within the valid range of the vocabulary
        vocab_size = tokenizer.vocab_size
        tokenized["input_ids"] = [
            [token if token < vocab_size else tokenizer.pad_token_id for token in seq]
            for seq in tokenized["input_ids"]
        ]
        return tokenized

    # Update the model's embedding layer to account for the new padding token
    tokenizer.model_max_length = 512
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return DataLoader(tokenized_dataset, batch_size=8, shuffle=train)
