import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class DepressionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': float(self.labels[idx])
        }

def load_data(data_path='data/enhanced_depression_dataset_with_emojis.csv', test_size=0.2, val_size=0.1):
    df = pd.read_csv(data_path)
    texts = df['emoji_version'].astype(str).tolist()
    labels = df['severity_label'].tolist()

    X_temp, X_test, y_temp, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=42, stratify=labels)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size/(1 - test_size), random_state=42, stratify=y_temp)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
