import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from data_loader import DepressionDataset, load_data
from model import DepressionRegressor
from sklearn.metrics import mean_absolute_error

def train_model():
    # Use corrected dataset path in load_data
    (X_train, y_train), (X_val, y_val), _ = load_data('data/enhanced_depression_dataset_with_emojis.csv')

    tokenizer = AutoTokenizer.from_pretrained("google/muril-base-cased")

    train_dataset = DepressionDataset(X_train, y_train, tokenizer)
    val_dataset = DepressionDataset(X_val, y_val, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DepressionRegressor().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = torch.nn.MSELoss()

    model.train()
    for epoch in range(5):  # You can change number of epochs
        total_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device).float()

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}")

    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device).float()
            outputs = model(input_ids, attention_mask)
            preds.extend(outputs.cpu().numpy())
            targets.extend(labels.cpu().numpy())

    mae = mean_absolute_error(targets, preds)
    print(f"Validation MAE: {mae:.3f}")

    torch.save(model.state_dict(), "src/depression_regressor.pth")
    print("✅ Model saved as 'depression_regressor.pth'")

if __name__ == "__main__":
    train_model()
