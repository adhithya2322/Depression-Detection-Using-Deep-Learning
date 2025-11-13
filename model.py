import torch
import torch.nn as nn
from transformers import AutoModel

class DepressionRegressor(nn.Module):
    def __init__(self, model_name="google/muril-base-cased"):
        super().__init__()
        self.muril = AutoModel.from_pretrained(model_name)
        self.regressor = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Output between 0 and 1
        )
    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.muril(input_ids=input_ids, attention_mask=attention_mask)
        cls_embed = outputs.last_hidden_state[:, 0]  # [CLS] token embedding
        return self.regressor(cls_embed).squeeze() * 4.0  # Scale to 0-4
