import shap
import torch
from transformers import AutoTokenizer
from model import DepressionRegressor

def explain_prediction(text):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = DepressionRegressor()
    model.load_state_dict(torch.load('depression_regressor.pth', map_location=device))
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("google/muril-base-cased")

    def f(x):
        inputs = tokenizer(x.tolist(), padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            outputs = model(inputs['input_ids'].to(device), inputs['attention_mask'].to(device))
        return outputs.cpu().numpy().flatten()

    explainer = shap.Explainer(f, tokenizer)
    shap_values = explainer([text])

    # Generate SHAP text plot HTML (without displaying)
    html_plot = shap.plots.text(shap_values[0], display=False)

    # Save the HTML content to file
    with open("shap_explanation.html", "w", encoding="utf-8") as f:
        f.write(html_plot)

    print("SHAP explanation saved to shap_explanation.html")

if __name__ == "__main__":
    example_text = "I feel very lonely and sad... 😞"
    explain_prediction(example_text)
