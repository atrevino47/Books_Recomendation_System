import torch
from transformers import DistilBertTokenizer, DistilBertModel
from torch import nn

# Load the tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model_path = './bert/distiloberto_512t.bin'
bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')

class SentimentClassifier(nn.Module):
    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = bert_model
        self.pre_classifier = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        self.dropout = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        # Obtain the last hidden states from the DistilBert model
        last_hidden_state = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )[0]  # Take the first element of the tuple which is the last_hidden_state

        # Apply a linear layer and dropout
        pooled_output = self.pre_classifier(last_hidden_state[:, 0])  # Use the first token's embeddings for pooling
        pooled_output = nn.ReLU()(pooled_output)  # Use a ReLU activation function
        pooled_output = self.dropout(pooled_output)
        # Pass the output of the dropout layer to the output layer
        return self.out(pooled_output)

# Assuming we have 2 classes (positive and negative)
model = SentimentClassifier(n_classes=2)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Function to preprocess the text
def preprocess(text):
    max_len = 512
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    return encoding['input_ids'], encoding['attention_mask']

# Prediction function
def predict_sentiment(text):
    input_ids, attention_mask = preprocess(text)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        # The output is logits, we need to take softmax to get probabilities
        probs = torch.nn.functional.softmax(outputs, dim=1)
        # Taking the class with the highest probability
        prediction = torch.argmax(probs, dim=1)
        return prediction.item()



while True:
# Example usage:
    input_text = input("Enter a sentence: ")
    prediction = predict_sentiment(input_text)
    classnames = ["positive", "negative"]
    print(f'Text: {input_text}')
    print(f'Sentiment: {classnames[prediction]}')


