import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib  # To save and load the vectorizer

class ResumeFitModel(nn.Module):
    def __init__(self, input_dim):
        super(ResumeFitModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

def train_model(csv_path):
    print("Training model...")

    data = pd.read_csv(csv_path)
    
    # Ensure all columns are numeric
    data = data.apply(pd.to_numeric, errors='coerce')

    # Extract features and labels
    X = data[['Cosine Similarity Score']].values
    y = data['Label'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    input_dim = X_train.shape[1]
    model = ResumeFitModel(input_dim=input_dim)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    
    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    # Save the model and the vectorizer
    torch.save(model.state_dict(), "data/output/model_weights.pth")
    print("Model training complete and saved.")

def evaluate_model(csv_path):
    print("Evaluating model...")

    data = pd.read_csv(csv_path)
    
    # Ensure all columns are numeric
    data = data.apply(pd.to_numeric, errors='coerce')

    # Extract features and labels
    X = data[['Cosine Similarity Score']].values
    y = data['Label'].values
    
    model = ResumeFitModel(input_dim=X.shape[1])  # Ensure input_dim matches the feature size
    model.load_state_dict(torch.load("data/output/model_weights.pth"))
    
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    
    model.eval()
    with torch.no_grad():
        predictions = model(X_tensor).round()
    
    accuracy = accuracy_score(y_tensor, predictions)
    print(f"Model accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    csv_path = 'data/output/trained_data.csv'  # Update with the actual path
    train_model(csv_path)
    evaluate_model(csv_path)
