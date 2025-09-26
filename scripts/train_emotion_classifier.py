import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Paths
FEATURES_FILE = "../Extracted Features/combined_wav2vec_features.csv"
OUTPUT_MODEL_DIR = "../models/emotion_classifier/"
RESULTS_DIR = "../results/"
os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Hyperparameters
EPOCHS = 75
LEARNING_RATE = 0.00005
BATCH_SIZE = 32
PATIENCE = 10
WEIGHT_DECAY = 2e-5
N_SPLITS = 5

# Dataset Class
class EmotionDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Model Definition
class EmotionClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, nhead=4, num_layers=4, dim_feedforward=512):
        super(EmotionClassifier, self).__init__()
        self.input_projection = nn.Linear(input_dim, 256)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.3,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.input_projection(x)
        x = x.unsqueeze(1)
        x = self.transformer_encoder(x)
        x = x.squeeze(1)
        return self.classifier(x)

# Load Data
logging.info("Loading features for training and testing...")
df = pd.read_csv(FEATURES_FILE)
df = df.drop(columns=['Dataset'], errors='ignore')
df.dropna(inplace=True)

X = df.iloc[:, :-1].values  # Features (768 dimensions)
y = pd.factorize(df['label'])[0]
label_names = pd.factorize(df['label'])[1]

# Compute smoothed class weights
class_counts = np.bincount(y)
total_samples = len(y)
class_weights = total_samples / (len(class_counts) * class_counts)
class_weights = np.clip(class_weights, a_min=0.8, a_max=5.0)
class_weights = torch.tensor(class_weights, dtype=torch.float)

# Split Data into Train and Test Sets (70/30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
test_dataset = EmotionDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Cross-Validation Setup
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
fold_accuracies = []
all_train_losses = []
all_val_losses = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
    logging.info(f"\nTraining Fold {fold+1}/{N_SPLITS}...")

    X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
    y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

    train_dataset = EmotionDataset(X_fold_train, y_fold_train)
    val_dataset = EmotionDataset(X_fold_val, y_fold_val)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Training Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmotionClassifier(input_dim=X.shape[1], num_classes=len(set(y))).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)

    # Training Loop with Early Stopping
    train_losses, val_losses, val_accuracies = [], [], []
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_path = os.path.join(OUTPUT_MODEL_DIR, f"best_emotion_classifier_transformer_fold{fold+1}.pth")

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        y_train_true, y_train_pred = [], []

        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            y_train_true.extend(labels.cpu().numpy())
            y_train_pred.extend(preds.cpu().numpy())

        train_losses.append(epoch_loss / len(train_loader))
        scheduler.step()

        # Validation
        model.eval()
        val_loss, y_val_true, y_val_pred = 0, [], []
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                val_loss += criterion(outputs, labels).item()
                preds = torch.argmax(outputs, dim=1)
                y_val_true.extend(labels.cpu().numpy())
                y_val_pred.extend(preds.cpu().numpy())

        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(accuracy_score(y_val_true, y_val_pred))
        logging.info(f"Fold {fold+1}, Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, Val Accuracy: {val_accuracies[-1]:.4f}")

        # Early stopping logic
        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"Saved best model for fold {fold+1} with Val Loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logging.info(f"Early stopping triggered after epoch {epoch+1} for fold {fold+1}")
                break

    # Store losses for this fold
    all_train_losses.append(train_losses)
    all_val_losses.append(val_losses)

    # Load the best model for this fold and evaluate
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    y_val_true, y_val_pred = [], []
    with torch.no_grad():
        for features, labels in val_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            preds = torch.argmax(outputs, dim=1)
            y_val_true.extend(labels.cpu().numpy())
            y_val_pred.extend(preds.cpu().numpy())

    val_accuracy = accuracy_score(y_val_true, y_val_pred)
    fold_accuracies.append(val_accuracy)
    logging.info(f"Fold {fold+1} Validation Accuracy: {val_accuracy:.4f}")

# Average cross-validation accuracy
avg_cv_accuracy = np.mean(fold_accuracies)
logging.info(f"\nAverage Cross-Validation Accuracy: {avg_cv_accuracy:.4f}")

# Compute average train and val losses across folds
max_epochs = max(len(losses) for losses in all_train_losses)  # Get the maximum number of epochs across folds
avg_train_losses = np.zeros(max_epochs)
avg_val_losses = np.zeros(max_epochs)
train_counts = np.zeros(max_epochs)  # To track how many folds contribute to each epoch
val_counts = np.zeros(max_epochs)

for fold in range(N_SPLITS):
    for epoch in range(len(all_train_losses[fold])):
        avg_train_losses[epoch] += all_train_losses[fold][epoch]
        train_counts[epoch] += 1
    for epoch in range(len(all_val_losses[fold])):
        avg_val_losses[epoch] += all_val_losses[fold][epoch]
        val_counts[epoch] += 1

# Compute the averages
avg_train_losses = avg_train_losses / train_counts
avg_val_losses = avg_val_losses / val_counts

# Plot Average Training and Validation Losses
plt.figure(figsize=(10, 5))
plt.plot(range(1, max_epochs + 1), avg_train_losses, label="Average Train Loss", color='blue')
plt.plot(range(1, max_epochs + 1), avg_val_losses, label="Average Val Loss", color='orange', linestyle='--')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Average Training and Validation Loss Across Folds")
plt.legend()
plt.grid()
plt.savefig(os.path.join(RESULTS_DIR, "avg_train_val_loss_plot_transformer.png"))
plt.show()

# Train the final model on the entire training set
logging.info("\nTraining final model on the entire training set...")
train_dataset = EmotionDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

model = EmotionClassifier(input_dim=X.shape[1], num_classes=len(set(y))).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)

train_losses = []
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    train_losses.append(epoch_loss / len(train_loader))
    scheduler.step()
    logging.info(f"Final Training Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_losses[-1]:.4f}")

# Save the final model
final_model_path = os.path.join(OUTPUT_MODEL_DIR, "final_emotion_classifier_transformer.pth")
torch.save(model.state_dict(), final_model_path)
logging.info("Final model saved.")

# Testing Phase
model.eval()
y_true, y_pred = [], []
logging.info("Testing the final model...")
with torch.no_grad():
    for features, labels in test_loader:
        features, labels = features.to(device), labels.to(device)
        outputs = model(features)
        preds = torch.argmax(outputs, dim=1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# Compute Metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
report = classification_report(y_true, y_pred, target_names=label_names)

logging.info("\n--- Testing Results ---")
logging.info(f"Accuracy: {accuracy * 100:.2f}%")
logging.info(report)

# Save Metrics
metrics = {
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1-Score": f1
}
metrics_df = pd.DataFrame([metrics])
metrics_df.to_csv(os.path.join(RESULTS_DIR, "test_metrics_transformer.csv"), index=False)

# Save Predictions
predictions_df = pd.DataFrame({
    "True Label": [label_names[i] for i in y_true],
    "Predicted Label": [label_names[i] for i in y_pred]
})
predictions_df.to_csv(os.path.join(RESULTS_DIR, "test_predictions_transformer.csv"), index=False)

# Confusion Matrix Plot
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_names, yticklabels=label_names)
plt.title("Confusion Matrix (Transformer)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix_transformer.png"))
plt.show()

# Bar Plot for Precision, Recall, and F1-Score
report_dict = classification_report(y_true, y_pred, target_names=label_names, output_dict=True)
metrics_df = pd.DataFrame(report_dict).transpose().iloc[:-3, :]
metrics_df[['precision', 'recall', 'f1-score']].plot(kind='bar', figsize=(12, 6))
plt.title("Class-wise Precision, Recall, and F1-Score (Transformer)")
plt.ylabel("Score")
plt.xlabel("Classes")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "class_metrics_transformer.png"))
plt.show()

# Pie Chart for Accuracy
plt.figure(figsize=(6, 6))
plt.pie([accuracy, 1 - accuracy], labels=["Correct Predictions", "Incorrect Predictions"], autopct='%1.1f%%', startangle=140)
plt.title("Overall Accuracy (Transformer)")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "accuracy_pie_chart_transformer.png"))
plt.show()

# Final Training Loss Plot (optional, since we already have the combined plot)
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(train_losses) + 1), train_losses, label="Final Train Loss", color='blue')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Final Training Loss (Transformer)")
plt.legend()
plt.grid()
plt.savefig(os.path.join(RESULTS_DIR, "final_training_loss_plot_transformer.png"))
plt.show()

logging.info(f"Test metrics, predictions, and plots saved to {RESULTS_DIR}")