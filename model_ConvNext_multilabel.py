# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset,random_split
from torchsummary import summary
from sklearn.metrics import accuracy_score
import numpy as np
import random
from dataset import GlaucomaDataset
import pandas as pd
import copy
import wandb
import torchmetrics
from torchmetrics.classification import MultilabelHammingDistance, MultilabelAUROC, MultilabelSensitivityAtSpecificity
import pickle
# %%
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    np.random.seed(seed)
    random.seed(seed)

# %%
set_seed(42)

# %%
num_epochs = 30
batch_size = 64
learning_rate = 1e-3
num_labels = 10

# %%
wandb.init(project='glaucoma-multilabel', config={
    'learning_rate': learning_rate,
    'epochs': num_epochs,
    'batch_size': batch_size,
})

class ConvNeXtWithAdditionalFeatures(nn.Module):
    def __init__(self, num_additional_features):
        super(ConvNeXtWithAdditionalFeatures, self).__init__()
        # Load the pre-trained ConvNeXt model
        self.convnext = models.convnext.convnext_base(weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1)
        # Replace the last classification layer with an identity layer to extract features
        num_features = self.convnext.classifier[2].in_features
        self.convnext.classifier[2] = nn.Identity()
        # Define a new classifier that takes concatenated features
        self.classifier = nn.Linear(num_features + num_additional_features, 10)  # Multilabel classification

    def forward(self, x, additional_features):
        # Pass input through ConvNeXt
        x = self.convnext(x)
        # Ensure additional_features has the correct shape
        if len(additional_features.shape) == 1:
            additional_features = additional_features.unsqueeze(1)
        # Concatenate ConvNeXt features with additional features
        x = torch.cat((x, additional_features), dim=1)
        # Pass concatenated features through the classifier
        x = self.classifier(x)
        #x = torch.sigmoid(x)
        return x

# Example usage:
def get_pretrained_convnext_with_additional_features(num_additional_features):
    model = ConvNeXtWithAdditionalFeatures(num_additional_features)
    return model

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_pretrained_convnext_with_additional_features(5).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

# %%
#summary(model, input_size=(3, 512, 512))

# %%
#print(model)

# %%
path =  "./preprocessed"
df = pd.read_csv("./final_labels.csv")
transform = transforms.Compose([
    transforms.AutoAugment(),
    transforms.ConvertImageDtype(torch.float32),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# %%
# Create datasets and dataloaders
dataset = GlaucomaDataset(path, df, additionalFeatures=pickle.load(open('./cup.pkl','rb')),transform=transform, mode='multi')
print(len(dataset))
# Calculate the sizes for each split
total_size = len(dataset)
train_size = int(0.7 * total_size)
val_size = int(0.2 * total_size)
test_size = total_size - train_size - val_size
# Split the dataset
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=16)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,num_workers=16)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False,num_workers=16)

criterion = nn.BCEWithLogitsLoss(pos_weight=dataset.getWeights().to(device))  
# %%
best_val_loss = float('inf')
best_model_wts = copy.deepcopy(model.state_dict())
# Initialize metrics
auroc = MultilabelAUROC(num_labels=num_labels).to(device)
sensitivity = MultilabelSensitivityAtSpecificity(num_labels=num_labels,min_specificity=.95).to(device)
hammingdistance = MultilabelHammingDistance(num_labels=num_labels).to(device)
# %%
# Training and validation loop
step = 0
for epoch in range(num_epochs):
    # Training phase
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    for i,(images, labels, additionalFeatures) in enumerate(train_loader):
        images, labels, additionalFeatures = images.to(device), labels.float().to(device), additionalFeatures.to(device)

        optimizer.zero_grad()
        outputs = model(images,additionalFeatures)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = torch.sigmoid(outputs) >= 0.5
        correct_predictions += (preds == labels).sum().item()
        total_predictions += labels.size(0)
        step += 1
        if i % 50 == 0:
            wandb.log({
                'step': step,
                'train_loss': loss.item()
            })
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_accuracy = correct_predictions / total_predictions
    wandb.log({
        'epoch': epoch + 1,
        'train_loss': epoch_loss,
        'train_accuracy': epoch_accuracy
    })

    # Validation phase
    model.eval()
    val_running_loss = 0.0
    val_correct_predictions = 0
    val_total_predictions = 0
    all_outputs = []
    all_labels =[]
    with torch.no_grad():
        for images, labels,additionalFeatures in val_loader:
            images, labels,additionalFeatures = images.to(device), labels.float().to(device), additionalFeatures.to(device)
            all_labels.append(labels)
            outputs = model(images,additionalFeatures)
            all_outputs.append(outputs)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item() * images.size(0)
            preds = torch.sigmoid(outputs) >= 0.5
            val_correct_predictions += (preds == labels).sum().item()
            val_total_predictions += labels.size(0)

    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)
    val_loss = val_running_loss / len(val_loader.dataset)
    val_accuracy = val_correct_predictions / val_total_predictions
    val_hamming = hammingdistance(all_outputs,all_labels.int())
    val_sensitivity = sensitivity(all_outputs,all_labels.int())
    val_auroc = auroc(all_outputs,all_labels.int())

    # Log validation metrics
    wandb.log({
        'epoch': epoch + 1,
        'val_loss': val_loss,
        'val_accuracy': val_accuracy,
        'val_hamming':val_hamming,
        'val_auroc': val_auroc
    })
    print(f"Epoch [{epoch+1}/{num_epochs}], ",f"Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_accuracy:.4f}, ",f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, ",f"Validation Hamming: {val_hamming}, ",f"Validation Sensitivity: {val_sensitivity}, ",f"Validation AUROC: {val_auroc}")
    
        # Check if the validation loss has improved
    if val_loss < best_val_loss:
        print(f'Validation loss decreased ({best_val_loss:.4f} --> {val_loss:.4f}).  Saving model ...')
        best_val_loss = val_loss
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(model.state_dict(), './outputs/best_model_multi.pth')

# %%
wandb.finish()

# %%



