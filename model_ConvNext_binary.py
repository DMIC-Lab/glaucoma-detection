# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset, random_split
from torchsummary import summary
from sklearn.metrics import accuracy_score
import numpy as np
import random
from dataset import GlaucomaDataset
import pandas as pd
import copy
import wandb
from torchmetrics.classification import BinaryAUROC, BinaryRecall, BinarySpecificity, BinarySensitivityAtSpecificity

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
num_epochs = 100
batch_size = 64
learning_rate = 1e-4

# %%
wandb.init(project='glaucoma-binary', config={
    'learning_rate': learning_rate,
    'epochs': num_epochs,
    'batch_size': batch_size,
})

# %%
def get_pretrained_convnext():
    model = models.convnext.convnext_base(weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1)
    # Modify the classifier for binary classification
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, 1)  # Binary classification output
    return model

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_pretrained_convnext().to(device)
model.load_state_dict(torch.load('./outputs/best_model_binary.pth',weights_only=True))
model = torch.compile(model)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

# %%
summary(model, input_size=(3, 512, 512))

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
positive_label = 'RG'  # Adjust if necessary
num_positive = len(df[df['Final Label'] == positive_label])
num_negative = len(df[df['Final Label'] != positive_label])

pos_weight_value = num_negative / num_positive
pos_weight = torch.tensor([pos_weight_value]).to(device)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
# %%
# Create datasets and dataloaders
dataset = GlaucomaDataset(path, df, transform)
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

# %%
best_val_loss = float('inf')
best_model_wts = copy.deepcopy(model.state_dict())
# Initialize metrics
train_AUROC = BinaryAUROC().to(device)
train_Recall = BinaryRecall().to(device)
train_Specificity = BinarySpecificity().to(device)
val_AUROC = BinaryAUROC().to(device)
val_Recall = BinaryRecall().to(device)
val_Specificity = BinarySpecificity().to(device)

# %%
# Training and validation loop
step = 0
for epoch in range(num_epochs):
    # Training phase
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    all_outputs = []
    all_labels = []
    for i,(images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.float().to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        all_outputs.append(outputs)
        all_labels.append(labels)
        preds = torch.sigmoid(outputs) >= 0.5
        correct_predictions += (preds == labels).sum().item()
        total_predictions += labels.size(0)
        step += 1
        if i % 50 == 0:
            wandb.log({
                'step': step,
                'train_step_loss': loss.item()
            })


    epoch_loss = running_loss / len(train_loader.dataset)
    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)
        # Compute training metrics
    epoch_accuracy = correct_predictions / total_predictions
    train_auroc = train_AUROC(all_outputs, all_labels)
    train_recall = train_Recall(all_outputs, all_labels)
    train_specificity = train_Specificity(all_outputs, all_labels)
    wandb.log({
        'epoch': epoch + 1,
        'train_loss': epoch_loss,
        'train_accuracy': epoch_accuracy,
        'train_auc': train_auroc,
        'train_recall': train_recall,
        'train_specificty': train_specificity
    })
    # Validation phase
    model.eval()
    val_running_loss = 0.0
    val_correct_predictions = 0
    val_total_predictions = 0
    val_all_outputs = []
    val_all_labels = []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.float().to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item() * images.size(0)
            val_all_outputs.append(outputs)
            val_all_labels.append(labels)
            preds = torch.sigmoid(outputs) >= 0.5
            val_correct_predictions += (preds == labels).sum().item()
            val_total_predictions += labels.size(0)

    val_loss = val_running_loss / len(val_loader.dataset)
    val_accuracy = val_correct_predictions / val_total_predictions
    val_all_outputs = torch.cat(val_all_outputs)
    val_all_labels = torch.cat(val_all_labels)
    val_auroc = val_AUROC(val_all_outputs, val_all_labels)
    val_recall = val_Recall(val_all_outputs, val_all_labels)
    val_specificity = val_Specificity(val_all_outputs, val_all_labels)
    # Log validation metrics
    wandb.log({
        'epoch': epoch + 1,
        'val_loss': val_loss,
        'val_accuracy': val_accuracy,
        'val_auc': val_auroc,
        'val_recall': val_recall,
        'val_specificity': val_specificity
        
    })
    print(f"Epoch [{epoch+1}/{num_epochs}], "
          f"Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_accuracy:.4f}, "
          f"Training AUROC: {train_auroc:.4f}, "
          f"Training Recall: {train_recall:.4f}, "
          f"Training Specificity: {train_specificity:.4f}, "
          f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, "
          f"Validation AUROC: {val_auroc:.4f}, "
          f"Validation Recall: {val_recall:.4f}, "
          f"Validation Specificity: {val_specificity:.4f} ")
    
        # Check if the validation loss has improved
    if val_loss < best_val_loss:
        print(f'Validation loss decreased ({best_val_loss:.4f} --> {val_loss:.4f}).  Saving model ...')
        best_val_loss = val_loss
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(model.state_dict(), './outputs/best_model_binary.pth')

# %%
wandb.finish()

# %%



