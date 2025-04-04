import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os
import wandb

from models import UNetResNet_low, UNetResNet

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

wandb.login(key="f4a726b2fe7929990149e82fb88da423cfa74e46")

class SegmentationTrainer:
    def __init__(self, dataset_path, batch_size, epochs, lr, step_size, gamma, n_classes):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.epochs = epochs
        self.dataset_path = dataset_path

        # Load dataset
        self.load_dataset()
        
        # Create DataLoaders
        self.train_loader = DataLoader(TensorDataset(self.train_images, self.train_masks), batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(TensorDataset(self.val_images, self.val_masks), batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(TensorDataset(self.test_images, self.test_masks), batch_size=batch_size, shuffle=False)
        
        # Initialize model, loss, optimizer, scheduler
        self.model = UNetResNet_low(n_classes=n_classes, pretrained=True).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        
        # Initialize WandB
        wandb.init(project="segmentation-trainer", config={
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": lr,
            "step_size": step_size,
            "gamma": gamma,
            "n_classes": n_classes
        })

    def load_dataset(self):
        data = torch.load(self.dataset_path)
        self.train_images, self.train_masks = data["train"]
        self.val_images, self.val_masks = data["val"]
        self.test_images, self.test_masks = data["test"]

        print(f"Number of training images: {self.train_images.shape[0]}")
        print(f"Number of validation images: {self.val_images.shape[0]}")
        print(f"Number of test images: {self.test_images.shape[0]}")
    
    def train(self):
        train_losses = []
        val_losses = []

        os.makedirs("checkpoints", exist_ok=True)

        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0

            for inputs, masks in self.train_loader:
                masks = masks.squeeze(1)
                inputs, masks = inputs.to(self.device), masks.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                outputs_resized = F.interpolate(outputs, size=masks.shape[1:], mode='bilinear', align_corners=False)
                loss = self.criterion(outputs_resized, masks)

                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            train_losses.append(running_loss / len(self.train_loader))

            # Perform validation in all epochs
            val_loss = self.validate()
            val_losses.append(val_loss)
            
            wandb.log({
                "epoch": epoch + 1,
                "training_loss": train_losses[-1],
                "validation_loss": val_loss
            })

            # Log training and validation losses
            print(f"Epoch [{epoch+1}/{self.epochs}], Training Loss: {train_losses[-1]:.4f}, Validation Loss: {val_loss:.4f}")
            
            # Save model checkpoint every 10 epochs
            if epoch % 10 == 0 or epoch == self.epochs - 1:
                checkpoint_path = f"checkpoints/model_epoch_{epoch+1}.pth"
                torch.save(self.model.state_dict(), checkpoint_path)
                print(f"Model checkpoint saved at {checkpoint_path}")

            self.scheduler.step()

        self.plot_losses(train_losses, val_losses)

    def validate(self):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, masks in self.val_loader:
                masks = masks.squeeze(1)
                inputs, masks = inputs.to(self.device), masks.to(self.device)
                outputs = self.model(inputs)
                outputs_resized = F.interpolate(outputs, size=masks.shape[1:], mode='bilinear', align_corners=False)
                loss = self.criterion(outputs_resized, masks)
                val_loss += loss.item()
        return val_loss / len(self.val_loader)

    def plot_losses(self, train_losses, val_losses):
        plt.figure(figsize=(10, 6))
        plt.yscale('log')
        plt.plot(train_losses, label='Training')
        plt.plot(val_losses, label='Validation')
        plt.xlabel("Epoch", fontsize=16)
        plt.ylabel("Loss", fontsize=16)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.legend(fontsize=20)
        plt.savefig("training_history.png", dpi=500, bbox_inches='tight')
        print("Training history saved as training_history.png")

if __name__ == "__main__":
    trainer = SegmentationTrainer(dataset_path="data/dataset.pt", batch_size=4, epochs=100, lr=1e-4, step_size=200, gamma=0.1, n_classes=3)
    trainer.train()
