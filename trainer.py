import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os
import wandb
from torchvision import transforms
import random

from models import UNetResNet_low, UNetResNet

wandb.login(key="f4a726b2fe7929990149e82fb88da423cfa74e46")

class SegmentationTransform:
    def __init__(self):
        self.image_transforms = transforms.Compose([
            transforms.ColorJitter(brightness=0.4, contrast=0.3, saturation=0.3, hue=0.2),  # Change light/contrast
        ])
        self.shared_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),  # Mirroring
            transforms.RandomRotation(degrees=10),   # Slight rotations
            transforms.RandomResizedCrop(size=(128, 128), scale=(0.7, 1.0))  # Slight crops
        ])

    def __call__(self, image, mask):
        # Apply ColorJitter only to the image
        image = self.image_transforms(image)

        # Apply shared transformations to both image and mask
        seed = random.randint(0, 2**32)
        random.seed(seed)
        torch.manual_seed(seed)
        image = self.shared_transforms(image)

        random.seed(seed)
        torch.manual_seed(seed)
        mask = self.shared_transforms(mask)

        return image, mask

class SegmentationTrainer:
    def __init__(self, dataset_path, batch_size, epochs, lr, step_size, gamma, n_classes):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.epochs = epochs
        self.dataset_path = dataset_path

        # Load dataset
        self.load_dataset()
        
        # Create DataLoaders
        segmentation_transform = SegmentationTransform()

        def collate_fn(batch):
            images, masks = zip(*[segmentation_transform(image, mask) for image, mask in batch])
            return torch.stack(images), torch.stack(masks)

        self.train_loader = DataLoader(
            TensorDataset(self.train_images, self.train_masks),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn
        )
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

        best_val_loss = float('inf')

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

            # Save model checkpoint if validation loss improves
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = f"checkpoints/best_{epoch}_{val_loss}.pth"
                torch.save(self.model.state_dict(), checkpoint_path)
                print(f"Best model checkpoint saved at {checkpoint_path} with validation loss: {best_val_loss:.4f}")

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
    trainer = SegmentationTrainer(dataset_path="data/dataset_2.pt", batch_size=4, epochs=1000, lr=5e-4, step_size=500, gamma=0.5, n_classes=3)
    trainer.train()
