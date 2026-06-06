import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_data=datasets.MNIST(root="data", train=True,  download=True, transform=transform)
test_data=datasets.MNIST(root="data", train=False, download=True, transform=transform)

train_loader=DataLoader(train_data, batch_size=64, shuffle=True)
test_loader=DataLoader(test_data,  batch_size=64, shuffle=False)

print("Data loaded!")

class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
    def forward(self, x):
        return self.model(x)

model=ImageClassifier()
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(), lr=0.001)

print("Model ready!")

epochs=5
train_losses=[]

for epoch in range(epochs):
    total_loss=0
    for images, labels in train_loader:
        optimizer.zero_grad()
        output=model(images)
        loss=criterion(output, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"Epoch {epoch+1}/{epochs}  Loss: {avg_loss:.4f}")

correct=0
total=0

with torch.no_grad():
    for images, labels in test_loader:
        output=model(images)
        predicted=output.argmax(dim=1)
        correct+= (predicted == labels).sum().item()
        total+= labels.size(0)

accuracy = 100 * correct / total
print(f"\nTest Accuracy: {accuracy:.2f}%")


plt.figure(figsize=(8,4))
plt.plot(range(1, epochs+1), train_losses, marker="o", color="blue")
plt.title("Training Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("training_loss.png")
plt.show()
print("Saved training_loss.png")

fig, axes=plt.subplots(2, 5, figsize=(12, 5))
images, labels=next(iter(test_loader))

with torch.no_grad():
    preds = model(images).argmax(dim=1)

for i, ax in enumerate(axes.flat):
    ax.imshow(images[i].squeeze(), cmap="gray")
    color = "green" if preds[i] == labels[i] else "red"
    ax.set_title(f"Pred: {preds[i].item()}  True: {labels[i].item()}", color=color)
    ax.axis("off")

plt.suptitle("Sample Predictions (Green=Correct, Red=Wrong)")
plt.savefig("predictions.png")
plt.show()
print("Saved predictions.png")