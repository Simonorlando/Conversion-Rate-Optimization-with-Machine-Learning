import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from PIL import Image

# Configura il dispositivo (GPU se disponibile)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path per la cartella di screenshot (modifica con il tuo percorso)
image_dir = '/Users/digitalangels/Desktop/screen'

# Preprocessing: Ridimensiona e normalizza le immagini
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Ridimensiona a 224x224
    transforms.ToTensor(),  # Converti in tensore
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # Normalizzazione standard per modelli pre-addestrati
])

# Trasformazione separata per la mappa di salienza (scala di grigi)
saliency_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),  # Converte in scala di grigi con 1 canale
    transforms.ToTensor()
])


# Creazione del DataLoader per caricare le immagini dagli screenshot
class ScreenshotDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, transform=None, saliency_transform=None):
        self.image_dir = image_dir
        self.image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if
                            fname.endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform
        self.saliency_transform = saliency_transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')

        # Converte l'immagine target in scala di grigi per la mappa di salienza
        target = image.convert('L')  # Converti in scala di grigi

        if self.transform:
            image = self.transform(image)  # Applica trasformazione per l'immagine RGB
        if self.saliency_transform:
            target = self.saliency_transform(target)  # Applica trasformazione per mappa di salienza

        return image, target  # Input immagine RGB, target immagine scala di grigi


# Carica il dataset di screenshot
dataset = ScreenshotDataset(image_dir, transform=transform, saliency_transform=saliency_transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)


# Modello base pre-addestrato su un dataset generico di salienza
class SaliencyModel(nn.Module):
    def __init__(self):
        super(SaliencyModel, self).__init__()
        # Usare ResNet50 pre-addestrato come backbone
        self.backbone = models.resnet50(pretrained=True)
        # Modifica l'ultimo livello per output singolo (mappa di salienza)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 224 * 224)  # Produce un output di dimensioni 224x224

    def forward(self, x):
        x = self.backbone(x)
        return x.view(-1, 1, 224, 224)  # Ridimensiona per ottenere (batch_size, 1, 224, 224)


# Inizializzazione del modello
model = SaliencyModel().to(device)

# Funzione di perdita (MSE tra output e immagine target in scala di grigi)
criterion = nn.MSELoss()

# Ottimizzatore
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Funzione di training
def train_model(model, dataloader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Azzeriamo i gradienti
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass e ottimizzazione
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")


# Addestramento del modello
train_model(model, dataloader, criterion, optimizer, num_epochs=10)

# Salvare il modello fine-tuned
torch.save(model.state_dict(), 'salicon_finetuned.pth')
print("Modello fine-tuned salvato.")
