import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms, models

# Definisci il percorso del modello fine-tuned e dell'immagine da analizzare
model_path = '/Users/digitalangels/Desktop/pythonProject_salienza/salicon_finetuned.pth'
screenshot_path = '/Users/digitalangels/Desktop/Screenshot 2024-10-16 at 11.43.55.png'

# Ricrea l'architettura del modello
class SaliencyModel(nn.Module):
    def __init__(self):
        super(SaliencyModel, self).__init__()
        self.backbone = models.resnet50(pretrained=False)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 224 * 224)

    def forward(self, x):
        x = self.backbone(x)
        return x.view(-1, 1, 224, 224)

# Inizializzazione del modello
model = SaliencyModel()

# Carica i pesi del modello fine-tuned
model.load_state_dict(torch.load(model_path))
model.eval()

# Definisci le trasformazioni per l'immagine da usare nel modello
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Per il modello, ridimensiona a 224x224
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Carica e trasforma l'immagine per il modello, mantenendo l'immagine originale
def load_image(image_path):
    image_original = Image.open(image_path).convert('RGB')  # Mantieni immagine originale
    image_for_model = transform(image_original).unsqueeze(0)  # Trasformazione per il modello
    return image_original, image_for_model

# Carica l'immagine
image_original, image_for_model = load_image(screenshot_path)

# Genera la mappa di salienza
def generate_saliency_map(image, model):
    with torch.no_grad():
        saliency_map = model(image)
    return saliency_map

# Genera la mappa di salienza dallo screenshot
saliency_map = generate_saliency_map(image_for_model, model)

# Converte la mappa di salienza in numpy array
saliency_map_np = saliency_map.squeeze().cpu().numpy()

# Ridimensiona la mappa di salienza alla risoluzione dell'immagine originale
saliency_map_tensor = torch.tensor(saliency_map_np).unsqueeze(0).unsqueeze(0)  # Aggiungi batch e canale
saliency_map_resized = F.interpolate(saliency_map_tensor,
                                     size=(image_original.height, image_original.width),
                                     mode='bilinear', align_corners=False).squeeze().numpy()

# Normalizza la mappa di salienza tra 0 e 1 per visualizzarla correttamente
saliency_map_resized = (saliency_map_resized - saliency_map_resized.min()) / (saliency_map_resized.max() - saliency_map_resized.min())

# Visualizza l'immagine originale e la mappa di salienza senza vincoli di colore
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Mostra l'immagine originale alla sua risoluzione originale
axs[0].imshow(image_original)
axs[0].set_title('Immagine originale')
axs[0].axis('off')

# Mostra la mappa di salienza con valori continui
im = axs[1].imshow(saliency_map_resized, cmap='jet', alpha=0.7)  # cmap 'jet' mostra i valori continui
axs[1].set_title('Mappa di Salienza')
axs[1].axis('off')

# Aggiungi la colorbar per mostrare la legenda dei valori
cbar = plt.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04)
cbar.set_label('Valori di salienza')

plt.tight_layout()
plt.show()
