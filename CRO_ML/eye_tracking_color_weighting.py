import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

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
# Aggiungi la dimensione batch e la dimensione dei canali per farla funzionare con interpolate
saliency_map_tensor = torch.tensor(saliency_map_np).unsqueeze(0).unsqueeze(0)  # Aggiungi batch e canale
saliency_map_resized = F.interpolate(saliency_map_tensor,
                                     size=(image_original.height, image_original.width),
                                     mode='bilinear', align_corners=False).squeeze().numpy()

# Evidenzia le zone in base alle regole specifiche:
highlighted_map = np.zeros((*saliency_map_resized.shape, 3))  # Crea un array RGB vuoto

# Zone morte (<= 0.25) evidenziate in nero
highlighted_map[saliency_map_resized <= 0.25] = [0, 0, 0]  # Nero

# Zone con valori tra 0.26 e 0.39 evidenziate in giallo
highlighted_map[(saliency_map_resized > 0.26) & (saliency_map_resized <= 0.39)] = [1, 1, 0]  # Giallo

# Zone con valori > 0.39 evidenziate in rosso
highlighted_map[saliency_map_resized > 0.39] = [1, 0, 0]  # Rosso

# Visualizza l'immagine originale e la mappa di salienza con evidenziazioni
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Mostra l'immagine originale alla sua risoluzione originale
axs[0].imshow(image_original)
axs[0].set_title('Immagine originale')
axs[0].axis('off')

# Mostra la mappa di salienza con zone evidenziate
axs[1].imshow(highlighted_map)
axs[1].set_title('Eye Tracking')
axs[1].axis('off')

plt.tight_layout()
plt.show()
