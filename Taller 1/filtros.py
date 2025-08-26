import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

# Cargar y transformar la imagen
def load_image(image_path):
    transform = transforms.Compose([
        transforms.Grayscale(),  # Convertir a escala de grises
        transforms.ToTensor()    # Convertir a tensor
    ])
    image = Image.open(image_path)
    image = transform(image)
    image = image.unsqueeze(0)  # Añadir un batch dimension
    return image



# Definir filtros de Prewitt
prewitt_x = torch.tensor([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
prewitt_y = torch.tensor([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
print(prewitt_x.shape)

# Aplicar el filtro de convolución
def apply_prewitt_filter(image, kernel):
    return F.conv2d(image, kernel, padding=1)

# Cargar imagen
image_path = 'fig4.jpg'
image = load_image(image_path)
print(image.shape)

# Aplicar filtros de Prewitt
edges_x = apply_prewitt_filter(image, prewitt_x)
edges_y = apply_prewitt_filter(image, prewitt_y)

# Calcular la magnitud de los bordes
edges = torch.sqrt(edges_x ** 2 + edges_y ** 2)

# Visualizar resultados
plt.figure(figsize=(12, 4))
plt.subplot(1, 4, 1)
plt.title('Imagen Original')
plt.imshow(image.squeeze(), cmap='gray')

plt.subplot(1, 4, 3)
plt.title('Filtro Prewitt Horizontal')
plt.imshow(edges_x.squeeze().detach().numpy(), cmap='gray')

plt.subplot(1, 4, 4)
plt.title('Filtro Prewitt Vertical')
plt.imshow(edges_y.squeeze().detach().numpy(), cmap='gray')

plt.subplot(1, 4, 2)
plt.title('Modulo de los gradientes')
plt.imshow(edges.squeeze().detach().numpy(), cmap='gray')

plt.show()

#%%
# Definir filtros de Sobel
sobel_x = torch.tensor(
    [[-1, 0, 1],
     [-2, 0, 2],
     [-1, 0, 1]],
    dtype=torch.float32).unsqueeze(0).unsqueeze(0)
sobel_y = torch.tensor(
    [[-1, -2, -1],
     [0, 0, 0],
     [1, 2, 1]],
    dtype=torch.float32).unsqueeze(0).unsqueeze(0)
print(sobel_x.shape)

# Aplicar el filtro de convolución
def apply_Sobel_filter(image, kernel):
    return F.conv2d(image, kernel, padding=1)

# Cargar imagen
image_path = 'ad-benneton.jpg'
image = load_image(image_path)
print(image.shape)

# Aplicar filtros de Prewitt
edges_x = apply_Sobel_filter(image, sobel_x)
edges_y = apply_Sobel_filter(image, sobel_y)

# Calcular la magnitud de los bordes
edges = torch.sqrt(edges_x ** 2 + edges_y ** 2)

# --- Binarización al % del máximo ---
porcentaje = 0.30
threshold = porcentaje * edges.max()
binary_edges = (edges > threshold).float()

# --- Binarización: conservar 5% más alto ---
flat = edges.flatten()
k = int(0.25 * flat.numel())  # cantidad de píxeles a conservar
topk_values, _ = torch.topk(flat, k)  # los k valores más altos
threshold = topk_values.min()  # el menor de esos valores sirve como umbral
binary_edges = (edges >= threshold).float()


# Visualizar resultados
plt.figure(figsize=(12, 4))
plt.subplot(1, 5, 1)
plt.title('Imagen Original')
plt.imshow(image.squeeze(), cmap='gray')

plt.subplot(1, 5, 2)
plt.title('Filtro Sobel Horizontal')
plt.imshow(edges_x.squeeze().detach().numpy(), cmap='gray')

plt.subplot(1, 5, 3)
plt.title('Filtro Sobel Vertical')
plt.imshow(edges_y.squeeze().detach().numpy(), cmap='gray')

plt.subplot(1, 5, 4)
plt.title('Modulo de los gradientes')
plt.imshow(edges.squeeze().detach().numpy(), cmap='gray')

plt.subplot(1, 5, 5)
plt.title('Máscara binarizada')
plt.imshow(binary_edges.squeeze().detach().numpy(), cmap='gray')


plt.show()