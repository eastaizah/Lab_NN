"""
Implementación Simple de Modelos Generativos
Lab 08: IA Generativa
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits


class SimpleVAE:
    """Variational Autoencoder simple."""
    
    def __init__(self, input_dim, latent_dim, hidden_dim=128):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        # Encoder
        self.W_enc1 = np.random.randn(hidden_dim, input_dim) * 0.01
        self.b_enc1 = np.zeros((hidden_dim, 1))
        
        self.W_mu = np.random.randn(latent_dim, hidden_dim) * 0.01
        self.b_mu = np.zeros((latent_dim, 1))
        
        self.W_logvar = np.random.randn(latent_dim, hidden_dim) * 0.01
        self.b_logvar = np.zeros((latent_dim, 1))
        
        # Decoder
        self.W_dec1 = np.random.randn(hidden_dim, latent_dim) * 0.01
        self.b_dec1 = np.zeros((hidden_dim, 1))
        
        self.W_dec2 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b_dec2 = np.zeros((input_dim, 1))
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def encode(self, X):
        """Encode a representación latente."""
        h = self.relu(self.W_enc1 @ X + self.b_enc1)
        mu = self.W_mu @ h + self.b_mu
        logvar = self.W_logvar @ h + self.b_logvar
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick."""
        std = np.exp(0.5 * logvar)
        eps = np.random.randn(*mu.shape)
        z = mu + std * eps
        return z
    
    def decode(self, z):
        """Decode desde representación latente."""
        h = self.relu(self.W_dec1 @ z + self.b_dec1)
        reconstruction = self.sigmoid(self.W_dec2 @ h + self.b_dec2)
        return reconstruction
    
    def forward(self, X):
        """Forward pass completo."""
        mu, logvar = self.encode(X)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar
    
    def compute_loss(self, X, reconstruction, mu, logvar):
        """Compute VAE loss = Reconstruction + KL divergence."""
        # Reconstruction loss (Binary Cross-Entropy)
        epsilon = 1e-10
        recon_loss = -np.mean(
            X * np.log(reconstruction + epsilon) + 
            (1 - X) * np.log(1 - reconstruction + epsilon)
        )
        
        # KL divergence
        kl_loss = -0.5 * np.mean(1 + logvar - mu**2 - np.exp(logvar))
        
        return recon_loss + kl_loss, recon_loss, kl_loss
    
    def generate(self, num_samples):
        """Generate new samples."""
        z = np.random.randn(self.latent_dim, num_samples)
        samples = self.decode(z)
        return samples


def entrenar_vae():
    """Entrenar VAE simple en dígitos."""
    
    print("=" * 70)
    print("ENTRENAMIENTO DE VARIATIONAL AUTOENCODER (VAE)")
    print("=" * 70)
    
    # 1. Cargar datos
    print("\n1. Cargando datos de dígitos...")
    digits = load_digits()
    X = digits.data / 16.0  # Normalizar a [0, 1]
    X = X.T  # (features, samples)
    
    print(f"   Forma de datos: {X.shape}")
    print(f"   Rango: [{X.min():.2f}, {X.max():.2f}]")
    
    # 2. Crear VAE
    print("\n2. Creando VAE...")
    vae = SimpleVAE(input_dim=64, latent_dim=2, hidden_dim=32)
    print(f"   Dimensión latente: {vae.latent_dim}")
    
    # 3. Entrenar (simplificado - sin backprop completo)
    print("\n3. Nota: Esta es una versión simplificada sin entrenamiento completo.")
    print("   Para ver el VAE en acción, ejecutar con PyTorch/TensorFlow.")
    
    # Forward pass de ejemplo
    reconstruction, mu, logvar = vae.forward(X[:, :100])
    loss, recon_loss, kl_loss = vae.compute_loss(X[:, :100], reconstruction, mu, logvar)
    
    print(f"\n   Pérdida inicial:")
    print(f"   - Total: {loss:.4f}")
    print(f"   - Reconstrucción: {recon_loss:.4f}")
    print(f"   - KL Divergence: {kl_loss:.4f}")
    
    # 4. Visualizar reconstrucciones
    print("\n4. Visualizando reconstrucciones...")
    
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    
    for i in range(5):
        # Original
        img_orig = X[:, i].reshape(8, 8)
        axes[0, i].imshow(img_orig, cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Originales', fontsize=10)
        
        # Reconstrucción
        img_recon = reconstruction[:, i].reshape(8, 8)
        axes[1, i].imshow(img_recon, cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Reconstrucciones', fontsize=10)
    
    plt.suptitle('VAE: Originales vs Reconstrucciones', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('vae_reconstruction.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 5. Visualizar espacio latente
    print("\n5. Visualizando espacio latente...")
    
    # Encode todas las muestras
    mu_all, _ = vae.encode(X)
    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(mu_all[0, :], mu_all[1, :], 
                         c=digits.target, cmap='tab10', 
                         alpha=0.6, s=20)
    plt.colorbar(scatter, label='Dígito')
    plt.xlabel('Dimensión Latente 1', fontsize=12)
    plt.ylabel('Dimensión Latente 2', fontsize=12)
    plt.title('Espacio Latente del VAE', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.savefig('vae_latent_space.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 6. Generar nuevos dígitos
    print("\n6. Generando nuevos dígitos...")
    
    generated = vae.generate(num_samples=10)
    
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.ravel()
    
    for i in range(10):
        img = generated[:, i].reshape(8, 8)
        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')
    
    plt.suptitle('Dígitos Generados por el VAE', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('vae_generated.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n" + "=" * 70)
    print("✓ Demostración de VAE completada!")
    print("\nNota: Para entrenar un VAE completo, usa PyTorch o TensorFlow")
    print("      Revisa practica.ipynb para ejemplos completos.")
    print("=" * 70)


class SimpleGAN:
    """GAN simple (estructura básica)."""
    
    def __init__(self, latent_dim, output_dim):
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        
        # Generator
        self.W_gen1 = np.random.randn(128, latent_dim) * 0.01
        self.b_gen1 = np.zeros((128, 1))
        self.W_gen2 = np.random.randn(output_dim, 128) * 0.01
        self.b_gen2 = np.zeros((output_dim, 1))
        
        # Discriminator
        self.W_disc1 = np.random.randn(128, output_dim) * 0.01
        self.b_disc1 = np.zeros((128, 1))
        self.W_disc2 = np.random.randn(1, 128) * 0.01
        self.b_disc2 = np.zeros((1, 1))
    
    def leaky_relu(self, x, alpha=0.2):
        return np.where(x > 0, x, alpha * x)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def tanh(self, x):
        return np.tanh(x)
    
    def generate(self, z):
        """Generator: z → fake image."""
        h = self.leaky_relu(self.W_gen1 @ z + self.b_gen1)
        fake = self.tanh(self.W_gen2 @ h + self.b_gen2)
        return fake
    
    def discriminate(self, x):
        """Discriminator: image → probability of being real."""
        h = self.leaky_relu(self.W_disc1 @ x + self.b_disc1)
        prob = self.sigmoid(self.W_disc2 @ h + self.b_disc2)
        return prob


def demo_gan():
    """Demostración conceptual de GAN."""
    
    print("\n" + "=" * 70)
    print("DEMOSTRACIÓN CONCEPTUAL DE GAN")
    print("=" * 70)
    
    print("\n1. Creando GAN...")
    gan = SimpleGAN(latent_dim=10, output_dim=64)
    
    print("\n2. Estructura:")
    print(f"   Generator: {gan.latent_dim} → 128 → {gan.output_dim}")
    print(f"   Discriminator: {gan.output_dim} → 128 → 1")
    
    print("\n3. Generando muestras...")
    z = np.random.randn(gan.latent_dim, 5)
    fake_images = gan.generate(z)
    
    print("\n4. Discriminando...")
    probs = gan.discriminate(fake_images)
    print(f"   Probabilidades de ser real: {probs.ravel()}")
    
    print("\n5. Visualizando salidas del generador...")
    
    fig, axes = plt.subplots(1, 5, figsize=(12, 3))
    
    for i in range(5):
        img = fake_images[:, i].reshape(8, 8)
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'P(real)={probs[0, i]:.2f}', fontsize=10)
        axes[i].axis('off')
    
    plt.suptitle('Salidas del Generador (sin entrenar)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('gan_samples.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n✓ Demostración completada!")
    print("\nNota: Para entrenar un GAN real, usa PyTorch o TensorFlow")
    print("=" * 70)


if __name__ == "__main__":
    # VAE
    entrenar_vae()
    
    # GAN
    demo_gan()
    
    print("\n" + "=" * 70)
    print("CONCEPTOS CLAVE DE IA GENERATIVA")
    print("=" * 70)
    print("\n1. VAE (Variational Autoencoder):")
    print("   - Encoder: X → z (espacio latente)")
    print("   - Decoder: z → X' (reconstrucción)")
    print("   - Loss: Reconstruction + KL Divergence")
    print("   - Uso: Generación, compresión, interpolación")
    
    print("\n2. GAN (Generative Adversarial Network):")
    print("   - Generator: ruido → imagen falsa")
    print("   - Discriminator: imagen → real/falso")
    print("   - Entrenamiento adversarial")
    print("   - Uso: Generación de imágenes realistas")
    
    print("\n3. Para implementaciones completas:")
    print("   - Usar PyTorch o TensorFlow")
    print("   - Ver practica.ipynb")
    print("   - Explorar modelos pre-entrenados")
    
    print("\n" + "=" * 70)
    print("✓ Lab 08 completado!")
    print("=" * 70)
