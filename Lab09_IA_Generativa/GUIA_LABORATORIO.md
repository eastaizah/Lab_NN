# Guía de Laboratorio: Inteligencia Artificial Generativa

## 📋 Información del Laboratorio

**Título:** Deep Learning Avanzado - Modelos Generativos (VAE, GAN y Diffusion)  
**Código:** Lab 09  
**Duración:** 3-4 horas  
**Nivel:** Avanzado  

## 🎯 Objetivos Específicos

Al completar este laboratorio, serás capaz de:

1. Distinguir claramente entre modelos discriminativos y modelos generativos
2. Comprender la arquitectura y funcionamiento de Autoencoders (AE) básicos
3. Implementar un Variational Autoencoder (VAE) completo desde cero
4. Entender el concepto de espacio latente probabilístico y su importancia
5. Aplicar el reparameterization trick para hacer backpropagation en VAEs
6. Calcular e interpretar la función de pérdida combinada (reconstrucción + KL divergencia)
7. Comprender la arquitectura adversarial de las Generative Adversarial Networks (GANs)
8. Implementar un GAN simple para generación de imágenes
9. Explorar y manipular el espacio latente para generar nuevas muestras
10. Realizar interpolaciones entre puntos en el espacio latente
11. Identificar y mitigar problemas comunes como mode collapse en GANs
12. Conocer los fundamentos de Diffusion Models y su proceso de denoising
13. Evaluar la calidad de modelos generativos usando métricas apropiadas
14. Reconocer las implicaciones éticas del uso de IA generativa

## 📚 Prerrequisitos

### Conocimientos

- **Redes Neuronales**: Arquitecturas, forward/backward pass (Labs 01-02)
- **Funciones de Activación**: ReLU, Sigmoid, Tanh (Lab 03)
- **Funciones de Pérdida**: MSE, Binary Cross-Entropy (Lab 04)
- **Backpropagation**: Cálculo de gradientes y actualización de pesos (Lab 05)
- **Entrenamiento**: Optimizadores, batch processing, epochs (Lab 06)
- **Métricas de Evaluación**: Análisis de rendimiento de modelos (Lab 07)
- **Frameworks Modernos**: PyTorch o TensorFlow (Lab 08)
- **Conceptos Probabilísticos**: Distribuciones normales, divergencia KL
- **Álgebra Lineal**: Operaciones matriciales avanzadas

### Software

- Python 3.8+
- PyTorch 1.10+ o TensorFlow 2.8+
- NumPy 1.19+
- Matplotlib y Seaborn (visualizaciones)
- Jupyter Notebook o JupyterLab
- torchvision o tensorflow-datasets (datasets)
- Pillow (procesamiento de imágenes)

### Material de Lectura

Antes de comenzar, lee:

- `teoria.md` - Marco teórico completo sobre modelos generativos
- `README.md` - Estructura del laboratorio y conceptos clave
- Papers recomendados:
  - "Auto-Encoding Variational Bayes" (Kingma & Welling, 2013) - VAE
  - "Generative Adversarial Networks" (Goodfellow et al., 2014) - GAN
  - "Denoising Diffusion Probabilistic Models" (Ho et al., 2020) - Diffusion

## 📖 Introducción

La **Inteligencia Artificial Generativa** representa uno de los avances más revolucionarios y fascinantes del deep learning moderno. A diferencia de los modelos discriminativos que aprendimos en laboratorios anteriores (que clasifican o predicen), los modelos generativos aprenden a **crear** contenido completamente nuevo: imágenes realistas, texto coherente, música original, e incluso diseños moleculares para nuevos fármacos.

### Contexto del Problema

En los laboratorios anteriores (Labs 01-08), hemos trabajado con **modelos discriminativos** que aprenden funciones del tipo P(y|X):

- **Clasificación de imágenes**: ¿Esta imagen contiene un gato o un perro? → P(y|imagen)
- **Detección de objetos**: ¿Dónde están los objetos en esta imagen? → P(ubicaciones|imagen)
- **Predicción numérica**: ¿Cuál es el precio de esta casa? → P(precio|características)

Estos modelos son excelentes para **reconocer** patrones, pero tienen una limitación fundamental: **no pueden crear contenido nuevo**. Si quieres generar una imagen de un gato que nunca ha existido, un modelo de clasificación no puede ayudarte.

**Los modelos generativos** abordan un problema fundamentalmente diferente: aprender la distribución de probabilidad de los datos P(X) para poder:

- **Generar muestras nuevas**: Crear imágenes, textos o audio completamente originales
- **Completar datos**: Rellenar partes faltantes de una imagen o documento
- **Transformar datos**: Convertir un boceto en una foto realista
- **Interpolar**: Crear transiciones suaves entre dos ejemplos
- **Comprimir**: Encontrar representaciones compactas de datos complejos

### Enfoque con Modelos Generativos

La IA generativa ha evolucionado a través de varias arquitecturas principales:

**1. Autoencoders (AE) - La Base:**
```
Entrada (X) → [Encoder] → Espacio Latente (z) → [Decoder] → Reconstrucción (X')
    784     →   [NN]    →      64           →    [NN]    →       784
```

- **Objetivo**: Comprimir datos a una representación compacta y reconstruirlos
- **Limitación**: El espacio latente puede tener "agujeros", generación limitada
- **Analogía**: Como comprimir una foto JPG - puedes recuperar algo similar, pero no puedes crear fotos nuevas

**2. Variational Autoencoders (VAE) - Probabilísticos:**
```
Entrada (X) → [Encoder] → μ, σ → Sample z ~ N(μ,σ²) → [Decoder] → X'
```

- **Innovación**: En lugar de codificar a un punto fijo, codifica a una **distribución de probabilidad**
- **Ventaja**: Espacio latente continuo y completo, puede generar nuevas muestras
- **Pérdida**: Reconstrucción + KL Divergence (fuerza el espacio latente a seguir una distribución normal)
- **Analogía**: Como aprender el "concepto" de gato en lugar de memorizar gatos específicos

**3. Generative Adversarial Networks (GAN) - Competitivos:**
```
Generator:     z (ruido) → G(z) → imagen falsa
Discriminator: imagen → D(imagen) → [0=falsa, 1=real]
```

- **Innovación**: Dos redes compitiendo - el generador intenta engañar, el discriminador intenta detectar
- **Ventaja**: Genera imágenes de altísima calidad y realismo
- **Desafío**: Entrenamiento inestable, sensible a hiperparámetros
- **Analogía**: Como un falsificador (G) compitiendo contra un detective de arte (D) - ambos mejoran constantemente

**4. Diffusion Models - Denoising:**
```
Forward:  Imagen limpia → ... → Ruido puro (añadir ruido gradualmente)
Reverse:  Ruido puro → ... → Imagen limpia (aprender a eliminar ruido)
```

- **Innovación**: Aprender a revertir un proceso de degradación gradual
- **Ventaja**: Calidad excepcional, entrenamiento estable
- **Uso**: DALL-E 2, Stable Diffusion, Midjourney
- **Analogía**: Como restaurar una pintura antigua eliminando capas de suciedad

### Conceptos Fundamentales

**1. Espacio Latente (Latent Space):**

El espacio latente es una representación comprimida y continua de los datos donde:
- **Cada punto representa una posible muestra** (ej: una imagen de un dígito)
- **Puntos cercanos representan muestras similares** (todos los "3" están juntos)
- **Podemos interpolar** entre puntos para crear transiciones suaves
- **Dimensionalidad reducida**: 784 píxeles → 64 dimensiones latentes

```python
# Ejemplo conceptual
punto_A = [0.5, 0.3, ..., 0.8]  # Representa un "3" escrito de cierta forma
punto_B = [0.6, 0.4, ..., 0.7]  # Representa otro "3" ligeramente diferente
intermedio = 0.5 * punto_A + 0.5 * punto_B  # Mezcla de ambos estilos
```

**2. Pérdida en VAE - Dos Objetivos:**

```python
Pérdida_Total = Pérdida_Reconstrucción + β × Pérdida_KL

# Reconstrucción: ¿Qué tan bien reconstruimos la entrada?
Pérdida_Reconstrucción = ||X - X'||²  # o Binary Cross-Entropy

# KL Divergence: ¿Qué tan "normal" es nuestro espacio latente?
Pérdida_KL = -0.5 × Σ(1 + log(σ²) - μ² - σ²)
```

- **Reconstrucción**: Asegura que podemos recuperar la entrada original
- **KL Divergence**: Fuerza el espacio latente a seguir N(0,1), haciéndolo suave y continuo
- **Trade-off**: Balance entre fidelidad de reconstrucción y calidad de generación

**3. Reparameterization Trick:**

Problema: No podemos hacer backpropagation a través de sampling aleatorio.

Solución elegante:
```python
# ❌ No diferenciable
z = sample_from(N(μ, σ²))

# ✅ Diferenciable - separamos la aleatoriedad
ε = sample_from(N(0, 1))  # Ruido estándar (sin parámetros)
z = μ + σ × ε              # Ahora μ y σ reciben gradientes
```

**4. Entrenamiento Adversarial (GAN):**

```python
# Fase 1: Entrenar Discriminador (D)
pérdida_D_real = -log(D(imágenes_reales))      # Maximizar D(real)
pérdida_D_fake = -log(1 - D(G(ruido)))         # Minimizar D(fake)
pérdida_D = pérdida_D_real + pérdida_D_fake

# Fase 2: Entrenar Generador (G)
pérdida_G = -log(D(G(ruido)))                  # Engañar a D
```

**Equilibrio de Nash**: Cuando G y D alcanzan un punto donde ninguno puede mejorar sin que el otro se adapte.

### Aplicaciones Prácticas

La IA generativa ha transformado múltiples industrias:

**1. Arte y Diseño:**
- **DALL-E 2, Midjourney**: Creación de arte desde texto
- **StyleGAN**: Generación de rostros ultrarrealistas
- **Artbreeder**: Mezcla y evolución de imágenes
- **Adobe Firefly**: Herramientas creativas profesionales

**2. Entretenimiento:**
- **Deepfakes**: Efectos especiales en cine (con regulación ética)
- **Generación de música**: AIVA, MuseNet
- **Generación de niveles**: Videojuegos procedurales
- **Animación**: Generación de movimientos realistas

**3. Medicina y Ciencia:**
- **Diseño de fármacos**: Generar moléculas candidatas
- **Síntesis de datos médicos**: Entrenar modelos sin comprometer privacidad
- **Aumento de datos**: Generar imágenes médicas sintéticas
- **Diseño de proteínas**: AlphaFold y variantes generativas

**4. Procesamiento de Lenguaje:**
- **ChatGPT, GPT-4**: Conversación y escritura
- **GitHub Copilot**: Generación de código
- **Jasper, Copy.ai**: Escritura creativa y marketing
- **Traducción avanzada**: Más natural y contextual

**5. Industria y Negocios:**
- **Diseño de productos**: Generación de prototipos
- **Arquitectura**: Diseños de edificios y espacios
- **Moda**: Generación de patrones y diseños textiles
- **Marketing**: Contenido personalizado a escala

### Motivación Histórica

La evolución de la IA generativa ha sido fascinante:

**2013 - VAE (Kingma & Welling):**
- Introducción del espacio latente probabilístico
- Permitió generación controlable por primera vez
- Fundamento matemático riguroso basado en inferencia variacional

**2014 - GAN (Ian Goodfellow):**
- Paradigma revolucionario: entrenamiento adversarial
- Historia: Goodfellow tuvo la idea en un bar durante un debate con colegas
- Demostró que la competición puede generar excelencia

**2015-2018 - Evolución de GANs:**
- **DCGAN**: Convoluciones para imágenes
- **ProGAN**: Generación progresiva de alta resolución
- **StyleGAN**: Control excepcional sobre características

**2020 - Diffusion Models:**
- **DDPM**: Fundamentos teóricos de modelos de difusión
- Convergencia de ideas de física (procesos estocásticos) y ML

**2021-2024 - Explosión Comercial:**
- **DALL-E 2, Midjourney**: Texto a imagen accesible al público
- **Stable Diffusion**: Open-source, democratización de la tecnología
- **ChatGPT**: IA generativa de texto mainstream
- **Sora**: Video generativo de alta calidad

**Impacto Cultural:**
- De herramienta de investigación a fenómeno global en menos de una década
- Debates sobre arte, creatividad y autoría
- Nuevas profesiones: "prompt engineering"
- Preguntas éticas sobre desinformación y derechos de autor

---

**En este laboratorio**, comenzaremos desde los fundamentos (Autoencoders) y avanzaremos hasta implementar VAE y GAN completos. Aprenderás no solo a usar estos modelos, sino a **entender profundamente** cómo funcionan, por qué funcionan, y cuándo usarlos.

**Advertencia Ética**: Con gran poder viene gran responsabilidad. La IA generativa puede usarse para crear arte hermoso y resolver problemas científicos, pero también para desinformación y deepfakes maliciosos. En este laboratorio también discutiremos las implicaciones éticas y cómo usar esta tecnología responsablemente.

¡Prepárate para crear tu primera imagen completamente generada por IA! 🎨🤖

---

## 🔬 Parte 1: Fundamentos - Autoencoder Simple (45 min)

### 1.1 Introducción Conceptual

Un **Autoencoder** es como un embudo que comprime información y luego la expande de nuevo.

**Analogía**: Imagina que tienes que enviar una foto de 1 MB a través de una conexión lenta:
- **Encoder (Compresor)**: Reduce la foto a 100 KB identificando solo lo esencial
- **Espacio Latente**: Los 100 KB que contienen la esencia de la imagen
- **Decoder (Descompresor)**: Reconstruye una imagen similar desde esos 100 KB

**Arquitectura Visual**:
```
Entrada (784)  →  Encoder  →  Latent (64)  →  Decoder  →  Salida (784)
[Imagen MNIST] → [Comprimir] → [Código]     → [Expandir] → [Reconstrucción]
     28×28     →    NN       →   8×8        →    NN      →     28×28
```

**Diferencia con modelos anteriores**:
- **Clasificador**: X → [NN] → etiqueta (ej: "es un 7")
- **Autoencoder**: X → [NN] → X' (reconstruye la entrada misma)

### 1.2 Implementación del Encoder

El encoder comprime la entrada a una representación de menor dimensión:

```python
import numpy as np
import matplotlib.pyplot as plt

class Encoder:
    """
    Encoder: Comprime datos de alta dimensión a espacio latente.
    """
    
    def __init__(self, input_dim, hidden_dim, latent_dim):
        """
        Args:
            input_dim: Dimensión de entrada (ej: 784 para MNIST)
            hidden_dim: Dimensión de capa oculta (ej: 256)
            latent_dim: Dimensión del espacio latente (ej: 64)
        """
        # Inicialización He para mejor convergencia
        self.W1 = np.random.randn(hidden_dim, input_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros((hidden_dim, 1))
        
        self.W2 = np.random.randn(latent_dim, hidden_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros((latent_dim, 1))
        
        print(f"✅ Encoder creado:")
        print(f"   {input_dim} → {hidden_dim} → {latent_dim}")
    
    def relu(self, x):
        """Activación ReLU."""
        return np.maximum(0, x)
    
    def forward(self, X):
        """
        Codifica entradas al espacio latente.
        
        Args:
            X: (input_dim, batch_size)
        
        Returns:
            z: (latent_dim, batch_size) - representación latente
        """
        # Capa 1: input → hidden
        self.h1 = self.relu(self.W1 @ X + self.b1)
        
        # Capa 2: hidden → latent
        self.z = self.relu(self.W2 @ self.h1 + self.b2)
        
        return self.z

# Ejemplo de uso
print("="*70)
print("1. IMPLEMENTANDO EL ENCODER")
print("="*70)

# Datos de prueba (simulamos imágenes 28×28 = 784)
batch_size = 5
input_dim = 784  # MNIST: 28×28
X_test = np.random.rand(input_dim, batch_size)

# Crear encoder
encoder = Encoder(input_dim=784, hidden_dim=256, latent_dim=64)

# Forward pass
z = encoder.forward(X_test)

print(f"\n📊 Resultados:")
print(f"   Entrada: {X_test.shape} → {X_test.size} valores")
print(f"   Latente: {z.shape} → {z.size} valores")
print(f"   Compresión: {X_test.size / z.size:.1f}x")
```

**Actividad 1.1**: Modifica las dimensiones del encoder (ej: latent_dim=32) y observa cómo cambia la compresión.

**Pregunta de Reflexión 1.1**: ¿Qué información se podría perder al comprimir de 784 a 64 dimensiones?

### 1.3 Implementación del Decoder

El decoder expande la representación latente de vuelta a la dimensión original:

```python
class Decoder:
    """
    Decoder: Expande representación latente a reconstrucción.
    """
    
    def __init__(self, latent_dim, hidden_dim, output_dim):
        """
        Args:
            latent_dim: Dimensión del espacio latente
            hidden_dim: Dimensión de capa oculta
            output_dim: Dimensión de salida (igual a input original)
        """
        self.W1 = np.random.randn(hidden_dim, latent_dim) * np.sqrt(2.0 / latent_dim)
        self.b1 = np.zeros((hidden_dim, 1))
        
        self.W2 = np.random.randn(output_dim, hidden_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros((output_dim, 1))
        
        print(f"✅ Decoder creado:")
        print(f"   {latent_dim} → {hidden_dim} → {output_dim}")
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def sigmoid(self, x):
        """Sigmoid para salida [0,1] (imágenes normalizadas)."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def forward(self, z):
        """
        Decodifica representación latente.
        
        Args:
            z: (latent_dim, batch_size)
        
        Returns:
            X_recon: (output_dim, batch_size) - reconstrucción
        """
        # Capa 1: latent → hidden
        self.h1 = self.relu(self.W1 @ z + self.b1)
        
        # Capa 2: hidden → output (con sigmoid para [0,1])
        self.X_recon = self.sigmoid(self.W2 @ self.h1 + self.b2)
        
        return self.X_recon

# Crear decoder
decoder = Decoder(latent_dim=64, hidden_dim=256, output_dim=784)

# Decodificar la representación latente anterior
X_recon = decoder.forward(z)

print(f"\n📊 Reconstrucción:")
print(f"   Latente: {z.shape}")
print(f"   Reconstrucción: {X_recon.shape}")
print(f"   Rango de valores: [{X_recon.min():.3f}, {X_recon.max():.3f}]")
```

**Actividad 1.2**: Verifica que la forma de `X_recon` sea igual a la de `X_test`.

### 1.4 Autoencoder Completo

Ahora combinamos encoder y decoder en un autoencoder completo:

```python
class SimpleAutoencoder:
    """
    Autoencoder simple: Encoder + Decoder.
    """
    
    def __init__(self, input_dim, hidden_dim, latent_dim):
        """
        Args:
            input_dim: Dimensión de entrada
            hidden_dim: Dimensión de capas ocultas
            latent_dim: Dimensión del espacio latente
        """
        print("\n" + "="*70)
        print("CREANDO AUTOENCODER")
        print("="*70)
        
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        print(f"\n📊 Arquitectura completa:")
        print(f"   {input_dim} → {hidden_dim} → {latent_dim} → {hidden_dim} → {input_dim}")
        print(f"   Ratio de compresión: {input_dim / latent_dim:.1f}x")
    
    def forward(self, X):
        """
        Forward pass completo: X → z → X'
        
        Args:
            X: (input_dim, batch_size)
        
        Returns:
            X_recon: Reconstrucción
            z: Representación latente
        """
        z = self.encoder.forward(X)
        X_recon = self.decoder.forward(z)
        return X_recon, z
    
    def compute_loss(self, X, X_recon):
        """
        Calcula pérdida de reconstrucción (MSE).
        
        Args:
            X: Entrada original
            X_recon: Reconstrucción
        
        Returns:
            loss: Mean Squared Error
        """
        mse = np.mean((X - X_recon) ** 2)
        return mse

# Crear autoencoder
ae = SimpleAutoencoder(input_dim=784, hidden_dim=256, latent_dim=64)

# Forward pass
X_recon, z = ae.forward(X_test)
loss = ae.compute_loss(X_test, X_recon)

print(f"\n🔄 Forward pass completo:")
print(f"   Entrada: {X_test.shape}")
print(f"   Latente: {z.shape}")
print(f"   Reconstrucción: {X_recon.shape}")
print(f"   Pérdida (MSE): {loss:.6f}")
```

**Actividad 1.3**: Calcula la pérdida para diferentes tamaños de espacio latente (32, 64, 128). ¿Cómo afecta al error?

**Pregunta de Reflexión 1.2**: ¿Por qué usamos MSE como función de pérdida en lugar de cross-entropy?

### 1.5 Visualización con Datos Reales

Probemos el autoencoder con dígitos MNIST reales:

```python
from sklearn.datasets import load_digits

def visualizar_autoencoder():
    """
    Visualiza reconstrucciones del autoencoder.
    """
    print("\n" + "="*70)
    print("PROBANDO CON DÍGITOS REALES")
    print("="*70)
    
    # Cargar datos de dígitos (8×8 = 64)
    digits = load_digits()
    X = digits.data / 16.0  # Normalizar a [0, 1]
    X = X.T  # (64, n_samples)
    
    print(f"\n📦 Dataset cargado:")
    print(f"   Forma: {X.shape}")
    print(f"   Rango: [{X.min():.2f}, {X.max():.2f}]")
    
    # Crear autoencoder (ajustado para 8×8)
    ae_small = SimpleAutoencoder(input_dim=64, hidden_dim=32, latent_dim=16)
    
    # Reconstruir primeras muestras
    n_samples = 10
    X_batch = X[:, :n_samples]
    X_recon, z = ae_small.forward(X_batch)
    loss = ae_small.compute_loss(X_batch, X_recon)
    
    print(f"\n📊 Resultados (sin entrenar):")
    print(f"   Pérdida: {loss:.6f}")
    
    # Visualizar originales vs reconstrucciones
    fig, axes = plt.subplots(2, n_samples, figsize=(15, 3))
    
    for i in range(n_samples):
        # Original
        img_orig = X[:, i].reshape(8, 8)
        axes[0, i].imshow(img_orig, cmap='gray', vmin=0, vmax=1)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Originales', fontsize=10, loc='left')
        
        # Reconstrucción
        img_recon = X_recon[:, i].reshape(8, 8)
        axes[1, i].imshow(img_recon, cmap='gray', vmin=0, vmax=1)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Reconstrucciones', fontsize=10, loc='left')
    
    plt.suptitle('Autoencoder Simple (Sin Entrenar)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('autoencoder_reconstruccion.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n💡 Observación:")
    print(f"   Las reconstrucciones son borrosas porque el autoencoder")
    print(f"   no está entrenado. En Lab 06 aprendimos a entrenar redes.")
    print(f"   Con entrenamiento, las reconstrucciones mejorarían mucho.")

visualizar_autoencoder()
```

**Actividad 1.4**: Cambia `latent_dim` a 8 y a 32. ¿Cómo afecta visualmente a las reconstrucciones?

### Actividades

**Actividad 1.5**: Implementa una función `compression_ratio()` que calcule y retorne el ratio de compresión del autoencoder.

**Actividad 1.6**: Crea un autoencoder con 3 capas en el encoder y 3 en el decoder. ¿Mejora la reconstrucción?

**Actividad 1.7**: Añade un método `info()` al `SimpleAutoencoder` que imprima el número total de parámetros.

### Preguntas de Reflexión

**Pregunta 1.3 (Concebir)**: ¿Qué aplicaciones prácticas tiene comprimir imágenes a un espacio latente pequeño?

**Pregunta 1.4 (Diseñar)**: Si quisieras comprimir 10x más, ¿cambiarías el `latent_dim` o la arquitectura completa? ¿Por qué?

**Pregunta 1.5 (Implementar)**: ¿Por qué usamos sigmoid en la última capa del decoder?

**Pregunta 1.6 (Operar)**: En un sistema de almacenamiento masivo de imágenes, ¿qué trade-offs considerarías entre compresión y calidad?

---

## 🔬 Parte 2: Variational Autoencoder (VAE) (60 min)

### 2.1 De Autoencoder a VAE - El Salto Conceptual

**Problema con Autoencoders Simples**:
```
Autoencoder tradicional:
  X → [Encoder] → z (punto fijo) → [Decoder] → X'
  
  Problema: El espacio latente tiene "agujeros"
  - z₁ = [0.5, 0.3] → dígito "3"
  - z₂ = [0.5, 0.4] → ¿? (podría ser basura)
  - No podemos generar muestras nuevas confiablemente
```

**Solución VAE**:
```
VAE:
  X → [Encoder] → (μ, σ) → Sample z ~ N(μ,σ²) → [Decoder] → X'
  
  Ventaja: El espacio latente es continuo y completo
  - Cualquier z muestreado de N(0,1) → imagen válida
  - Podemos generar infinitas muestras nuevas
  - Interpolación suave entre puntos
```

**Analogía Detallada**:

Imagina que quieres comprimir la forma de escribir dígitos:

- **Autoencoder**: Guarda la posición exacta de cada trazo
  - Problema: Solo puedes reproducir los dígitos exactos que guardaste
  - Como tomar fotos de cada dígito

- **VAE**: Aprende la "distribución" de cómo se escriben los dígitos
  - Codifica: "Este '3' tiene un bucle superior de tamaño medio (μ=0.5) con variación (σ=0.1)"
  - Puede generar infinitos "3" diferentes pero todos válidos
  - Como aprender el "concepto" de cómo escribir un "3"

### 2.2 Encoder Probabilístico - μ y σ

En un VAE, el encoder no produce un punto fijo z, sino parámetros de una distribución:

```python
class VAEEncoder:
    """
    Encoder para VAE: produce μ (media) y log(σ²) (log-varianza).
    """
    
    def __init__(self, input_dim, hidden_dim, latent_dim):
        """
        Args:
            input_dim: Dimensión de entrada
            hidden_dim: Dimensión de capa oculta compartida
            latent_dim: Dimensión del espacio latente
        """
        # Capa compartida
        self.W_shared = np.random.randn(hidden_dim, input_dim) * np.sqrt(2.0 / input_dim)
        self.b_shared = np.zeros((hidden_dim, 1))
        
        # Rama para μ (media)
        self.W_mu = np.random.randn(latent_dim, hidden_dim) * np.sqrt(2.0 / hidden_dim)
        self.b_mu = np.zeros((latent_dim, 1))
        
        # Rama para log(σ²) (log-varianza)
        self.W_logvar = np.random.randn(latent_dim, hidden_dim) * np.sqrt(2.0 / hidden_dim)
        self.b_logvar = np.zeros((latent_dim, 1))
        
        print(f"✅ VAE Encoder creado:")
        print(f"   Entrada: {input_dim}")
        print(f"   Hidden: {hidden_dim}")
        print(f"   Latente: {latent_dim} (μ y log-σ²)")
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def forward(self, X):
        """
        Codifica entrada a distribución latente.
        
        Args:
            X: (input_dim, batch_size)
        
        Returns:
            mu: (latent_dim, batch_size) - media
            logvar: (latent_dim, batch_size) - log-varianza
        """
        # Capa compartida
        h = self.relu(self.W_shared @ X + self.b_shared)
        
        # Ramificación
        mu = self.W_mu @ h + self.b_mu
        logvar = self.W_logvar @ h + self.b_logvar
        
        return mu, logvar

# Ejemplo
print("\n" + "="*70)
print("2. VAE ENCODER - CODIFICACIÓN PROBABILÍSTICA")
print("="*70)

X_test = np.random.rand(64, 5)  # 5 imágenes de 8×8
vae_encoder = VAEEncoder(input_dim=64, hidden_dim=32, latent_dim=16)

mu, logvar = vae_encoder.forward(X_test)

print(f"\n📊 Salidas del encoder:")
print(f"   μ shape: {mu.shape}")
print(f"   log(σ²) shape: {logvar.shape}")
print(f"\n   Ejemplo para muestra 0:")
print(f"   μ₀ = {mu[:5, 0]}...")  # Primeras 5 dims
print(f"   log(σ²)₀ = {logvar[:5, 0]}...")
```

**Actividad 2.1**: Calcula σ = exp(0.5 × logvar) manualmente para la primera muestra. ¿Qué valores obtienes?

**Pregunta de Reflexión 2.1**: ¿Por qué parametrizamos log(σ²) en lugar de σ directamente?

### 2.3 Reparameterization Trick - La Clave del VAE

**El Problema**:
```python
# ❌ NO FUNCIONA - No podemos hacer backpropagation
z = np.random.normal(mu, sigma)  # Sampling rompe el flujo de gradientes
```

**La Solución Elegante**:
```python
# ✅ FUNCIONA - Reparameterization trick
epsilon = np.random.normal(0, 1, size=mu.shape)  # Ruido estándar
z = mu + sigma * epsilon  # Ahora μ y σ reciben gradientes
```

Implementación:

```python
def reparameterize(mu, logvar):
    """
    Reparameterization trick: permite backpropagation a través de sampling.
    
    z = μ + σ × ε, donde ε ~ N(0,1)
    
    Args:
        mu: (latent_dim, batch_size) - media
        logvar: (latent_dim, batch_size) - log-varianza
    
    Returns:
        z: (latent_dim, batch_size) - muestras latentes
    """
    # Calcular desviación estándar desde log-varianza
    std = np.exp(0.5 * logvar)
    
    # Muestrear ruido de distribución normal estándar
    epsilon = np.random.randn(*mu.shape)
    
    # Reparameterizar: z = μ + σ × ε
    z = mu + std * epsilon
    
    return z

# Demostración
print("\n" + "="*70)
print("3. REPARAMETERIZATION TRICK")
print("="*70)

# Usar μ y logvar del ejemplo anterior
z = reparameterize(mu, logvar)

print(f"\n📊 Muestreo latente:")
print(f"   Input: μ={mu.shape}, log(σ²)={logvar.shape}")
print(f"   Output: z={z.shape}")

print(f"\n   Estadísticas de z:")
print(f"   Media: {z.mean():.4f} (debería estar cerca de 0)")
print(f"   Std: {z.std():.4f} (debería estar cerca de 1)")

# Visualizar distribución
print(f"\n💡 Explicación:")
print(f"   1. Calculamos σ = exp(0.5 × log(σ²))")
print(f"   2. Muestreamos ε ~ N(0,1)")
print(f"   3. Calculamos z = μ + σ × ε")
print(f"   4. Ahora los gradientes fluyen a través de μ y σ!")
```

**Actividad 2.2**: Genera 1000 muestras de z usando el mismo μ y logvar. Grafica su distribución.

**Pregunta de Reflexión 2.2**: ¿Por qué este "truco" permite hacer backpropagation?

### 2.4 VAE Decoder - Igual que Antes

El decoder del VAE es idéntico al del autoencoder simple:

```python
class VAEDecoder:
    """
    Decoder para VAE: igual que autoencoder normal.
    """
    
    def __init__(self, latent_dim, hidden_dim, output_dim):
        self.W1 = np.random.randn(hidden_dim, latent_dim) * np.sqrt(2.0 / latent_dim)
        self.b1 = np.zeros((hidden_dim, 1))
        
        self.W2 = np.random.randn(output_dim, hidden_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros((output_dim, 1))
        
        print(f"✅ VAE Decoder creado:")
        print(f"   {latent_dim} → {hidden_dim} → {output_dim}")
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def forward(self, z):
        """
        Decodifica z a reconstrucción.
        
        Args:
            z: (latent_dim, batch_size)
        
        Returns:
            X_recon: (output_dim, batch_size)
        """
        h = self.relu(self.W1 @ z + self.b1)
        X_recon = self.sigmoid(self.W2 @ h + self.b2)
        return X_recon

vae_decoder = VAEDecoder(latent_dim=16, hidden_dim=32, output_dim=64)
X_recon = vae_decoder.forward(z)

print(f"\n📊 Reconstrucción:")
print(f"   z: {z.shape} → X_recon: {X_recon.shape}")
```

### 2.5 Función de Pérdida VAE - Dos Componentes

**La pérdida del VAE combina dos objetivos**:

```python
Pérdida_Total = Reconstrucción + β × KL_Divergence

1. Reconstrucción: ¿Qué tan bien reconstruimos X?
2. KL Divergence: ¿Qué tan "normal" es nuestro espacio latente?
```

Implementación detallada:

```python
def vae_loss(X, X_recon, mu, logvar, beta=1.0):
    """
    Calcula la pérdida completa del VAE.
    
    Loss = Reconstruction + β × KL_Divergence
    
    Args:
        X: (input_dim, batch_size) - entrada original
        X_recon: (input_dim, batch_size) - reconstrucción
        mu: (latent_dim, batch_size) - media latente
        logvar: (latent_dim, batch_size) - log-varianza latente
        beta: peso de KL divergence (típicamente 1.0)
    
    Returns:
        total_loss: pérdida total
        recon_loss: pérdida de reconstrucción
        kl_loss: KL divergence
    """
    batch_size = X.shape[1]
    
    # 1. Pérdida de Reconstrucción (Binary Cross-Entropy)
    epsilon = 1e-10  # Para estabilidad numérica
    recon_loss = -np.sum(
        X * np.log(X_recon + epsilon) + 
        (1 - X) * np.log(1 - X_recon + epsilon)
    ) / batch_size
    
    # 2. KL Divergence
    # KL(N(μ,σ²) || N(0,1)) = -0.5 × Σ(1 + log(σ²) - μ² - σ²)
    kl_loss = -0.5 * np.sum(1 + logvar - mu**2 - np.exp(logvar)) / batch_size
    
    # 3. Pérdida Total
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, kl_loss

# Calcular pérdida
print("\n" + "="*70)
print("4. FUNCIÓN DE PÉRDIDA VAE")
print("="*70)

total_loss, recon_loss, kl_loss = vae_loss(X_test, X_recon, mu, logvar)

print(f"\n📊 Componentes de pérdida (sin entrenar):")
print(f"   Reconstrucción: {recon_loss:.4f}")
print(f"   KL Divergence:  {kl_loss:.4f}")
print(f"   TOTAL:          {total_loss:.4f}")

print(f"\n💡 Interpretación:")
print(f"   - Reconstrucción: qué tan bien recuperamos la entrada")
print(f"   - KL Divergence: qué tan cerca está q(z|X) de N(0,1)")
print(f"   - Balance: trade-off entre fidelidad y regularización")
```

**Actividad 2.3**: Varía β de 0.1 a 10. ¿Cómo cambia el balance entre las dos pérdidas?

**Pregunta de Reflexión 2.3**: ¿Qué pasaría si β=0? ¿Y si β=1000?

### 2.6 VAE Completo - Todo Junto

Ahora juntamos todas las piezas:

```python
class VariationalAutoencoder:
    """
    Variational Autoencoder completo.
    """
    
    def __init__(self, input_dim, hidden_dim, latent_dim):
        """
        Args:
            input_dim: Dimensión de entrada
            hidden_dim: Dimensión de capas ocultas
            latent_dim: Dimensión del espacio latente
        """
        print("\n" + "="*70)
        print("CREANDO VARIATIONAL AUTOENCODER (VAE)")
        print("="*70)
        
        self.encoder = VAEEncoder(input_dim, hidden_dim, latent_dim)
        self.decoder = VAEDecoder(latent_dim, hidden_dim, input_dim)
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        print(f"\n📊 Arquitectura VAE:")
        print(f"   Input: {input_dim}")
        print(f"   → Encoder → (μ, log-σ²): {latent_dim}")
        print(f"   → Reparameterize → z: {latent_dim}")
        print(f"   → Decoder → Output: {input_dim}")
    
    def forward(self, X):
        """
        Forward pass completo del VAE.
        
        Args:
            X: (input_dim, batch_size)
        
        Returns:
            X_recon: reconstrucción
            mu: media latente
            logvar: log-varianza latente
        """
        # Encoder: X → (μ, log-σ²)
        mu, logvar = self.encoder.forward(X)
        
        # Reparameterization trick: (μ, log-σ²) → z
        z = reparameterize(mu, logvar)
        
        # Decoder: z → X'
        X_recon = self.decoder.forward(z)
        
        return X_recon, mu, logvar
    
    def compute_loss(self, X, X_recon, mu, logvar, beta=1.0):
        """Calcula pérdida VAE."""
        return vae_loss(X, X_recon, mu, logvar, beta)
    
    def generate(self, num_samples):
        """
        Genera nuevas muestras muestreando z ~ N(0,1).
        
        Args:
            num_samples: número de muestras a generar
        
        Returns:
            X_generated: muestras generadas
        """
        # Muestrear z desde prior N(0,1)
        z = np.random.randn(self.latent_dim, num_samples)
        
        # Decodificar
        X_generated = self.decoder.forward(z)
        
        return X_generated

# Crear VAE completo
vae = VariationalAutoencoder(input_dim=64, hidden_dim=32, latent_dim=16)

# Forward pass
X_recon_vae, mu_vae, logvar_vae = vae.forward(X_test)
total, recon, kl = vae.compute_loss(X_test, X_recon_vae, mu_vae, logvar_vae)

print(f"\n🔄 Test forward pass:")
print(f"   Input: {X_test.shape}")
print(f"   Output: {X_recon_vae.shape}")
print(f"   Pérdida total: {total:.4f}")
```

**Actividad 2.4**: Crea un VAE con `latent_dim=8`. ¿Es más difícil reconstruir con menos dimensiones?

### 2.7 Exploración del Espacio Latente

Una de las características más poderosas del VAE es su espacio latente continuo:

```python
def explorar_espacio_latente():
    """
    Explora el espacio latente del VAE.
    """
    from sklearn.datasets import load_digits
    
    print("\n" + "="*70)
    print("5. EXPLORACIÓN DEL ESPACIO LATENTE")
    print("="*70)
    
    # Cargar datos
    digits = load_digits()
    X = digits.data / 16.0
    X = X.T  # (64, n_samples)
    y = digits.target
    
    # Crear VAE con latent_dim=2 para visualización
    vae_viz = VariationalAutoencoder(input_dim=64, hidden_dim=32, latent_dim=2)
    
    # Encodear todos los dígitos
    mu_all, logvar_all = vae_viz.encoder.forward(X)
    
    # Visualizar espacio latente
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(mu_all[0, :], mu_all[1, :], 
                         c=y, cmap='tab10', 
                         alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
    plt.colorbar(scatter, label='Dígito')
    plt.xlabel('Dimensión Latente 1', fontsize=12)
    plt.ylabel('Dimensión Latente 2', fontsize=12)
    plt.title('Espacio Latente del VAE (2D)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    plt.axvline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    plt.tight_layout()
    plt.savefig('vae_latent_space.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n💡 Observaciones:")
    print(f"   - Cada color representa un dígito diferente")
    print(f"   - Dígitos similares están cercanos en el espacio latente")
    print(f"   - El espacio es continuo (no hay saltos bruscos)")
    print(f"   - Con entrenamiento, las separaciones serían más claras")

explorar_espacio_latente()
```

**Actividad 2.5**: Modifica para usar `latent_dim=3` y haz una visualización 3D.

### 2.8 Generación de Nuevas Muestras

La ventaja del VAE: podemos generar muestras completamente nuevas:

```python
def generar_nuevas_muestras():
    """
    Genera dígitos completamente nuevos.
    """
    print("\n" + "="*70)
    print("6. GENERACIÓN DE NUEVAS MUESTRAS")
    print("="*70)
    
    # Usar el VAE creado anteriormente
    num_samples = 16
    
    print(f"\n🎨 Generando {num_samples} dígitos nuevos...")
    X_generated = vae.generate(num_samples)
    
    # Visualizar
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    axes = axes.ravel()
    
    for i in range(num_samples):
        img = X_generated[:, i].reshape(8, 8)
        axes[i].imshow(img, cmap='gray', vmin=0, vmax=1)
        axes[i].axis('off')
        axes[i].set_title(f'Muestra {i+1}', fontsize=9)
    
    plt.suptitle('Dígitos Generados por VAE (sin entrenar)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('vae_generated_samples.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n💡 Nota importante:")
    print(f"   Estas muestras se ven aleatorias porque el VAE no está entrenado.")
    print(f"   Con entrenamiento (usando backpropagation del Lab 05),")
    print(f"   generaría dígitos realistas y variados.")

generar_nuevas_muestras()
```

**Actividad 2.6**: Genera 100 muestras. ¿Hay alguna que se parezca a un dígito por casualidad?

### 2.9 Interpolación en el Espacio Latente

Podemos crear transiciones suaves entre dos dígitos:

```python
def interpolar_digitos():
    """
    Interpola entre dos dígitos en el espacio latente.
    """
    from sklearn.datasets import load_digits
    
    print("\n" + "="*70)
    print("7. INTERPOLACIÓN EN ESPACIO LATENTE")
    print("="*70)
    
    # Cargar datos
    digits = load_digits()
    X = digits.data / 16.0
    X = X.T
    
    # Elegir dos dígitos (ej: índices 0 y 10)
    idx1, idx2 = 0, 100
    x1 = X[:, idx1:idx1+1]
    x2 = X[:, idx2:idx2+1]
    
    # Encodear a espacio latente
    mu1, logvar1 = vae.encoder.forward(x1)
    mu2, logvar2 = vae.encoder.forward(x2)
    
    # Interpolar (10 pasos)
    n_steps = 10
    alphas = np.linspace(0, 1, n_steps)
    
    interpolated = []
    for alpha in alphas:
        # Interpolación lineal en espacio latente
        z_interp = (1 - alpha) * mu1 + alpha * mu2
        
        # Decodificar
        x_interp = vae.decoder.forward(z_interp)
        interpolated.append(x_interp)
    
    # Visualizar
    fig, axes = plt.subplots(1, n_steps, figsize=(15, 2))
    
    for i, x in enumerate(interpolated):
        img = x.reshape(8, 8)
        axes[i].imshow(img, cmap='gray', vmin=0, vmax=1)
        axes[i].axis('off')
        axes[i].set_title(f'α={alphas[i]:.1f}', fontsize=9)
    
    plt.suptitle(f'Interpolación: Dígito {digits.target[idx1]} → Dígito {digits.target[idx2]}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('vae_interpolation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n💡 Interpretación:")
    print(f"   Cada paso muestra una mezcla gradual entre los dos dígitos.")
    print(f"   Con VAE entrenado, la transición sería suave y realista.")
    print(f"   Esto demuestra la continuidad del espacio latente.")

interpolar_digitos()
```

**Actividad 2.7**: Interpola entre tres dígitos diferentes usando interpolación esférica en lugar de lineal.

### Actividades Integradoras

**Actividad 2.8**: Implementa una función `encode_decode_test()` que codifique y decodifique 100 imágenes y mida el error promedio.

**Actividad 2.9**: Crea una visualización 2D del espacio latente mostrando reconstrucciones en una cuadrícula (latent space walk).

**Actividad 2.10**: Implementa `beta_vae()` que permita variar β y observa cómo afecta a las reconstrucciones vs regularización.

**Actividad 2.11**: Compara VAE con Autoencoder simple: ¿cuál genera mejores muestras nuevas?

### Preguntas de Reflexión

**Pregunta 2.4 (Concebir)**: ¿Cómo podrías usar un VAE para detectar imágenes anómalas o "fuera de distribución"?

**Pregunta 2.5 (Diseñar)**: Si quisieras generar dígitos de un número específico (ej: solo "7"), ¿cómo modificarías la arquitectura?

**Pregunta 2.6 (Implementar)**: ¿Por qué KL divergence fuerza el espacio latente a ser N(0,1)? ¿Qué pasaría sin esta regularización?

**Pregunta 2.7 (Operar)**: En una aplicación de generación de rostros, ¿qué dimensión de espacio latente recomendarías y por qué?

---

## 🔬 Parte 3: Generative Adversarial Networks (GAN) (45 min)

### 3.1 Introducción al Entrenamiento Adversarial

**GAN: Un Juego de Dos Jugadores**

Imagina dos redes neuronales compitiendo:

- **Generador (G)**: Intenta crear imágenes falsas que parezcan reales
  - Como un falsificador de billetes
  - Input: ruido aleatorio z
  - Output: imagen falsa

- **Discriminador (D)**: Intenta distinguir imágenes reales de falsas
  - Como un detective de billetes falsos
  - Input: imagen (real o falsa)
  - Output: probabilidad de ser real [0, 1]

**El Proceso de Competición**:

```
Ronda 1:
  G crea billetes falsos malos → D detecta fácilmente
  
Ronda 2:
  D mejora su detección → G ajusta su técnica
  
Ronda 3:
  G crea billetes mejores → D se vuelve más experto
  
...

Ronda N:
  G crea billetes casi perfectos ↔ D apenas puede distinguir
  
¡EQUILIBRIO! → G genera imágenes realistas
```

**Arquitectura Visual**:
```
Generador:
  z (ruido) → [NN] → imagen falsa
  (100,)    → Dense → (784,)
  
Discriminador:
  imagen → [NN] → probabilidad
  (784,) → Dense → (1,)  # 0=falso, 1=real
  
Entrenamiento:
  1. Entrenar D: maximizar D(real), minimizar D(fake)
  2. Entrenar G: maximizar D(fake) (engañar a D)
  3. Repetir alternadamente
```

### 3.2 Implementación del Generador

El generador transforma ruido aleatorio en imágenes:

```python
class Generator:
    """
    Generador de GAN: ruido → imagen falsa.
    """
    
    def __init__(self, latent_dim, hidden_dim, output_dim):
        """
        Args:
            latent_dim: Dimensión del ruido de entrada (ej: 100)
            hidden_dim: Dimensión de capas ocultas (ej: 128)
            output_dim: Dimensión de salida (ej: 784 para MNIST)
        """
        # Capa 1
        self.W1 = np.random.randn(hidden_dim, latent_dim) * np.sqrt(2.0 / latent_dim)
        self.b1 = np.zeros((hidden_dim, 1))
        
        # Capa 2
        self.W2 = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros((hidden_dim, 1))
        
        # Capa de salida
        self.W3 = np.random.randn(output_dim, hidden_dim) * np.sqrt(2.0 / hidden_dim)
        self.b3 = np.zeros((output_dim, 1))
        
        print(f"✅ Generador creado:")
        print(f"   {latent_dim} → {hidden_dim} → {hidden_dim} → {output_dim}")
        
        self.latent_dim = latent_dim
    
    def leaky_relu(self, x, alpha=0.2):
        """LeakyReLU: permite gradientes negativos pequeños."""
        return np.where(x > 0, x, alpha * x)
    
    def tanh(self, x):
        """Tanh: salida en [-1, 1]."""
        return np.tanh(x)
    
    def forward(self, z):
        """
        Genera imágenes falsas desde ruido.
        
        Args:
            z: (latent_dim, batch_size) - ruido
        
        Returns:
            fake_images: (output_dim, batch_size) - en rango [-1, 1]
        """
        # Capa 1
        h1 = self.leaky_relu(self.W1 @ z + self.b1)
        
        # Capa 2
        h2 = self.leaky_relu(self.W2 @ h1 + self.b2)
        
        # Capa de salida con tanh para [-1, 1]
        fake_images = self.tanh(self.W3 @ h2 + self.b3)
        
        return fake_images
    
    def generate_noise(self, batch_size):
        """
        Genera ruido aleatorio para el generador.
        
        Args:
            batch_size: número de muestras de ruido
        
        Returns:
            z: (latent_dim, batch_size)
        """
        return np.random.randn(self.latent_dim, batch_size)

# Crear generador
print("\n" + "="*70)
print("1. GENERADOR DE GAN")
print("="*70)

generator = Generator(latent_dim=100, hidden_dim=128, output_dim=64)

# Generar muestras falsas
z = generator.generate_noise(batch_size=5)
fake_images = generator.forward(z)

print(f"\n📊 Generación:")
print(f"   Ruido z: {z.shape}")
print(f"   Imágenes falsas: {fake_images.shape}")
print(f"   Rango: [{fake_images.min():.3f}, {fake_images.max():.3f}]")
```

**Actividad 3.1**: Genera 100 imágenes falsas y visualiza algunas. ¿Se parecen a dígitos (sin entrenar)?

**Pregunta de Reflexión 3.1**: ¿Por qué usamos tanh en la última capa en lugar de sigmoid?

### 3.3 Implementación del Discriminador

El discriminador clasifica imágenes como reales o falsas:

```python
class Discriminator:
    """
    Discriminador de GAN: imagen → probabilidad de ser real.
    """
    
    def __init__(self, input_dim, hidden_dim):
        """
        Args:
            input_dim: Dimensión de entrada (ej: 784)
            hidden_dim: Dimensión de capas ocultas (ej: 128)
        """
        # Capa 1
        self.W1 = np.random.randn(hidden_dim, input_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros((hidden_dim, 1))
        
        # Capa 2
        self.W2 = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros((hidden_dim, 1))
        
        # Capa de salida (clasificación binaria)
        self.W3 = np.random.randn(1, hidden_dim) * np.sqrt(2.0 / hidden_dim)
        self.b3 = np.zeros((1, 1))
        
        print(f"✅ Discriminador creado:")
        print(f"   {input_dim} → {hidden_dim} → {hidden_dim} → 1")
    
    def leaky_relu(self, x, alpha=0.2):
        return np.where(x > 0, x, alpha * x)
    
    def sigmoid(self, x):
        """Sigmoid: salida en [0, 1] (probabilidad)."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def forward(self, images):
        """
        Discrimina si las imágenes son reales o falsas.
        
        Args:
            images: (input_dim, batch_size)
        
        Returns:
            probs: (1, batch_size) - probabilidad de ser real
        """
        # Capa 1
        h1 = self.leaky_relu(self.W1 @ images + self.b1)
        
        # Capa 2
        h2 = self.leaky_relu(self.W2 @ h1 + self.b2)
        
        # Capa de salida con sigmoid
        probs = self.sigmoid(self.W3 @ h2 + self.b3)
        
        return probs

# Crear discriminador
print("\n" + "="*70)
print("2. DISCRIMINADOR DE GAN")
print("="*70)

discriminator = Discriminator(input_dim=64, hidden_dim=128)

# Probar con imágenes falsas
probs_fake = discriminator.forward(fake_images)

print(f"\n📊 Discriminación (imágenes falsas):")
print(f"   Input: {fake_images.shape}")
print(f"   Output probabilities: {probs_fake.shape}")
print(f"   Probabilidades: {probs_fake.ravel()}")
print(f"\n💡 Interpretación:")
print(f"   Valores cercanos a 0 = imagen falsa")
print(f"   Valores cercanos a 1 = imagen real")
print(f"   Sin entrenar, las probabilidades son aleatorias (~0.5)")
```

**Actividad 3.2**: Crea imágenes "reales" (de MNIST) y compara las probabilidades del discriminador con las de imágenes falsas.

### 3.4 Funciones de Pérdida GAN

**Pérdida del Discriminador**:
```python
Objetivo: Maximizar D(real) y minimizar D(fake)
Pérdida_D = -[log(D(real)) + log(1 - D(fake))]
```

**Pérdida del Generador**:
```python
Objetivo: Maximizar D(fake) para engañar a D
Pérdida_G = -log(D(fake))
```

Implementación:

```python
def discriminator_loss(D_real, D_fake):
    """
    Pérdida del discriminador (Binary Cross-Entropy).
    
    Args:
        D_real: probabilidades para imágenes reales
        D_fake: probabilidades para imágenes falsas
    
    Returns:
        loss: pérdida del discriminador
    """
    epsilon = 1e-10  # Estabilidad numérica
    
    # Maximizar D(real) → minimizar -log(D(real))
    loss_real = -np.mean(np.log(D_real + epsilon))
    
    # Minimizar D(fake) → minimizar -log(1 - D(fake))
    loss_fake = -np.mean(np.log(1 - D_fake + epsilon))
    
    # Pérdida total
    loss = loss_real + loss_fake
    
    return loss, loss_real, loss_fake

def generator_loss(D_fake):
    """
    Pérdida del generador.
    
    Args:
        D_fake: probabilidades del discriminador para imágenes falsas
    
    Returns:
        loss: pérdida del generador
    """
    epsilon = 1e-10
    
    # Maximizar D(fake) → minimizar -log(D(fake))
    loss = -np.mean(np.log(D_fake + epsilon))
    
    return loss

# Ejemplo de cálculo
print("\n" + "="*70)
print("3. FUNCIONES DE PÉRDIDA GAN")
print("="*70)

from sklearn.datasets import load_digits

# Cargar imágenes reales
digits = load_digits()
real_images = (digits.data / 8.0 - 1.0).T[:, :5]  # Normalizar a [-1, 1]

# Discriminar reales y falsas
D_real = discriminator.forward(real_images)
D_fake = discriminator.forward(fake_images)

# Calcular pérdidas
d_loss, d_loss_real, d_loss_fake = discriminator_loss(D_real, D_fake)
g_loss = generator_loss(D_fake)

print(f"\n📊 Pérdidas (sin entrenar):")
print(f"   Discriminador:")
print(f"     - Pérdida real:  {d_loss_real:.4f}")
print(f"     - Pérdida fake:  {d_loss_fake:.4f}")
print(f"     - TOTAL:         {d_loss:.4f}")
print(f"\n   Generador:")
print(f"     - Pérdida:       {g_loss:.4f}")

print(f"\n💡 Objetivo del entrenamiento:")
print(f"   - D intenta minimizar d_loss")
print(f"   - G intenta minimizar g_loss")
print(f"   - Convergen cuando D(real)≈1 y D(fake)≈0.5 (equilibrio)")
```

**Actividad 3.3**: Calcula las pérdidas con diferentes probabilidades manualmente para entender cómo funcionan.

**Pregunta de Reflexión 3.2**: ¿Por qué la meta de G es que D(fake)≈0.5 y no D(fake)≈1?

### 3.5 Loop de Entrenamiento GAN (Conceptual)

Aunque no implementaremos backpropagation completo aquí (eso fue Lab 05), veamos la estructura del entrenamiento:

```python
def entrenar_gan_conceptual(generator, discriminator, real_images, 
                            n_epochs=100, batch_size=32):
    """
    Esquema conceptual del entrenamiento de GAN.
    
    NOTA: Esta es una versión simplificada sin backpropagation.
    Para implementación completa, usar PyTorch/TensorFlow.
    """
    print("\n" + "="*70)
    print("4. ESQUEMA DE ENTRENAMIENTO GAN")
    print("="*70)
    
    print(f"\n📚 Proceso de entrenamiento:")
    print(f"   Total épocas: {n_epochs}")
    print(f"   Batch size: {batch_size}")
    
    # Simulación del proceso
    for epoch in range(5):  # Solo 5 para demo
        print(f"\n--- Época {epoch+1} ---")
        
        # Paso 1: Entrenar Discriminador
        print("  🔵 Entrenando Discriminador:")
        
        # 1a. Forward pass en imágenes reales
        batch_real = real_images[:, :batch_size]
        D_real = discriminator.forward(batch_real)
        
        # 1b. Generar imágenes falsas
        z = generator.generate_noise(batch_size)
        batch_fake = generator.forward(z)
        D_fake = discriminator.forward(batch_fake)
        
        # 1c. Calcular pérdida de D
        d_loss, d_real, d_fake = discriminator_loss(D_real, D_fake)
        
        print(f"     D(real): {D_real.mean():.3f}, D(fake): {D_fake.mean():.3f}")
        print(f"     Pérdida D: {d_loss:.4f}")
        
        # 1d. [BACKPROP AQUÍ] Actualizar pesos de D
        # discriminator.pesos -= learning_rate * gradientes
        
        # Paso 2: Entrenar Generador
        print("  🔴 Entrenando Generador:")
        
        # 2a. Generar nuevas imágenes falsas
        z = generator.generate_noise(batch_size)
        batch_fake = generator.forward(z)
        
        # 2b. Obtener predicción de D (sin actualizar D)
        D_fake_for_G = discriminator.forward(batch_fake)
        
        # 2c. Calcular pérdida de G
        g_loss = generator_loss(D_fake_for_G)
        
        print(f"     D(fake): {D_fake_for_G.mean():.3f}")
        print(f"     Pérdida G: {g_loss:.4f}")
        
        # 2d. [BACKPROP AQUÍ] Actualizar pesos de G
        # generator.pesos -= learning_rate * gradientes
    
    print(f"\n💡 Notas importantes:")
    print(f"   1. D y G se entrenan alternadamente")
    print(f"   2. D entrena primero (necesita ser un buen juez)")
    print(f"   3. G se entrena manteniendo D fijo")
    print(f"   4. El balance entre D y G es crucial")
    print(f"   5. En práctica, usar frameworks con autograd (PyTorch/TF)")

# Ejecutar demo
entrenar_gan_conceptual(generator, discriminator, real_images)
```

**Actividad 3.4**: Dibuja un diagrama del flujo de entrenamiento GAN mostrando cuándo se actualizan D y G.

### 3.6 Problemas Comunes en GANs

```python
def demostrar_problemas_gan():
    """
    Ilustra problemas comunes en el entrenamiento de GANs.
    """
    print("\n" + "="*70)
    print("5. PROBLEMAS COMUNES EN GANS")
    print("="*70)
    
    print(f"\n⚠️  1. MODE COLLAPSE")
    print(f"   Síntoma: G genera solo unas pocas variaciones")
    print(f"   Ejemplo: Solo genera el dígito '1' repetidamente")
    print(f"   Causa: G encuentra un 'truco' para engañar a D")
    print(f"   Solución: Minibatch discrimination, Unrolled GAN")
    
    print(f"\n⚠️  2. VANISHING GRADIENTS")
    print(f"   Síntoma: D se vuelve demasiado bueno, G no aprende")
    print(f"   Ejemplo: D siempre dice 0 para fake → log(0) = -∞")
    print(f"   Causa: D discrimina perfectamente muy rápido")
    print(f"   Solución: Gradient penalty (WGAN-GP), ajustar learning rates")
    
    print(f"\n⚠️  3. INESTABILIDAD")
    print(f"   Síntoma: Pérdidas oscilan violentamente")
    print(f"   Ejemplo: D loss oscila entre 0.1 y 5.0")
    print(f"   Causa: Balance incorrecto entre D y G")
    print(f"   Solución: Ajustar learning rates, arquitecturas (DCGAN)")
    
    print(f"\n⚠️  4. CONVERGENCIA LENTA")
    print(f"   Síntoma: Requiere miles de épocas")
    print(f"   Ejemplo: Imágenes siguen borrosas después de 100 épocas")
    print(f"   Causa: Problema difícil, espacio de búsqueda enorme")
    print(f"   Solución: Paciencia, mejor arquitectura, pre-training")
    
    # Visualizar mode collapse
    print(f"\n🎨 Simulación de Mode Collapse:")
    
    # Generar múltiples muestras
    z = generator.generate_noise(batch_size=20)
    fakes = generator.forward(z)
    
    # Calcular varianza (baja varianza = mode collapse)
    varianza = np.var(fakes, axis=1).mean()
    
    print(f"   Varianza promedio: {varianza:.6f}")
    print(f"   {'✅ Diversidad saludable' if varianza > 0.01 else '❌ Posible mode collapse'}")

demostrar_problemas_gan()
```

**Actividad 3.5**: Investiga qué es WGAN-GP y cómo mejora la estabilidad del entrenamiento.

### 3.7 Visualización de Progreso

```python
def visualizar_progreso_gan():
    """
    Visualiza el progreso del GAN durante entrenamiento (simulado).
    """
    print("\n" + "="*70)
    print("6. VISUALIZACIÓN DE PROGRESO")
    print("="*70)
    
    # Simular épocas de entrenamiento
    epocas = [0, 10, 50, 100]
    
    fig, axes = plt.subplots(len(epocas), 5, figsize=(12, 10))
    
    for i, epoca in enumerate(epocas):
        # Simular mejora progresiva (solo para demo)
        # En realidad, necesitarías guardar checkpoints durante entrenamiento
        
        z = generator.generate_noise(batch_size=5)
        fakes = generator.forward(z)
        
        # Añadir ruido que disminuye con las épocas (simula mejora)
        noise_level = 1.0 - (epoca / 100)
        fakes_noisy = fakes + np.random.randn(*fakes.shape) * noise_level * 0.3
        fakes_noisy = np.clip(fakes_noisy, -1, 1)
        
        for j in range(5):
            img = (fakes_noisy[:, j] + 1) / 2  # Convertir [-1,1] a [0,1]
            img = img.reshape(8, 8)
            
            axes[i, j].imshow(img, cmap='gray', vmin=0, vmax=1)
            axes[i, j].axis('off')
            
            if j == 0:
                axes[i, j].set_ylabel(f'Época {epoca}', fontsize=10)
    
    plt.suptitle('Progreso del Generador (Simulado)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('gan_progress.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n💡 En entrenamiento real:")
    print(f"   - Época 0: Ruido aleatorio")
    print(f"   - Época 10: Formas borrosas")
    print(f"   - Época 50: Dígitos reconocibles")
    print(f"   - Época 100: Dígitos claros y variados")

visualizar_progreso_gan()
```

### 3.8 Estrategias de Entrenamiento

```python
def estrategias_entrenamiento_gan():
    """
    Mejores prácticas para entrenar GANs.
    """
    print("\n" + "="*70)
    print("7. ESTRATEGIAS DE ENTRENAMIENTO")
    print("="*70)
    
    estrategias = {
        "1. Label Smoothing": {
            "desc": "Usar etiquetas 0.9 en vez de 1.0 para reales",
            "ventaja": "Previene overconfidence del discriminador",
            "codigo": "labels_real = 0.9, labels_fake = 0.0"
        },
        "2. Entrenar D más veces": {
            "desc": "Entrenar D k veces por cada entrenamiento de G",
            "ventaja": "D se mantiene más fuerte, da mejor señal a G",
            "codigo": "for _ in range(k): train_discriminator()"
        },
        "3. Learning Rate Diferente": {
            "desc": "lr_D = 0.0002, lr_G = 0.0001",
            "ventaja": "Controla el balance entre D y G",
            "codigo": "optimizer_D = Adam(lr=0.0002), optimizer_G = Adam(lr=0.0001)"
        },
        "4. Batch Normalization": {
            "desc": "Normalizar activaciones en cada capa",
            "ventaja": "Estabiliza entrenamiento, acelera convergencia",
            "codigo": "En frameworks: nn.BatchNorm2d() en PyTorch"
        },
        "5. LeakyReLU": {
            "desc": "Usar LeakyReLU en lugar de ReLU",
            "ventaja": "Evita neuronas muertas, mejora gradientes",
            "codigo": "activation = LeakyReLU(0.2)"
        },
        "6. Progressive Growing": {
            "desc": "Empezar con imágenes pequeñas, crecer gradualmente",
            "ventaja": "Entrenamiento más estable, mejor calidad final",
            "codigo": "Técnica avanzada (ProGAN, StyleGAN)"
        }
    }
    
    for nombre, info in estrategias.items():
        print(f"\n{nombre}:")
        print(f"   📝 {info['desc']}")
        print(f"   ✅ {info['ventaja']}")
        print(f"   💻 {info['codigo']}")
    
    print(f"\n🎯 Recomendación para principiantes:")
    print(f"   1. Empezar con arquitectura DCGAN (probada y estable)")
    print(f"   2. Usar Adam optimizer con lr=0.0002, beta1=0.5")
    print(f"   3. Normalizar imágenes a [-1, 1]")
    print(f"   4. Añadir ruido a las etiquetas (label smoothing)")
    print(f"   5. Monitorear D_loss y G_loss constantemente")

estrategias_entrenamiento_gan()
```

### Actividades

**Actividad 3.6**: Implementa una función que calcule el "equilibrio" entre D y G basándose en sus pérdidas.

**Actividad 3.7**: Crea una visualización que muestre cómo D y G compiten a lo largo del tiempo (gráfico de pérdidas).

**Actividad 3.8**: Implementa `conditional_gan_sketch()` que muestre cómo añadir condicionamiento (ej: generar un dígito específico).

**Actividad 3.9**: Compara la arquitectura de tu GAN con DCGAN. ¿Qué diferencias encuentras?

### Preguntas de Reflexión

**Pregunta 3.3 (Concebir)**: ¿En qué aplicaciones reales sería más útil un GAN que un VAE? ¿Y viceversa?

**Pregunta 3.4 (Diseñar)**: Si D llega a 100% de precisión muy rápido, ¿qué ajustes harías para balancear el entrenamiento?

**Pregunta 3.5 (Implementar)**: ¿Por qué usamos LeakyReLU en GANs en lugar de ReLU estándar?

**Pregunta 3.6 (Operar)**: En un sistema de generación de rostros en producción, ¿cómo detectarías y manejarías mode collapse?


---

## 🚀 Desafíos Avanzados

### Desafío 1: VAE con Framework Moderno (PyTorch/TensorFlow)

**Objetivo**: Implementar y entrenar un VAE completo usando un framework moderno.

**Requisitos**:
- Implementar VAE completo con backpropagation automático
- Entrenar en MNIST durante 20 épocas
- Visualizar reconstrucciones a lo largo del entrenamiento
- Generar 100 nuevas muestras
- Crear un latent space walk (visualización 2D)

**Criterios de éxito**:
- Pérdida de reconstrucción < 100
- Dígitos generados reconocibles
- Espacio latente con clusters claros

**Pista**:
```python
# En PyTorch
import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(...)
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        # Decoder
        self.decoder = nn.Sequential(...)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    # ... forward, loss, etc.
```

### Desafío 2: GAN Simple Funcional

**Objetivo**: Entrenar un GAN simple que genere dígitos MNIST reconocibles.

**Requisitos**:
- Implementar Generador y Discriminador en PyTorch/TensorFlow
- Entrenar durante 50 épocas
- Guardar imágenes generadas cada 10 épocas
- Implementar al menos 2 estrategias de estabilización
- Graficar pérdidas de D y G a lo largo del tiempo

**Criterios de éxito**:
- Al menos 50% de dígitos generados son reconocibles
- Pérdidas de D y G se estabilizan (no divergen)
- Diversidad: genera múltiples tipos de dígitos

### Desafío 3: Comparación VAE vs GAN

**Objetivo**: Comparar objetivamente VAE y GAN en la misma tarea.

**Requisitos**:
- Entrenar ambos modelos en MNIST
- Generar 100 muestras de cada modelo
- Comparar métricas:
  - Calidad visual (evaluación humana)
  - Diversidad (varianza de píxeles)
  - Tiempo de entrenamiento
  - Estabilidad del entrenamiento
- Crear informe con visualizaciones

**Criterios de éxito**:
- Análisis cuantitativo con al menos 3 métricas
- Conclusiones claras sobre cuándo usar cada modelo
- Recomendaciones basadas en evidencia

### Desafío 4: Conditional VAE (CVAE)

**Objetivo**: Extender VAE para generar dígitos específicos bajo demanda.

**Requisitos**:
- Modificar arquitectura para incorporar etiquetas
- Entrenar CVAE en MNIST
- Generar dígitos específicos (ej: 10 muestras del "7")
- Demostrar control sobre la generación

**Criterios de éxito**:
- Puede generar dígitos específicos con >80% precisión
- Mantiene diversidad dentro de cada clase
- Latent space organizado por clases

**Pista**:
```python
# En el encoder
encoded = encoder(torch.cat([x, one_hot_label], dim=1))

# En el decoder
decoded = decoder(torch.cat([z, one_hot_label], dim=1))
```

### Desafío 5: Interpolación Avanzada

**Objetivo**: Crear interpolaciones suaves y creativas en el espacio latente.

**Requisitos**:
- Implementar interpolación lineal
- Implementar interpolación esférica (SLERP)
- Crear un "video" de interpolación (50 frames)
- Interpolar entre 3+ puntos (no solo 2)

**Criterios de éxito**:
- Transiciones suaves entre dígitos
- No hay saltos bruscos o artefactos
- Video guardado como GIF o MP4

**Fórmula SLERP**:
```python
def slerp(z1, z2, alpha):
    """Spherical linear interpolation."""
    omega = np.arccos(np.clip(np.dot(z1/np.linalg.norm(z1), 
                                     z2/np.linalg.norm(z2)), -1, 1))
    sin_omega = np.sin(omega)
    return (np.sin((1-alpha)*omega) / sin_omega * z1 + 
            np.sin(alpha*omega) / sin_omega * z2)
```

### Desafío 6: Autoencoder Denoising

**Objetivo**: Usar autoencoder para eliminar ruido de imágenes.

**Requisitos**:
- Añadir ruido gaussiano a imágenes MNIST
- Entrenar autoencoder para reconstruir originales
- Evaluar en diferentes niveles de ruido (σ = 0.1, 0.3, 0.5)
- Visualizar antes/después

**Criterios de éxito**:
- Mejora visual clara en imágenes ruidosas
- MSE de imágenes denoised < MSE de imágenes ruidosas
- Funciona con múltiples niveles de ruido

### Desafío 7: Implementación de DCGAN

**Objetivo**: Implementar la arquitectura DCGAN (Deep Convolutional GAN).

**Requisitos**:
- Usar capas convolucionales en G y D
- Seguir guías de arquitectura de DCGAN paper
- Entrenar en MNIST o Fashion-MNIST
- Generar imágenes de calidad superior a GAN simple

**Arquitectura DCGAN**:
```python
Generador:
- FC: latent_dim → 7×7×256
- ConvTranspose2d: 7×7×256 → 14×14×128
- ConvTranspose2d: 14×14×128 → 28×28×64
- ConvTranspose2d: 28×28×64 → 28×28×1

Discriminador:
- Conv2d: 28×28×1 → 14×14×64
- Conv2d: 14×14×64 → 7×7×128
- Conv2d: 7×7×128 → 4×4×256
- FC: 4×4×256 → 1
```

---

## 📊 Análisis de Resultados y Métricas

### Métricas para Modelos Generativos

**1. Inception Score (IS)**:
```python
def inception_score(generated_images, n_splits=10):
    """
    Mide calidad y diversidad de imágenes generadas.
    Requiere: modelo Inception pre-entrenado
    
    Interpretación:
    - Mayor es mejor
    - IS > 5: Buena calidad para MNIST
    - IS > 10: Excelente para ImageNet
    """
    # Implementación requiere modelo Inception
    pass
```

**2. Fréchet Inception Distance (FID)**:
```python
def frechet_inception_distance(real_images, fake_images):
    """
    Compara distribuciones de características.
    Requiere: modelo Inception pre-entrenado
    
    Interpretación:
    - Menor es mejor
    - FID < 50: Buena calidad
    - FID < 10: Excelente calidad
    """
    # Implementación requiere modelo Inception
    pass
```

**3. Reconstruction Error (VAE)**:
```python
def evaluate_reconstruction(vae, test_images):
    """
    Mide qué tan bien reconstruye el VAE.
    """
    reconstructed, mu, logvar = vae.forward(test_images)
    mse = np.mean((test_images - reconstructed) ** 2)
    return mse

# Uso
mse = evaluate_reconstruction(vae, X_test)
print(f"MSE de reconstrucción: {mse:.4f}")
```

**4. Latent Space Quality**:
```python
def evaluate_latent_space(vae, X, y):
    """
    Evalúa organización del espacio latente.
    """
    from sklearn.metrics import silhouette_score
    
    # Encodear al espacio latente
    mu, _ = vae.encoder.forward(X)
    
    # Calcular silhouette score (clustering quality)
    score = silhouette_score(mu.T, y)
    
    print(f"Silhouette Score: {score:.4f}")
    print(f"Interpretación: {score > 0.5 and 'Buena separación' or 'Pobre separación'}")
    
    return score
```

**5. Mode Coverage (GAN)**:
```python
def evaluate_mode_coverage(gan, n_samples=1000):
    """
    Evalúa si el GAN genera todas las clases (0-9).
    """
    # Generar muestras
    z = gan.generate_noise(n_samples)
    generated = gan.forward(z)
    
    # Clasificar con un clasificador pre-entrenado
    # (requiere tener un clasificador MNIST entrenado)
    predictions = classifier.predict(generated)
    
    # Contar clases únicas
    unique_classes = len(np.unique(predictions))
    
    print(f"Clases generadas: {unique_classes}/10")
    print(f"Mode Collapse: {'Sí' if unique_classes < 8 else 'No'}")
    
    return unique_classes
```

### Visualizaciones de Análisis

```python
def analisis_completo_vae(vae, X_test, y_test):
    """
    Análisis completo del VAE.
    """
    print("="*70)
    print("ANÁLISIS COMPLETO DEL VAE")
    print("="*70)
    
    # 1. Reconstrucciones
    X_recon, mu, logvar = vae.forward(X_test[:, :10])
    
    fig, axes = plt.subplots(2, 10, figsize=(15, 3))
    for i in range(10):
        axes[0, i].imshow(X_test[:, i].reshape(8, 8), cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].imshow(X_recon[:, i].reshape(8, 8), cmap='gray')
        axes[1, i].axis('off')
    axes[0, 0].set_ylabel('Original', fontsize=10)
    axes[1, 0].set_ylabel('Reconstrucción', fontsize=10)
    plt.suptitle('Reconstrucciones del VAE')
    plt.tight_layout()
    plt.savefig('vae_analysis_reconstruction.png', dpi=300)
    plt.show()
    
    # 2. Espacio latente
    mu_all, _ = vae.encoder.forward(X_test)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(mu_all[0, :], mu_all[1, :], 
                         c=y_test, cmap='tab10', alpha=0.6)
    plt.colorbar(scatter, label='Clase')
    plt.xlabel('Dimensión Latente 1')
    plt.ylabel('Dimensión Latente 2')
    plt.title('Espacio Latente del VAE')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('vae_analysis_latent.png', dpi=300)
    plt.show()
    
    # 3. Generación
    generated = vae.generate(16)
    
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(generated[:, i].reshape(8, 8), cmap='gray')
        ax.axis('off')
    plt.suptitle('Muestras Generadas')
    plt.tight_layout()
    plt.savefig('vae_analysis_generated.png', dpi=300)
    plt.show()
    
    # 4. Métricas
    mse = evaluate_reconstruction(vae, X_test)
    silhouette = evaluate_latent_space(vae, X_test, y_test)
    
    print(f"\n📊 Métricas:")
    print(f"   MSE Reconstrucción: {mse:.4f}")
    print(f"   Silhouette Score: {silhouette:.4f}")
    print(f"   Conclusión: {'Buen modelo' if mse < 0.1 and silhouette > 0.3 else 'Necesita mejoras'}")
```

---

## 🎓 Conclusiones y Reflexión Final

### Resumen de Conceptos Aprendidos

En este laboratorio has aprendido:

**1. Fundamentos de Modelos Generativos**:
- ✅ Diferencia entre modelos discriminativos y generativos
- ✅ Concepto de espacio latente y su importancia
- ✅ Compresión y reconstrucción de información

**2. Variational Autoencoders (VAE)**:
- ✅ Encoder probabilístico (μ, σ)
- ✅ Reparameterization trick para backpropagation
- ✅ Pérdida combinada: Reconstrucción + KL Divergence
- ✅ Generación de nuevas muestras desde prior N(0,1)
- ✅ Interpolación en espacio latente

**3. Generative Adversarial Networks (GAN)**:
- ✅ Arquitectura adversarial: Generador vs Discriminador
- ✅ Entrenamiento alternado y equilibrio de Nash
- ✅ Problemas comunes: mode collapse, vanishing gradients
- ✅ Estrategias de estabilización del entrenamiento
- ✅ Diferencias arquitectónicas y casos de uso

**4. Implementación Práctica**:
- ✅ Implementación desde cero de autoencoders
- ✅ Construcción de VAE completo con todas sus componentes
- ✅ Creación de GAN con generador y discriminador
- ✅ Visualización y análisis de resultados
- ✅ Evaluación de calidad de modelos generativos

### Comparación Final: VAE vs GAN

| Aspecto | VAE | GAN |
|---------|-----|-----|
| **Calidad** | Media-Alta | Muy Alta |
| **Diversidad** | Alta (sin mode collapse) | Media (riesgo de mode collapse) |
| **Estabilidad** | Alta (entrenamiento predecible) | Baja (sensible a hiperparámetros) |
| **Velocidad** | Rápida | Media |
| **Interpretabilidad** | Alta (espacio latente estructurado) | Baja (espacio latente menos estructurado) |
| **Control** | Alto (interpolación, aritmética) | Medio |
| **Dificultad** | Media | Alta |
| **Mejor para** | Compresión, interpolación, densidad | Generación de alta calidad |

**Cuándo usar VAE**:
- Necesitas espacio latente interpretable
- Quieres estabilidad de entrenamiento
- Necesitas calcular likelihood de datos
- Interpolación suave es importante
- Aplicaciones: compresión, denoising, anomaly detection

**Cuándo usar GAN**:
- Calidad visual es prioridad #1
- Tienes recursos para experimentación
- Puedes tolerar inestabilidad
- No necesitas likelihood
- Aplicaciones: generación de imágenes realistas, style transfer, super-resolución

### Evolución Histórica y Estado del Arte

**2013-2014: Nacimiento**
- VAE (Kingma & Welling, 2013)
- GAN (Goodfellow et al., 2014)
- Primeras generaciones borrosas

**2015-2017: Mejoras Arquitectónicas**
- DCGAN: Convoluciones para GANs
- ProGAN: Crecimiento progresivo
- β-VAE: Control sobre disentanglement

**2018-2020: Salto de Calidad**
- StyleGAN: Control fino sobre características
- BigGAN: Escala masiva
- VQ-VAE: Representaciones discretas

**2020-2024: Era Moderna**
- **Diffusion Models**: Nueva familia dominante
  - DALL-E 2, Stable Diffusion, Midjourney
  - Mejor calidad que GANs, más estable que GANs
- **Transformers Generativos**: GPT, DALL-E
- **Modelos Multimodales**: Texto + Imagen
- **Aplicaciones Comerciales**: Accesibles al público

**Tendencias Futuras**:
- 🔮 Generación 3D y video de alta calidad
- 🔮 Control más fino y preciso
- 🔮 Modelos más eficientes (menos parámetros)
- 🔮 Generación personalizada y adaptativa
- 🔮 Integración con otras modalidades (audio, texto, 3D)

### Consideraciones Éticas - Responsabilidad

**Potencial Positivo**:
- 🎨 **Arte y Creatividad**: Democratización de creación artística
- 🔬 **Ciencia**: Diseño de fármacos, simulaciones
- 🏥 **Medicina**: Generación de datos sintéticos (privacidad)
- 🎓 **Educación**: Contenido personalizado
- ♿ **Accesibilidad**: Generación de descripciones, traducciones

**Riesgos y Preocupaciones**:
- ⚠️ **Deepfakes**: Desinformación, manipulación
- ⚠️ **Sesgos**: Reproducción de sesgos en datos
- ⚠️ **Derechos de Autor**: ¿De quién es el arte generado?
- ⚠️ **Trabajo**: Impacto en artistas, diseñadores
- ⚠️ **Privacidad**: Generación de rostros sin consentimiento

**Principios de Uso Responsable**:

1. **Transparencia**: Siempre revelar que el contenido es generado por IA
2. **Consentimiento**: No generar contenido de personas sin permiso
3. **Verificación**: Implementar watermarking y detección
4. **Regulación**: Seguir leyes y regulaciones locales
5. **Evaluación de Impacto**: Considerar consecuencias sociales
6. **Equidad**: Detectar y mitigar sesgos
7. **Educación**: Educar al público sobre IA generativa

**Recursos sobre Ética**:
- Partnership on AI: [www.partnershiponai.org](https://www.partnershiponai.org)
- Montreal Declaration for Responsible AI
- EU AI Act
- Adobe Content Authenticity Initiative

### Próximos Pasos en tu Aprendizaje

**1. Profundizar en Teoría**:
- 📚 Leer papers originales (VAE, GAN, Diffusion)
- 📚 Estudiar matemáticas avanzadas (teoría de información, inferencia variacional)
- 📚 Explorar arquitecturas modernas (StyleGAN3, DALL-E 2)

**2. Práctica con Frameworks**:
- 💻 Implementar VAE y GAN completos en PyTorch/TensorFlow
- 💻 Entrenar en datasets complejos (CelebA, ImageNet)
- 💻 Experimentar con Stable Diffusion y Hugging Face

**3. Proyectos Aplicados**:
- 🚀 Crear generador de arte personalizado
- 🚀 Implementar style transfer avanzado
- 🚀 Desarrollar herramienta de aumento de datos
- 🚀 Contribuir a proyectos open-source

**4. Especialización**:
- 🎯 Diffusion Models (DDPM, DDIM, Score-based)
- 🎯 Transformers Generativos (GPT, DALL-E)
- 🎯 Generación 3D (NeRF, 3D-aware GANs)
- 🎯 Audio/Música generativa (WaveNet, Jukebox)

### Recursos Adicionales

**Papers Fundamentales**:
- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) - Kingma & Welling (2013)
- [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661) - Goodfellow et al. (2014)
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) - Ho et al. (2020)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Vaswani et al. (2017)

**Tutoriales y Cursos**:
- [PyTorch VAE Tutorial](https://pytorch.org/tutorials/)
- [TensorFlow GAN Guide](https://www.tensorflow.org/tutorials/generative/dcgan)
- [Hugging Face Diffusers](https://huggingface.co/docs/diffusers/)
- [Fast.ai Practical Deep Learning](https://course.fast.ai/)
- [Stanford CS236: Deep Generative Models](https://deepgenerativemodels.github.io/)

**Implementaciones de Referencia**:
- [PyTorch Examples: VAE](https://github.com/pytorch/examples/tree/master/vae)
- [PyTorch Examples: DCGAN](https://github.com/pytorch/examples/tree/master/dcgan)
- [Stable Diffusion](https://github.com/CompVis/stable-diffusion)
- [StyleGAN3](https://github.com/NVlabs/stylegan3)

**Comunidades y Foros**:
- r/MachineLearning (Reddit)
- Papers with Code
- Weights & Biases (wandb.ai)
- Hugging Face Forums

**Herramientas Prácticas**:
- [Google Colab](https://colab.research.google.com/) - GPUs gratuitas
- [Weights & Biases](https://wandb.ai/) - Tracking de experimentos
- [TensorBoard](https://www.tensorflow.org/tensorboard) - Visualización
- [Gradio](https://gradio.app/) - Demos interactivas

---

## ✅ Checklist de Verificación

Antes de dar por completado este laboratorio, verifica que puedes:

### Conceptos Teóricos
- [ ] Explicar la diferencia entre modelos discriminativos y generativos
- [ ] Describir qué es el espacio latente y por qué es importante
- [ ] Explicar el reparameterization trick y por qué es necesario
- [ ] Entender la función de pérdida del VAE (reconstrucción + KL)
- [ ] Describir cómo funciona el entrenamiento adversarial de GANs
- [ ] Identificar problemas comunes (mode collapse, vanishing gradients)

### Implementación Práctica
- [ ] Implementar un autoencoder simple desde cero
- [ ] Construir un encoder probabilístico (μ, σ)
- [ ] Implementar el reparameterization trick
- [ ] Crear un VAE completo funcional
- [ ] Implementar generador y discriminador de GAN
- [ ] Calcular pérdidas de VAE y GAN correctamente

### Experimentación
- [ ] Entrenar (o intentar entrenar) un VAE en MNIST
- [ ] Visualizar reconstrucciones y compararlas con originales
- [ ] Explorar el espacio latente en 2D
- [ ] Generar nuevas muestras desde el prior
- [ ] Realizar interpolaciones en el espacio latente
- [ ] Experimentar con hiperparámetros (latent_dim, β, etc.)

### Análisis y Evaluación
- [ ] Evaluar calidad de reconstrucciones (MSE, visual)
- [ ] Analizar organización del espacio latente
- [ ] Comparar VAE y GAN en la misma tarea
- [ ] Identificar mode collapse o problemas de entrenamiento
- [ ] Documentar experimentos y resultados

### Aplicaciones y Ética
- [ ] Identificar casos de uso apropiados para VAE vs GAN
- [ ] Entender aplicaciones reales de IA generativa
- [ ] Reconocer implicaciones éticas (deepfakes, sesgos)
- [ ] Conocer principios de uso responsable
- [ ] Saber cómo detectar contenido generado por IA

### Próximos Pasos
- [ ] Tener plan para implementar con frameworks modernos
- [ ] Conocer recursos para aprendizaje continuo
- [ ] Identificar área de especialización de interés
- [ ] Saber dónde buscar papers y código de referencia

---

## 🎉 ¡Felicitaciones!

Has completado el **Laboratorio 09: Inteligencia Artificial Generativa**.

### Lo que has logrado:

🎯 **Dominaste los fundamentos** de modelos generativos desde cero  
🧠 **Implementaste VAE completo** con todas sus componentes  
⚔️ **Construiste GAN** con entrenamiento adversarial  
🎨 **Generaste contenido nuevo** (¡aunque sea simple!)  
📊 **Analizaste y evaluaste** modelos generativos  
🤔 **Reflexionaste sobre ética** y uso responsable  

### El viaje continúa:

Este laboratorio es solo el comienzo de tu aventura en IA generativa. El campo evoluciona rápidamente:

- **2013**: VAEs generan dígitos borrosos
- **2014**: GANs prometen revolucionar generación
- **2020**: Diffusion models superan a GANs
- **2023**: ChatGPT y DALL-E son mainstream
- **2024**: Modelos multimodales, video, 3D...
- **¿2025+?**: ¡Tú puedes ser parte de la innovación!

### Mensaje final:

> "La creatividad no es dominio exclusivo de los humanos. Con IA generativa, hemos creado herramientas que pueden sorprendernos, inspirarnos y amplificar nuestra creatividad. Pero con gran poder viene gran responsabilidad. Usa estas técnicas sabiamente, éticamente, y para hacer del mundo un lugar mejor."

**¡Ahora tienes las bases para crear cosas asombrosas! 🚀**

---

**Laboratorio diseñado con 💙 para aprender Deep Learning desde cero**  
**Serie completa**: Labs 01-09 | De Neuronas a IA Generativa

**¿Dudas o feedback?** Comparte tus experimentos, proyectos y preguntas con la comunidad.

**Próximo desafío**: Implementa tu primer modelo generativo completo y entrénalo hasta que genere algo que te sorprenda. ¡El límite es tu imaginación! 🎨🤖

---

**#DeepLearning #IAGenerativa #VAE #GAN #MachineLearning #AI**
