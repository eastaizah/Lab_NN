# Teoría: IA Generativa

## Introducción

La **Inteligencia Artificial Generativa** es una rama del deep learning que se enfoca en **crear** contenido nuevo en lugar de simplemente clasificarlo o predecirlo. Modelos generativos pueden crear imágenes, texto, música, videos y más.

## ¿Qué es un Modelo Generativo?

### Modelos Discriminativos vs Generativos

**Modelos Discriminativos** (lo que hemos visto):
- Aprenden P(y|X): probabilidad de la etiqueta dado los datos
- Ejemplos: Clasificación de imágenes, detección de objetos
- Pregunta: "¿Es esto un gato o un perro?"

**Modelos Generativos**:
- Aprenden P(X): la distribución de los datos
- O P(X|y): datos condicionados a etiquetas
- Pueden generar nuevas muestras
- Pregunta: "¿Puedes crear una imagen de un gato?"

## Tipos de Modelos Generativos

### 1. Autoencoders (AE)

**Concepto**: Codificar datos a espacio latente comprimido y reconstruir.

**Arquitectura**:
```
Input → [Encoder] → Latent Space → [Decoder] → Output
  X   →    φ(X)    →      z      →    ψ(z)   →   X'
```

**Objetivo**: Minimizar error de reconstrucción
```
Loss = ||X - X'||²
```

**Aplicaciones**:
- Reducción de dimensionalidad
- Denoising (eliminación de ruido)
- Compresión

**Limitación**: No genera datos nuevos muy bien

### 2. Variational Autoencoders (VAE)

**Concepto**: Autoencoders con espacio latente probabilístico.

**Diferencia clave**: En lugar de codificar a un punto z, codifica a una **distribución** p(z).

**Arquitectura**:
```
X → [Encoder] → μ, σ → Sample z ~ N(μ, σ²) → [Decoder] → X'
```

**Loss function**:
```
Loss = Reconstruction Loss + KL Divergence
     = ||X - X'||² + KL(q(z|X) || p(z))
```

Donde:
- Reconstruction: qué tan bien reconstruye
- KL Divergence: qué tan cerca está q(z|X) de prior p(z)

**Ventajas**:
- Espacio latente continuo y suave
- Puede generar nuevas muestras
- Control sobre generación

**Aplicaciones**:
- Generación de caras
- Generación de dígitos
- Interpolación entre imágenes

### 3. Generative Adversarial Networks (GANs)

**Concepto**: Dos redes compiten entre sí.

**Componentes**:

1. **Generator (Generador)**: Crea datos falsos
   ```
   z (ruido) → G(z) → imagen falsa
   ```

2. **Discriminator (Discriminador)**: Distingue real de falso
   ```
   X → D(X) → probabilidad de ser real
   ```

**Entrenamiento adversarial**:
```
while not converged:
    # Entrenar Discriminador
    - Clasificar imágenes reales como reales
    - Clasificar imágenes falsas (de G) como falsas
    
    # Entrenar Generador
    - Generar imágenes que engañen a D
```

**Función objetivo**:
```
min_G max_D E[log D(X)] + E[log(1 - D(G(z)))]
```

**Ventajas**:
- Genera imágenes de alta calidad
- No requiere emparejamiento explícito

**Desafíos**:
- Entrenamiento inestable
- Mode collapse (genera poca variedad)
- Difícil de debuggear

**Aplicaciones**:
- Generación de caras realistas
- Style transfer
- Imagen a imagen (pix2pix)
- Super-resolución

### 4. Diffusion Models

**Concepto**: Aprender a revertir un proceso de difusión (añadir ruido).

**Proceso**:

1. **Forward (difusión)**: Añadir ruido gradualmente
   ```
   X₀ → X₁ → X₂ → ... → Xₜ (puro ruido)
   ```

2. **Reverse (denoising)**: Aprender a remover ruido
   ```
   Xₜ → Xₜ₋₁ → ... → X₁ → X₀ (imagen limpia)
   ```

**Ventajas**:
- Entrenamiento más estable que GANs
- Calidad de imagen excelente
- Control flexible

**Ejemplos**:
- DALL-E 2
- Stable Diffusion
- Midjourney

### 5. Transformers Generativos

**Concepto**: Usar arquitectura transformer para generación.

**Ejemplos famosos**:
- **GPT (Generative Pre-trained Transformer)**: Texto
- **DALL-E**: Imágenes desde texto
- **Codex**: Código

**Características**:
- Autoregresivo: genera token por token
- Escalable a modelos enormes
- Few-shot learning

## Comparación de Modelos

| Modelo | Calidad | Diversidad | Estabilidad | Control | Velocidad |
|--------|---------|-----------|-------------|---------|-----------|
| **Autoencoder** | Baja | Baja | Alta | Bajo | Rápida |
| **VAE** | Media | Media | Alta | Medio | Rápida |
| **GAN** | Alta | Media | Baja | Medio | Rápida |
| **Diffusion** | Muy Alta | Alta | Alta | Alto | Lenta |
| **Transformer** | Muy Alta | Alta | Media | Alto | Media |

## Aplicaciones de IA Generativa

### 1. Generación de Imágenes
- Crear arte
- Diseño de productos
- Edición fotográfica
- Síntesis de caras

### 2. Generación de Texto
- Chatbots (ChatGPT)
- Escritura creativa
- Resúmenes
- Traducción

### 3. Generación de Audio
- Síntesis de voz
- Generación de música
- Efectos de sonido

### 4. Generación de Video
- Deepfakes
- Animación
- Efectos especiales

### 5. Diseño Molecular
- Descubrimiento de fármacos
- Diseño de proteínas

## Implementación Básica: VAE Simple

### Arquitectura

```python
class VAE:
    def __init__(self):
        # Encoder
        self.encoder = [
            Dense(128) → ReLU,
            Dense(64) → ReLU,
            Dense(latent_dim * 2)  # μ y log(σ²)
        ]
        
        # Decoder
        self.decoder = [
            Dense(64) → ReLU,
            Dense(128) → ReLU,
            Dense(input_dim) → Sigmoid
        ]
```

### Forward Pass

```python
def encode(X):
    h = encoder(X)
    μ = h[:, :latent_dim]
    log_σ² = h[:, latent_dim:]
    return μ, log_σ²

def reparameterize(μ, log_σ²):
    σ = exp(0.5 * log_σ²)
    ε = random_normal()
    z = μ + σ * ε
    return z

def decode(z):
    return decoder(z)

def forward(X):
    μ, log_σ² = encode(X)
    z = reparameterize(μ, log_σ²)
    X_reconstructed = decode(z)
    return X_reconstructed, μ, log_σ²
```

### Loss Function

```python
def vae_loss(X, X_recon, μ, log_σ²):
    # Reconstruction loss
    recon_loss = binary_crossentropy(X, X_recon)
    
    # KL divergence
    kl_loss = -0.5 * sum(1 + log_σ² - μ² - exp(log_σ²))
    
    return recon_loss + kl_loss
```

## Implementación Básica: GAN Simple

### Arquitectura

```python
class Generator:
    def __init__(self):
        self.model = [
            Dense(128) → ReLU,
            Dense(256) → ReLU,
            Dense(784) → Tanh  # Salida en [-1, 1]
        ]
    
    def forward(noise):
        return model(noise)

class Discriminator:
    def __init__(self):
        self.model = [
            Dense(256) → LeakyReLU,
            Dense(128) → LeakyReLU,
            Dense(1) → Sigmoid  # Probabilidad de ser real
        ]
    
    def forward(X):
        return model(X)
```

### Entrenamiento

```python
for epoch in range(epochs):
    # 1. Entrenar Discriminador
    real_data = sample_real_data()
    fake_data = generator(random_noise())
    
    d_loss_real = -log(discriminator(real_data))
    d_loss_fake = -log(1 - discriminator(fake_data))
    d_loss = d_loss_real + d_loss_fake
    
    update(discriminator, d_loss)
    
    # 2. Entrenar Generador
    fake_data = generator(random_noise())
    g_loss = -log(discriminator(fake_data))
    
    update(generator, g_loss)
```

## Conceptos Avanzados

### Latent Space (Espacio Latente)

**Definición**: Representación comprimida y continua de datos.

**Propiedades deseables**:
- **Continuidad**: Puntos cercanos → salidas similares
- **Completitud**: Cualquier punto → salida válida
- **Disentanglement**: Cada dimensión controla un factor

**Aplicaciones**:
- Interpolación
- Manipulación semántica
- Exploración de variedades

### Conditional Generation

**Concepto**: Generar condicionado a información adicional.

```
z, y (etiqueta) → Generator → imagen de clase y
```

**Ejemplos**:
- "Genera un dígito 7"
- "Genera una cara rubia"
- Text-to-image: "Un gato tocando piano"

### Mode Collapse (en GANs)

**Problema**: El generador produce poca variedad.

**Síntoma**: Todas las muestras se parecen

**Soluciones**:
- Minibatch discrimination
- Unrolled GANs
- Gradient penalties (WGAN)

## Evaluación de Modelos Generativos

### Métricas

1. **Inception Score (IS)**
   - Evalúa calidad y diversidad
   - Basado en clasificador pre-entrenado

2. **Fréchet Inception Distance (FID)**
   - Compara distribuciones de características
   - Más bajo = mejor

3. **Evaluación Humana**
   - A/B testing
   - Encuestas de calidad

### Desafíos

- No hay métrica perfecta
- Trade-off calidad vs diversidad
- Depende de la aplicación

## Ética y Consideraciones

### Riesgos

1. **Deepfakes**: Desinformación
2. **Sesgos**: Reproducir sesgos de datos
3. **Derechos de autor**: ¿De quién es el arte generado?
4. **Uso malicioso**: Falsificación, fraude

### Mitigaciones

- Watermarking
- Detección de contenido generado
- Transparencia
- Regulación
- Educación pública

## Futuro de IA Generativa

### Tendencias

1. **Multimodalidad**: Texto + imagen + audio
2. **Personalización**: Modelos personalizados
3. **Eficiencia**: Modelos más pequeños y rápidos
4. **Control**: Mejor control sobre generación
5. **Democratización**: Acceso más amplio

### Áreas Emergentes

- Generación 3D
- Video de alta calidad
- Diseño molecular
- Código (GitHub Copilot)

## Recursos de Aprendizaje

### Papers Fundamentales

- **VAE**: "Auto-Encoding Variational Bayes" (Kingma & Welling, 2013)
- **GAN**: "Generative Adversarial Networks" (Goodfellow et al., 2014)
- **Diffusion**: "Denoising Diffusion Probabilistic Models" (Ho et al., 2020)

### Tutoriales

- PyTorch GAN Tutorial
- TensorFlow VAE Guide
- Hugging Face Diffusers

## Resumen

**IA Generativa** es fascinante porque:
- Crea en lugar de clasificar
- Tiene aplicaciones creativas
- Está en rápida evolución

**Modelos principales**:
- **VAE**: Espacio latente probabilístico
- **GAN**: Competición adversarial
- **Diffusion**: Proceso de denoising
- **Transformers**: Generación autoregresiva

**Clave**: Balance entre calidad, diversidad y control

---

**¡El futuro es generativo! 🎨🤖**
