# Lab 09: IA Generativa

## Objetivos
1. Comprender modelos generativos vs discriminativos
2. Entender arquitecturas VAE y GAN
3. Implementar VAE simple
4. Explorar espacio latente
5. Generar nuevas muestras

## Estructura
```
Lab08_IA_Generativa/
├── README.md
├── teoria.md
├── practica.ipynb
└── codigo/
    └── generativo.py
```

## Modelos Generativos

### Discriminativos (lo que vimos):
- Aprenden P(y|X)
- Clasifican, predicen
- Ejemplo: "¿Es un gato?"

### Generativos (nuevos):
- Aprenden P(X)
- Crean contenido nuevo
- Ejemplo: "Genera un gato"

## Principales Arquitecturas

### 1. Autoencoder (AE)
```
X → [Encoder] → z → [Decoder] → X'
```
- Compresión y reconstrucción
- No genera bien cosas nuevas

### 2. Variational Autoencoder (VAE)
```
X → [Encoder] → μ, σ → sample z → [Decoder] → X'
```
- Espacio latente probabilístico
- Genera nuevas muestras
- Loss: Reconstruction + KL divergence

### 3. GAN (Generative Adversarial Network)
```
Generator:      z → G(z) → fake image
Discriminator:  image → real/fake
```
- Competencia adversarial
- Imágenes muy realistas
- Entrenamiento complejo

### 4. Diffusion Models
```
Forward:  X → ... → ruido
Backward: ruido → ... → X
```
- Muy alta calidad
- DALL-E, Stable Diffusion

## Práctica

### Ejecutar:
```bash
cd codigo/
python generativo.py
```

### Notebook:
```bash
jupyter notebook practica.ipynb
```

## Conceptos Clave

### Espacio Latente
Representación comprimida de datos:
- Continuidad
- Interpolación
- Control semántico

### Reparameterization Trick (VAE)
```python
z = μ + σ * ε  # donde ε ~ N(0,1)
```
Permite backpropagation a través de sampling.

### Adversarial Training (GAN)
```python
# Entrenar D
loss_D = -log(D(real)) - log(1 - D(fake))

# Entrenar G
loss_G = -log(D(fake))
```

## Aplicaciones

- **Imágenes**: Arte, diseño, edición
- **Texto**: ChatGPT, escritura
- **Audio**: Síntesis de voz, música
- **Video**: Deepfakes, animación
- **Ciencia**: Diseño molecular

## Ejercicios

1. Entrenar VAE en MNIST
2. Explorar espacio latente
3. Generar interpolaciones
4. Implementar GAN simple (con framework)

## Ética

**Considerar**:
- Deepfakes y desinformación
- Sesgos en datos de entrenamiento
- Derechos de autor
- Uso responsable

## Frameworks Recomendados

Para IA generativa seria:
```bash
# PyTorch
pip install torch torchvision

# TensorFlow
pip install tensorflow

# Hugging Face (modelos pre-entrenados)
pip install transformers diffusers
```

## Modelos Pre-entrenados

- **Stable Diffusion**: Texto → Imagen
- **GPT**: Generación de texto
- **DALL-E**: Texto → Imagen
- **StyleGAN**: Generación de caras

## Verificación
- [ ] Entiendo diferencia generativo vs discriminativo
- [ ] Conozco arquitecturas VAE y GAN
- [ ] Puedo implementar VAE básico
- [ ] Entiendo espacio latente
- [ ] Sé sobre aplicaciones y ética

## Recursos

### Papers
- VAE: "Auto-Encoding Variational Bayes" (2013)
- GAN: "Generative Adversarial Networks" (2014)
- Diffusion: "Denoising Diffusion Probabilistic Models" (2020)

### Tutoriales
- PyTorch GAN Tutorial
- TensorFlow VAE Guide
- Hugging Face Diffusers

## Conclusión

Has completado el curso de **Redes Neuronales desde Cero**:

1. ✓ Fundamentos de neuronas
2. ✓ Arquitecturas de redes
3. ✓ Funciones de activación
4. ✓ Funciones de pérdida
5. ✓ Backpropagation
6. ✓ Entrenamiento completo
7. ✓ Frameworks modernos
8. ✓ IA Generativa

**¡Ahora tienes las bases para deep learning! 🎓**

---

**El futuro es generativo - úsalo responsablemente! 🎨🤖**
