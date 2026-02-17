# Teoría: Transformers y Mecanismos de Atención

## 1. Introducción

Los Transformers revolucionaron el procesamiento del lenguaje natural (NLP) y el aprendizaje profundo en general desde su introducción en el paper "Attention is All You Need" (Vaswani et al., 2017).

### ¿Por qué Transformers?

**Limitaciones de RNNs/LSTMs:**
- **Procesamiento secuencial**: No se pueden paralelizar, lento en GPUs
- **Dependencias de largo alcance**: Dificultad para capturar relaciones distantes
- **Gradientes**: Problemas con vanishing/exploding gradients en secuencias largas
- **Cuello de botella de memoria**: El estado oculto debe comprimir toda la información
- **Complejidad temporal**: O(n) para procesar secuencia de longitud n

**Solución: Self-Attention**
- Procesamiento paralelo de toda la secuencia
- Conexiones directas entre cualquier par de posiciones
- Captura dependencias de largo alcance eficientemente
- Escalabilidad a secuencias muy largas
- Base de modelos modernos: BERT, GPT, T5, etc.

**Aplicaciones Exitosas:**
- NLP: ChatGPT, BERT, GPT-4, traducción automática
- Visión: Vision Transformers (ViT), DALL-E, Stable Diffusion
- Audio: Whisper, MusicGen
- Biología: AlphaFold (predicción de estructura de proteínas)
- Multimodal: CLIP, Flamingo

## 2. Mecanismo de Self-Attention

### 2.1 Intuición

El **self-attention** permite que cada elemento de una secuencia "atienda" a todos los demás elementos, determinando cuáles son más relevantes para su representación.

**Ejemplo: "El gato bebió la leche porque tenía sed"**

Al procesar "tenía", el modelo debe atender a "gato" (no a "leche") para entender que el gato tenía sed.

Self-attention calcula:
- ¿Qué palabras son relevantes para cada palabra?
- ¿Cuánta "atención" dar a cada palabra?

### 2.2 Queries, Keys y Values (Q, K, V)

La atención se basa en tres conceptos tomados de sistemas de recuperación de información:

**Query (Q)**: "¿Qué estoy buscando?"
- Representa la palabra actual que queremos enriquecer
- Vector que codifica qué información necesita esta posición

**Key (K)**: "¿Qué información tengo?"
- Representa el contenido que ofrece cada palabra
- Vector que codifica qué información puede ofrecer esta posición

**Value (V)**: "La información real que proporciono"
- Representa la información actual que se transmitirá
- Vector con el contenido semántico de esta posición

**Proceso:**
1. Cada palabra se proyecta a Q, K, V mediante matrices aprendibles
2. Se calcula similaridad entre Query de una palabra y Keys de todas las demás
3. Las similaridades se normalizan (softmax) para obtener pesos de atención
4. Se calcula suma ponderada de Values usando estos pesos

**Fórmulas:**
```
Q = X · W_Q    (n × d_k)
K = X · W_K    (n × d_k)
V = X · W_V    (n × d_v)
```

Donde:
- X: matriz de entrada (n × d_model), n = longitud secuencia
- W_Q, W_K: matrices de proyección (d_model × d_k)
- W_V: matriz de proyección (d_model × d_v)
- d_model: dimensión del modelo
- d_k, d_v: dimensiones de Q/K y V

### 2.3 Scaled Dot-Product Attention

**Fórmula completa:**
```
Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V
```

**Paso a paso:**

**1. Producto punto Q·K^T** (n × n)
```
scores[i,j] = Q[i] · K[j]^T
```
- Mide compatibilidad entre query i y key j
- Valores altos = alta relevancia

**2. Escalado por √d_k**
```
scores = scores / √d_k
```
- **¿Por qué escalar?** Cuando d_k es grande, el producto punto tiene varianza d_k
- Gradientes de softmax se vuelven muy pequeños sin escalado
- √d_k normaliza la varianza a ~1

**3. Aplicar Softmax** (normalización)
```
attention_weights[i,:] = softmax(scores[i,:])
```
- Convierte scores en distribución de probabilidad
- Suma de pesos = 1 para cada posición
- attention_weights[i,j] = cuánta atención da posición i a posición j

**4. Suma ponderada de Values**
```
output[i] = Σ_j attention_weights[i,j] · V[j]
```
- Combina información de todas las posiciones
- Ponderada por relevancia (attention weights)

**Ejemplo numérico:**
```python
# Entrada: 3 palabras, dim=4
Q = [[1,0,1,0],    # "El"
     [0,1,1,0],    # "gato"
     [1,1,0,1]]    # "duerme"

K = [[1,0,1,0],
     [0,1,1,0],
     [1,1,0,1]]

# Scores = Q·K^T
scores = [[2, 1, 2],
          [1, 2, 1],
          [2, 1, 3]]

# Escalado (d_k=4, √4=2)
scores = scores / 2

# Softmax → attention_weights
# Suma ponderada de V → output
```

### 2.4 Masked Attention (para decoders)

En modelos generativos (GPT), necesitamos **masked attention** para prevenir que el modelo "vea el futuro".

**Máscara causal:**
```
Mask = [[0,  -∞, -∞],
        [0,   0, -∞],
        [0,   0,  0]]

scores_masked = scores + Mask
```

Después de softmax, -∞ se convierte en 0:
- Posición 0 solo ve posición 0
- Posición 1 ve posiciones 0,1
- Posición 2 ve posiciones 0,1,2

Esto permite entrenamiento paralelo pero genera autoregressivamente.

## 3. Multi-Head Attention

### 3.1 Motivación

**Problema con Single-Head Attention:**
- Aprende un solo tipo de relación entre palabras
- Limitado en su capacidad representacional

**Ejemplo:**
En "El perro grande corrió rápido":
- Un head puede capturar relación sintáctica (sujeto-verbo)
- Otro head puede capturar relación semántica (perro-grande)
- Otro head puede capturar relación de modificación (corrió-rápido)

**Solución: Multi-Head Attention**
- Múltiples cabezas de atención en paralelo
- Cada cabeza aprende diferentes patrones de atención
- Resultados se concatenan y proyectan

### 3.2 Arquitectura

**Proceso:**
```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W_O

donde head_i = Attention(Q·W_Q^i, K·W_K^i, V·W_V^i)
```

**Parámetros:**
- h: número de cabezas (típicamente 8 o 16)
- d_k = d_v = d_model / h (dimensión por cabeza)
- Cada cabeza tiene sus propias matrices W_Q^i, W_K^i, W_V^i
- W_O: matriz de proyección final (d_model × d_model)

**Ejemplo con h=8, d_model=512:**
- d_k = d_v = 512/8 = 64 por cabeza
- 8 cabezas procesan en paralelo
- Cada cabeza produce output de dimensión 64
- Concatenar: 8 × 64 = 512
- Proyección final W_O: 512 × 512

**Ventajas:**
- Captura múltiples tipos de relaciones simultáneamente
- Mayor capacidad representacional
- Cada cabeza se especializa en diferentes patrones
- Paralelizable (todas las cabezas se computan simultáneamente)

### 3.3 Interpretación de Cabezas

Estudios muestran que diferentes cabezas aprenden diferentes funciones:
- **Cabezas sintácticas**: Relaciones gramaticales (sujeto-verbo, modificadores)
- **Cabezas posicionales**: Atención a posiciones relativas
- **Cabezas semánticas**: Similitud de significado
- **Cabezas de correferencia**: Pronombres y sus antecedentes

## 4. Positional Encoding

### 4.1 Problema

Self-attention es **permutation-invariant**:
- No tiene noción de orden/posición
- Attention("El gato duerme") = Attention("duerme gato El")

Pero el orden es crucial en lenguaje:
- "El perro persigue al gato" ≠ "El gato persigue al perro"

### 4.2 Solución: Positional Encoding

Añadir información de posición a las embeddings de entrada:

```
Input_final = Word_Embedding + Positional_Encoding
```

### 4.3 Positional Encoding Sinusoidal

**Fórmula original de Transformer:**
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

Donde:
- pos: posición en la secuencia (0, 1, 2, ...)
- i: dimensión (0 a d_model/2)
- Dimensiones pares usan seno, impares usan coseno

**Propiedades:**
- **Determinístico**: Misma codificación para misma posición
- **Bounded**: Valores entre [-1, 1]
- **Generalización**: Puede extrapolar a secuencias más largas
- **Relaciones lineales**: PE(pos+k) es función lineal de PE(pos)

**Intuición:**
- Diferentes frecuencias para diferentes dimensiones
- Dimensiones bajas: frecuencias altas (cambio rápido)
- Dimensiones altas: frecuencias bajas (cambio lento)
- Patrón único para cada posición
- Permite al modelo aprender distancias relativas

**Ejemplo:**
```python
pos=0: [sin(0/10000^0), cos(0/10000^0), sin(0/10000^0.2), ...]
     = [0, 1, 0, 1, ...]

pos=1: [sin(1/10000^0), cos(1/10000^0), sin(1/10000^0.2), ...]
     = [0.84, 0.54, 0.01, ...]
```

### 4.4 Learned Positional Embeddings

**Alternativa:** Aprender embeddings de posición como parámetros

```python
# PyTorch
pos_embedding = nn.Embedding(max_seq_length, d_model)
```

**Ventajas:**
- Más flexible, puede adaptarse a los datos
- Usado en BERT

**Desventajas:**
- Limitado a longitud máxima vista en entrenamiento
- No generaliza a secuencias más largas

**Comparación:**
- **Sinusoidal**: Generalización, usado en Transformer original
- **Learned**: Flexibilidad, usado en BERT, GPT-2

**Alternativas modernas:**
- **RoPE (Rotary Position Embedding)**: Usado en GPT-NeoX, PaLM
- **ALiBi**: Modifica attention scores directamente
- **Relative Positional Encoding**: Codifica distancias relativas

## 5. Arquitectura Completa del Transformer

### 5.1 Transformer Encoder-Decoder Original

**Arquitectura general:**
```
Input → Encoder → Decoder → Output

Encoder: N capas idénticas
Decoder: N capas idénticas (con masked attention)
```

### 5.2 Encoder Layer

Cada capa del encoder tiene dos sub-capas:

**1. Multi-Head Self-Attention**
```
x_1 = LayerNorm(x + MultiHeadAttention(x, x, x))
```

**2. Position-wise Feed-Forward Network**
```
x_2 = LayerNorm(x_1 + FFN(x_1))
```

**Feed-Forward Network:**
```
FFN(x) = max(0, x·W_1 + b_1)·W_2 + b_2
      = ReLU(x·W_1 + b_1)·W_2 + b_2
```

Típicamente:
- W_1: (d_model × d_ff), d_ff = 4 × d_model = 2048
- W_2: (d_ff × d_model)
- Aplicada independientemente a cada posición

**Componentes clave:**
- **Residual connections**: x + Sublayer(x)
- **Layer normalization**: Estabiliza entrenamiento
- **Dropout**: Regularización

### 5.3 Decoder Layer

Cada capa del decoder tiene tres sub-capas:

**1. Masked Multi-Head Self-Attention**
```
x_1 = LayerNorm(x + MaskedMultiHeadAttention(x, x, x))
```

**2. Cross-Attention (Encoder-Decoder Attention)**
```
x_2 = LayerNorm(x_1 + MultiHeadAttention(x_1, encoder_output, encoder_output))
```
- Query: del decoder
- Key y Value: del encoder
- Permite que decoder atienda a toda la secuencia de entrada

**3. Position-wise Feed-Forward Network**
```
x_3 = LayerNorm(x_2 + FFN(x_2))
```

### 5.4 Input/Output Embeddings

**Input Processing:**
```
1. Token Embedding: Vocab → d_model
2. Positional Encoding: añadir información de posición
3. Dropout: regularización

input = Dropout(TokenEmbedding(tokens) + PositionalEncoding(positions))
```

**Output Processing:**
```
1. Linear projection: d_model → vocab_size
2. Softmax: probabilidades sobre vocabulario

logits = Linear(decoder_output)
probs = Softmax(logits)
```

### 5.5 Parámetros Típicos

**Transformer Base:**
- N = 6 capas (encoder y decoder)
- d_model = 512
- h = 8 cabezas
- d_k = d_v = 64
- d_ff = 2048
- dropout = 0.1
- ~65M parámetros

**Transformer Big:**
- N = 6 capas
- d_model = 1024
- h = 16 cabezas
- d_k = d_v = 64
- d_ff = 4096
- ~213M parámetros

**GPT-3:**
- N = 96 capas (solo decoder)
- d_model = 12288
- h = 96 cabezas
- ~175B parámetros

## 6. Variantes de Transformers

### 6.1 BERT (Encoder-only)

**Bidirectional Encoder Representations from Transformers**

**Arquitectura:**
- Solo stack de encoders (sin decoder)
- Atención bidireccional (ve contexto completo)
- Pre-entrenamiento con dos tareas:

**1. Masked Language Modeling (MLM):**
```
Input:  "El [MASK] bebió la leche"
Target: "gato"
```
- Enmascara 15% de tokens aleatoriamente
- Modelo predice tokens enmascarados
- Aprende representaciones contextuales bidireccionales

**2. Next Sentence Prediction (NSP):**
```
Sentence A: "El gato duerme."
Sentence B: "Está muy cansado."
Label: IsNext = True
```

**Parámetros BERT-Base:**
- L = 12 capas
- H = 768 (d_model)
- A = 12 cabezas
- ~110M parámetros

**Parámetros BERT-Large:**
- L = 24 capas
- H = 1024
- A = 16 cabezas
- ~340M parámetros

**Uso típico:**
```
Pre-entrenamiento: Corpus grande sin etiquetar
Fine-tuning: Tarea específica (clasificación, NER, QA)
```

**Ventajas:**
- Excelente para tareas de comprensión
- Aprende representaciones ricas bidireccionales
- Transfer learning efectivo

**Aplicaciones:**
- Clasificación de texto
- Named Entity Recognition (NER)
- Question Answering
- Similitud semántica

### 6.2 GPT (Decoder-only)

**Generative Pre-trained Transformer**

**Arquitectura:**
- Solo stack de decoders (sin encoder)
- Atención causal/masked (solo ve contexto izquierdo)
- Pre-entrenamiento: Language Modeling

**Pre-entrenamiento:**
```
Input:  "El gato bebió"
Target: "la"

Input:  "El gato bebió la"
Target: "leche"
```

Predice siguiente palabra dado contexto anterior (autoregresivo).

**Evolución:**
- **GPT-1** (2018): 117M parámetros, 12 capas
- **GPT-2** (2019): 1.5B parámetros, 48 capas
- **GPT-3** (2020): 175B parámetros, 96 capas
- **GPT-4** (2023): ~1.7T parámetros (estimado)

**GPT-3 Características:**
```
- n_layers = 96
- d_model = 12288
- n_heads = 96
- d_head = 128
- context_window = 2048 tokens
- vocabulary = 50257 tokens
```

**Ventajas:**
- Excelente para generación de texto
- Few-shot learning (aprende de pocos ejemplos)
- Versatilidad sin fine-tuning

**Aplicaciones:**
- Generación de texto
- Traducción
- Resumen
- Código (Codex, GitHub Copilot)
- Chat (ChatGPT)

### 6.3 Comparación BERT vs GPT

| Aspecto | BERT | GPT |
|---------|------|-----|
| Arquitectura | Encoder-only | Decoder-only |
| Atención | Bidireccional | Causal/Unidireccional |
| Pre-entrenamiento | MLM + NSP | Language Modeling |
| Fortaleza | Comprensión | Generación |
| Contexto | Ve todo | Solo izquierda |
| Fine-tuning | Necesario | Opcional (prompting) |
| Tareas | Clasificación, NER, QA | Generación, traducción |

### 6.4 T5 (Text-to-Text)

**Text-to-Text Transfer Transformer**

**Idea central:** Todas las tareas como text-to-text
```
Traducción:
Input:  "translate English to German: Hello"
Output: "Hallo"

Clasificación:
Input:  "sentiment: This movie is great!"
Output: "positive"

Resumen:
Input:  "summarize: [long text]"
Output: "[summary]"
```

**Arquitectura:**
- Encoder-Decoder completo (como Transformer original)
- Pre-entrenado con span corruption
- ~11B parámetros (T5-11B)

### 6.5 Otros Transformers Importantes

**Encoder-Decoder:**
- **BART**: Denoising autoencoder, bueno para generación
- **mT5**: T5 multilingüe
- **PEGASUS**: Especializado en resumen

**Encoder-only:**
- **RoBERTa**: BERT mejorado (más datos, más entrenamiento)
- **ELECTRA**: Detección de tokens reemplazados
- **DeBERTa**: Disentangled attention

**Decoder-only:**
- **GPT-Neo/GPT-J**: Alternativas open-source a GPT-3
- **LLaMA**: Meta's efficient LLM
- **PaLM**: Google's 540B model

## 7. Vision Transformers (ViT)

### 7.1 Motivación

**Pregunta:** ¿Pueden Transformers reemplazar CNNs en visión?

**Respuesta (2020):** ¡Sí! "An Image is Worth 16x16 Words"

### 7.2 Arquitectura ViT

**Proceso:**

**1. Image Patching**
```
Imagen: 224×224×3
Dividir en patches: 14×14 patches de 16×16 píxeles
Flatten cada patch: 16×16×3 = 768 dimensiones
```

**2. Linear Projection**
```
Cada patch → embedding de dimensión d_model
Patches se tratan como "tokens"
```

**3. Position Embeddings**
```
Añadir embeddings de posición aprendidos
Similar a BERT
```

**4. Class Token**
```
Añadir [CLS] token especial al inicio
Su representación final se usa para clasificación
```

**5. Transformer Encoder**
```
Stack de transformer encoders estándar
Self-attention entre patches
```

**6. Classification Head**
```
MLP sobre [CLS] token
Predice clase de imagen
```

**Diagrama:**
```
[CLS] [P1] [P2] ... [P196]
  ↓     ↓    ↓   ...   ↓
  +─────+────+───...───+   (Position Embeddings)
  ↓
Transformer Encoder (×L capas)
  ↓
[CLS] output → MLP → Class probabilities
```

### 7.3 Parámetros ViT

**ViT-Base:**
- Patch size: 16×16
- d_model = 768
- Layers = 12
- Heads = 12
- ~86M parámetros

**ViT-Large:**
- Patch size: 16×16
- d_model = 1024
- Layers = 24
- Heads = 16
- ~307M parámetros

### 7.4 Ventajas de ViT

- **Escalabilidad**: Rendimiento mejora con más datos
- **Global context**: Atención en toda la imagen desde capa 1
- **Interpretabilidad**: Visualización de attention maps
- **Transferibilidad**: Pre-entrenamiento efectivo

### 7.5 Limitaciones

- **Requiere muchos datos**: CNNs tienen mejor inductive bias
- **Computacionalmente intensivo**: O(n²) para n patches
- **Sin invariancias**: No tiene invariancia a traslación/escala

### 7.6 Variantes y Mejoras

- **DeiT**: Data-efficient ViT con distillation
- **Swin Transformer**: Shifted windows, jerárquico
- **BEiT**: BERT-style pre-training para imágenes
- **MAE**: Masked Autoencoders, reconstrucción de patches

## 8. Transfer Learning y Fine-tuning

### 8.1 Paradigma

**Proceso típico:**
```
1. Pre-training: Modelo grande en datos masivos
2. Fine-tuning: Adaptar a tarea específica
```

**Ventajas:**
- Menos datos necesarios para tarea específica
- Mejor rendimiento que entrenar desde cero
- Conocimiento transferible entre tareas

### 8.2 Estrategias de Fine-tuning

**1. Feature Extraction**
```python
# Congelar encoder, solo entrenar cabeza de clasificación
for param in bert.parameters():
    param.requires_grad = False
    
classifier = nn.Linear(768, num_classes)  # Solo entrenar esto
```

**2. Full Fine-tuning**
```python
# Entrenar todos los parámetros
# Learning rate bajo para no destruir pre-entrenamiento
optimizer = AdamW(model.parameters(), lr=2e-5)
```

**3. Gradual Unfreezing**
```python
# Descongelar capas gradualmente
# Empezar por capas superiores (más específicas)
```

**4. Discriminative Learning Rates**
```python
# Learning rates diferentes por capa
# Capas bajas: lr bajo (conocimiento general)
# Capas altas: lr alto (específico de tarea)
```

### 8.3 Prompt Engineering (GPT-style)

**Zero-shot:**
```
Prompt: "Classify sentiment: This movie is terrible!"
Output: "Negative"
```

**Few-shot:**
```
Prompt: "
Classify sentiment:
Example: 'Great film!' → Positive
Example: 'Boring movie' → Negative
Now classify: 'Amazing plot!' → 
"
Output: "Positive"
```

**Instruction tuning:**
```
"You are a helpful assistant. Classify the sentiment of the following text as positive or negative: [text]"
```

### 8.4 Tareas Comunes

**Clasificación de Texto:**
```python
# BERT
[CLS] text [SEP] → Linear → Softmax → Class
```

**Named Entity Recognition:**
```python
# Token-level classification
Each token → Linear → Softmax → Entity tag
```

**Question Answering:**
```python
# Span selection
[CLS] question [SEP] context [SEP]
→ Predict start and end positions
```

**Generación de Texto:**
```python
# GPT
Input prompt → Autoregressive generation → Text
```

## 9. Visualización e Interpretabilidad de Atención

### 9.1 Attention Maps

**Visualización básica:**
```python
# Attention weights: (batch, heads, seq_len, seq_len)
# Para palabra i, mostrar atención a todas las palabras j

plt.imshow(attention_weights[0, head_idx])
plt.xlabel("Key positions")
plt.ylabel("Query positions")
```

**Interpretación:**
- Valor alto en (i,j): Token i atiende fuertemente a token j
- Diagonal: Self-attention (palabra consigo misma)
- Patrones: Revelan relaciones sintácticas y semánticas

### 9.2 Herramientas de Visualización

**1. BertViz**
```python
from bertviz import head_view, model_view

# Visualizar atención entre tokens
head_view(attention, tokens)

# Visualizar todas las capas y cabezas
model_view(attention, tokens)
```

**2. Attention Flow**
- Trazar flujo de información entre capas
- Identificar caminos de atención importantes

**3. Neuron Activation**
- Visualizar qué neuronas se activan
- Identificar especialización de cabezas

### 9.3 Patrones Comunes Descubiertos

**Patrones Sintácticos:**
- **Dependencia**: Verbos atienden a sujetos
- **Modificación**: Adjetivos atienden a sustantivos
- **Correferencia**: Pronombres atienden a antecedentes

**Patrones Posicionales:**
- **Cabezas de posición**: Atienden a posiciones fijas relativas
- **Cabezas de amplitud**: Atención a ventanas de contexto

**Patrones Semánticos:**
- **Similitud**: Palabras relacionadas se atienden
- **Negación**: Atención especial a "no", "sin", etc.

### 9.4 Limitaciones de Interpretación

- Atención no es explicación causal
- No muestra todo el flujo de información (residual connections)
- Diferentes cabezas pueden ser redundantes
- Difícil interpretar en modelos muy grandes

## 10. Aplicaciones Modernas y Estado del Arte

### 10.1 Large Language Models (LLMs)

**GPT-4 (OpenAI, 2023)**
- Multimodal (texto + imágenes)
- Razonamiento mejorado
- Menos alucinaciones
- Aplicaciones: ChatGPT, API

**PaLM 2 (Google, 2023)**
- 540B parámetros
- Multilingüe
- Razonamiento matemático
- Aplicaciones: Bard

**LLaMA 2 (Meta, 2023)**
- Open source
- 7B, 13B, 70B variantes
- Fine-tuned para chat

**Claude (Anthropic, 2023)**
- Constitutional AI
- 100K context window
- Safer responses

### 10.2 Generación de Imágenes

**DALL-E 2/3 (OpenAI)**
```
Text: "A dog astronaut eating pizza on Mars"
→ Realistic image generation
```
- Transformer + Diffusion
- Text-to-image
- Inpainting, variations

**Stable Diffusion**
- Open source
- Latent diffusion
- Text-to-image
- Imagen-to-image

**Midjourney**
- Artistic style
- High quality outputs

### 10.3 Multimodal

**CLIP (Contrastive Language-Image Pre-training)**
```
Image encoder + Text encoder
→ Shared embedding space
→ Zero-shot classification
```

**Flamingo (DeepMind)**
- Vision + Language
- Few-shot learning
- Visual question answering

**GPT-4V (Vision)**
- Multimodal GPT-4
- Image understanding
- Visual reasoning

### 10.4 Audio y Habla

**Whisper (OpenAI)**
- Speech-to-text
- Multilingüe
- Robust to accents

**MusicGen (Meta)**
- Text-to-music
- Audio continuation
- Melody conditioning

### 10.5 Ciencia

**AlphaFold 2 (DeepMind)**
- Predicción de estructura de proteínas
- Transformers + geometric deep learning
- Revolucionó biología estructural

**ESM (Meta)**
- Protein language models
- Evolutionary scale modeling
- Predicción de función proteica

### 10.6 Código

**Codex / GPT-4 Code**
- GitHub Copilot
- Code generation
- Debugging assistance

**StarCoder**
- Open source code LLM
- Multiple programming languages
- Code completion

## 11. Detalles de Implementación

### 11.1 Layer Normalization

**Fórmula:**
```
LN(x) = γ ⊙ (x - μ) / √(σ² + ε) + β
```

Donde:
- μ = mean(x) por ejemplo (no por batch)
- σ² = variance(x) por ejemplo
- γ, β: parámetros aprendibles
- ε: estabilidad numérica (1e-5)

**Pre-LN vs Post-LN:**

**Post-LN (original):**
```
x = x + LN(Sublayer(x))
```

**Pre-LN (más estable):**
```
x = x + Sublayer(LN(x))
```

### 11.2 Warmup Learning Rate Schedule

**Problema:** Gradientes inestables al inicio

**Solución:**
```
lr(step) = d_model^(-0.5) · min(step^(-0.5), step · warmup_steps^(-1.5))
```

**Fases:**
1. Warmup: Aumentar linealmente lr (0 a max_lr)
2. Decay: Decrecer inversamente con √step

**Típico:**
- warmup_steps = 4000
- Estabiliza entrenamiento inicial

### 11.3 Label Smoothing

**Problema:** Overconfidence en predicciones

**Solución:**
```
y_smooth = y · (1 - ε) + ε / K
```

Donde:
- y: one-hot label
- ε: smoothing factor (0.1)
- K: número de clases

**Efecto:**
- Previene overconfidence
- Mejora generalización
- Target: 0.9 para clase correcta, 0.1/K para otras

### 11.4 Gradient Clipping

**Prevenir gradientes explosivos:**
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 11.5 Optimización

**Adam/AdamW:**
- Adam: Adaptive learning rates
- AdamW: Weight decay corregido

**Parámetros típicos:**
```python
optimizer = AdamW(
    params,
    lr=5e-5,
    betas=(0.9, 0.98),  # Momentum
    eps=1e-9,
    weight_decay=0.01
)
```

## 12. Complejidad Computacional

### 12.1 Self-Attention

**Complejidad:**
- Tiempo: O(n² · d)
- Memoria: O(n²)

Donde n = longitud de secuencia, d = dimensión del modelo

**Problema:** Cuadrático en longitud de secuencia
- GPT-3: 2048 tokens → 2048² = 4M operaciones
- Documentos largos: inviable

### 12.2 Alternativas Eficientes

**1. Sparse Attention**
- Atender solo a subconjunto de posiciones
- Patrones: local, strided, random
- Usado en: Sparse Transformer, Longformer

**2. Linformer**
- Proyectar K, V a dimensión baja
- Complejidad: O(n · d)

**3. Performer**
- Aproximar attention con kernel methods
- Complejidad: O(n · d)

**4. Flash Attention**
- Optimización IO-aware
- Más rápido sin cambiar algoritmo

**5. Sliding Window**
- Atención local en ventana
- Usado en: Longformer, BigBird

### 12.3 Comparación RNN vs Transformer

| Aspecto | RNN | Transformer |
|---------|-----|-------------|
| Complejidad por capa | O(n·d²) | O(n²·d) |
| Operaciones secuenciales | O(n) | O(1) |
| Camino máximo | O(n) | O(1) |
| Paralelización | No | Sí |
| Dependencias largas | Difícil | Fácil |

## Resumen

**Transformers revolucionaron Deep Learning:**

✅ **Self-Attention**: Captura dependencias globales eficientemente
✅ **Paralelización**: Entrenamiento más rápido que RNNs
✅ **Escalabilidad**: Rendimiento mejora con más datos y parámetros
✅ **Versatilidad**: NLP, visión, audio, multimodal
✅ **Transfer Learning**: Pre-entrenamiento efectivo

**Conceptos clave:**
1. **Self-Attention**: Q, K, V y scaled dot-product
2. **Multi-Head**: Múltiples patrones de atención
3. **Positional Encoding**: Información de posición
4. **Encoder-Decoder**: Arquitectura completa
5. **BERT vs GPT**: Bidireccional vs autoregresivo
6. **ViT**: Transformers para visión
7. **Fine-tuning**: Adaptación a tareas específicas

**Aplicaciones modernas:**
- ChatGPT, GPT-4, Claude
- BERT, RoBERTa para NLU
- DALL-E, Stable Diffusion para imágenes
- Whisper para audio
- AlphaFold para ciencia

¡Los Transformers son la arquitectura fundamental del Deep Learning moderno!
