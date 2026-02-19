# Guía de Laboratorio: Transformers y Mecanismos de Auto-Atención
## 📋 Información del Laboratorio
**Título:** Transformers, Self-Attention y Modelos de Lenguaje Pre-entrenados  
**Código:** Lab 12  
**Duración:** 4-5 horas  
**Nivel:** Avanzado  
## 🎯 Objetivos Específicos
Al completar este laboratorio, serás capaz de:
1. Comprender y derivar el mecanismo de **Self-Attention** con matrices Query, Key y Value
2. Implementar **Scaled Dot-Product Attention** desde cero con NumPy
3. Construir **Multi-Head Attention** con múltiples cabezas en paralelo
4. Aplicar **Positional Encoding** sinusoidal para codificar información de posición
5. Ensamblar un **Transformer Encoder Block** completo con PyTorch
6. Aplicar **fine-tuning** de BERT para tareas de clasificación de sentimientos
7. Generar texto de forma autoregresiva con **GPT-2**
8. Visualizar e interpretar **mapas de atención**
9. Comparar la complejidad computacional de Transformers vs. RNNs/LSTMs
10. Reconocer el ecosistema de **modelos pre-entrenados** y cuándo utilizarlos
## 📚 Prerrequisitos
### Conocimientos
- Python intermedio-avanzado (clases, decoradores, comprensión de listas)
- Álgebra lineal (multiplicación matricial, transpuesta, softmax)
- Redes neuronales feedforward (Lab 01–06)
- Frameworks de Deep Learning — PyTorch (Lab 08)
- Redes Recurrentes y LSTMs (Lab 11) — comparativa clave
- Conceptos de NLP: tokens, embeddings, vocabulario
### Software
- Python 3.8+
- NumPy 1.21+
- PyTorch 1.12+
- Matplotlib 3.4+
- Transformers (Hugging Face) 4.20+
- Datasets (Hugging Face) 2.0+
- Jupyter Notebook (recomendado)
```bash
pip install numpy matplotlib torch transformers datasets sentencepiece
```
### Material de Lectura
Antes de comenzar, lee:
- `teoria.md` — Marco teórico completo sobre Transformers y Self-Attention
- `README.md` — Estructura del laboratorio y recursos disponibles
- **Vaswani et al. (2017)** — "Attention Is All You Need" (abstract y figuras 1-2)
## 📖 Introducción
Los **Transformers** representan el avance más significativo en Deep Learning de la última década. Introducidos en 2017 con el paper "Attention Is All You Need" (Vaswani et al.), reemplazaron a las redes recurrentes en prácticamente todas las tareas de procesamiento de lenguaje natural y están expandiéndose a visión computacional, audio, bioinformática y más. Modelos como GPT-4, BERT, DALL-E, Whisper y AlphaFold están todos construidos sobre esta arquitectura.
### Contexto del Problema: Las Limitaciones de las RNNs
En el Lab 11 trabajaste con RNNs y LSTMs. Estas arquitecturas procesan secuencias **token por token**, lo que genera tres problemas fundamentales:
1. **Procesamiento secuencial**: No es posible paralelizar — el token en la posición *t* depende del estado oculto de la posición *t-1*. Esto hace el entrenamiento lento.
2. **Cuello de botella de información**: Toda la información de una secuencia larga debe comprimirse en un único vector de estado oculto de dimensión fija.
3. **Gradientes que desaparecen o explotan**: Aunque las LSTMs mitigan este problema con compuertas, no lo eliminan completamente en secuencias muy largas (> 500 tokens).
```
RNN/LSTM (secuencial — lento):
x₁ → [h₁] → x₂ → [h₂] → x₃ → [h₃] → ... → xₙ → [hₙ] → salida

Transformer (paralelo — rápido):
x₁ ─┐
x₂ ─┤──> [Self-Attention] ──> [FFN] ──> salida₁, salida₂, ..., salida_n
x₃ ─┤        (todos al mismo tiempo)
xₙ ─┘
```
### La Solución: Mecanismo de Atención
La idea clave de los Transformers es el **mecanismo de atención**: en lugar de pasar información a través de estados ocultos secuenciales, cada posición de la secuencia puede "atender" directamente a **cualquier otra posición** con un coste O(1) en profundidad.
**Analogía de búsqueda en base de datos:**
Imagina que tienes una base de datos con entradas (Key → Value). Cuando lanzas una consulta (Query), obtienes como resultado una combinación ponderada de todos los valores, donde el peso de cada uno depende de cuán compatible es tu consulta con esa clave.
```
Query: "¿quién está hambriento?"
Keys:  ["gato", "leche", "bowl"]
─────────────────────────────────────────
Score("gato")  = 0.85  ← Alta compatibilidad
Score("leche") = 0.10
Score("bowl")  = 0.05
─────────────────────────────────────────
Atención ≈ 0.85 × V("gato") + 0.10 × V("leche") + 0.05 × V("bowl")
```
### Enfoque con Transformers
La arquitectura Transformer reemplaza la recurrencia con tres componentes clave:
```
INPUT TOKENS (x₁, x₂, ..., xₙ)
        ↓
TOKEN EMBEDDINGS + POSITIONAL ENCODING
        ↓
┌─────────────────────────────────────┐
│     TRANSFORMER ENCODER BLOCK        │  × N capas
│  ┌─────────────────────────────────┐ │
│  │  Multi-Head Self-Attention      │ │
│  │  (Q, K, V desde la misma seq.)  │ │
│  └────────────────┬────────────────┘ │
│        Add & LayerNorm               │
│  ┌─────────────────────────────────┐ │
│  │  Feed-Forward Network (FFN)     │ │
│  │  (proyección lineal × 2 + ReLU) │ │
│  └────────────────┬────────────────┘ │
│        Add & LayerNorm               │
└─────────────────────────────────────┘
        ↓
REPRESENTACIONES CONTEXTUALES
```
### Conceptos Fundamentales
**1. Scaled Dot-Product Attention:**
$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$
Donde:
- **Q** (Queries): lo que queremos buscar — forma `(seq_len, d_k)`
- **K** (Keys): lo que se puede encontrar — forma `(seq_len, d_k)`
- **V** (Values): contenido a recuperar — forma `(seq_len, d_v)`
- **√d_k**: factor de escala para estabilizar gradientes
**2. Multi-Head Attention:**
$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1,\ldots,\text{head}_h)\,W^O$$
$$\text{head}_i = \text{Attention}(QW_i^Q,\, KW_i^K,\, VW_i^V)$$
Cada cabeza aprende a atender diferentes tipos de relaciones (sintácticas, semánticas, de posición, etc.).
**3. Positional Encoding Sinusoidal:**
$$\text{PE}(\text{pos},\, 2i) = \sin\!\left(\frac{\text{pos}}{10000^{2i/d_{\text{model}}}}\right)$$
$$\text{PE}(\text{pos},\, 2i+1) = \cos\!\left(\frac{\text{pos}}{10000^{2i/d_{\text{model}}}}\right)$$
### Aplicaciones Prácticas
Los Transformers son la base de la IA moderna:
- **NLP**: GPT-4 (generación), BERT (comprensión), T5 (texto-a-texto), Llama (open-source)
- **Visión**: ViT, DINO, Segment Anything Model (SAM) — tratan parches de imagen como tokens
- **Audio**: Whisper (reconocimiento de voz), AudioLM, MusicGen
- **Ciencia**: AlphaFold2 (plegamiento de proteínas), ESMFold, modelos de diseño de fármacos
- **Multimodal**: CLIP, DALL-E 3, GPT-4V, Gemini
### Motivación Histórica
La secuencia de hitos que llevó a los Transformers modernos:
- **1986** — Backpropagation (Rumelhart et al.)
- **1997** — LSTM (Hochreiter & Schmidhuber)
- **2014** — Mecanismo de atención para traducción (Bahdanau et al.)
- **2017** — "Attention Is All You Need" — el Transformer original (Vaswani et al.)
- **2018** — BERT (Google) y GPT (OpenAI) — pre-entrenamiento masivo
- **2020** — GPT-3 (175B parámetros) — few-shot learning emergente
- **2022** — ChatGPT — RLHF aplicado a GPT
- **2023+** — GPT-4, Llama 2/3, Gemini, Claude 3 — era de los LLMs
## 🔬 Parte 1: Self-Attention desde Cero con NumPy (45 min)
### 1.1 Introducción Conceptual
El mecanismo de Self-Attention permite a cada token de una secuencia calcular su representación como una suma ponderada de **todos los otros tokens** (incluido él mismo). La intuición es que el significado de una palabra depende de su contexto.
**Ejemplo lingüístico:**
```
"El banco estaba lleno de peces"  →  "banco" debe atender a "peces"
"El banco rechazó mi préstamo"    →  "banco" debe atender a "préstamo"
```
Self-Attention resuelve esta ambigüedad contextualmente.
### 1.2 Implementación de Scaled Dot-Product Attention
```python
import numpy as np
def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Implementa Scaled Dot-Product Attention desde cero.
    
    Args:
        Q: Matriz de Queries  (seq_len_q, d_k)
        K: Matriz de Keys     (seq_len_k, d_k)
        V: Matriz de Values   (seq_len_k, d_v)
        mask: Máscara opcional (seq_len_q, seq_len_k), -inf en posiciones a ignorar
    
    Returns:
        output: Representación atendida (seq_len_q, d_v)
        pesos:  Pesos de atención       (seq_len_q, seq_len_k)
    """
    d_k = Q.shape[-1]
    
    # Paso 1: Calcular scores de similitud Q·Kᵀ
    scores = Q @ K.T                          # (seq_len_q, seq_len_k)
    
    # Paso 2: Escalar para estabilizar gradientes
    scores = scores / np.sqrt(d_k)
    
    # Paso 3: Aplicar máscara si se proporciona (para decoder)
    if mask is not None:
        scores = scores + mask                # -inf → 0 en softmax
    
    # Paso 4: Softmax para obtener pesos de atención
    # Restar el máximo por fila para estabilidad numérica
    scores_estables = scores - np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores_estables)
    pesos = exp_scores / exp_scores.sum(axis=-1, keepdims=True)
    
    # Paso 5: Suma ponderada de Values
    output = pesos @ V                        # (seq_len_q, d_v)
    
    return output, pesos
# ───── Demostración con secuencia simple ─────
np.random.seed(42)
seq_len = 4     # "El gato bebe leche"
d_model = 8    # dimensión del embedding
# Simular embeddings de entrada (en práctica, estos vienen de la capa de embedding)
X = np.random.randn(seq_len, d_model)
# Proyecciones lineales para Q, K, V
d_k = d_v = d_model
W_Q = np.random.randn(d_model, d_k) * 0.1
W_K = np.random.randn(d_model, d_k) * 0.1
W_V = np.random.randn(d_model, d_v) * 0.1
Q = X @ W_Q    # (4, 8)
K = X @ W_K    # (4, 8)
V = X @ W_V    # (4, 8)
output, attention_weights = scaled_dot_product_attention(Q, K, V)
print(f"Input shape:           {X.shape}")
print(f"Q, K, V shapes:        {Q.shape}")
print(f"Output shape:          {output.shape}")
print(f"Attention weights:\n{np.round(attention_weights, 3)}")
print(f"\nCada fila suma a 1.0: {np.allclose(attention_weights.sum(axis=1), 1.0)}")
```
**Actividad 1.1**: Ejecuta el código y examina la matriz de pesos de atención. ¿Qué posición atiende más a sí misma? ¿Por qué tiene sentido?
**Actividad 1.2**: Modifica los embeddings de entrada de modo que los tokens 0 y 2 sean casi idénticos. ¿Cómo cambia la distribución de atención?
### 1.3 Self-Attention Completo como Clase
```python
class SelfAttentionNumPy:
    """
    Self-Attention completo con proyecciones Q, K, V aprendibles.
    """
    def __init__(self, d_model, d_k=None, d_v=None):
        self.d_model = d_model
        self.d_k = d_k or d_model
        self.d_v = d_v or d_model
        
        # Inicialización Xavier
        scale = np.sqrt(2.0 / (d_model + self.d_k))
        self.W_Q = np.random.randn(d_model, self.d_k) * scale
        self.W_K = np.random.randn(d_model, self.d_k) * scale
        self.W_V = np.random.randn(d_model, self.d_v) * scale
        self.W_O = np.random.randn(self.d_v, d_model) * scale
    
    def forward(self, X, mask=None):
        """
        Args:
            X:    (seq_len, d_model)
            mask: (seq_len, seq_len) opcional
        Returns:
            output: (seq_len, d_model)
            weights: (seq_len, seq_len)
        """
        Q = X @ self.W_Q     # (seq_len, d_k)
        K = X @ self.W_K     # (seq_len, d_k)
        V = X @ self.W_V     # (seq_len, d_v)
        
        attn_out, weights = scaled_dot_product_attention(Q, K, V, mask)
        output = attn_out @ self.W_O    # (seq_len, d_model)
        
        return output, weights
# Prueba
np.random.seed(0)
sa = SelfAttentionNumPy(d_model=16)
X_test = np.random.randn(6, 16)    # secuencia de 6 tokens
out, w = sa.forward(X_test)
print(f"Input:  {X_test.shape}")
print(f"Output: {out.shape}")
print(f"Pesos de atención (6×6):\n{np.round(w, 3)}")
```
**Actividad 1.3**: Implementa una **máscara causal** (triangular inferior) para el decoder. En un decoder autoregresivo, el token en posición *t* solo puede atender a posiciones ≤ *t*.
```python
def crear_mascara_causal(seq_len):
    """
    Crea una máscara triangular inferior para atención causal.
    Las posiciones superiores reciben -inf para ser ignoradas en softmax.
    """
    # Comienza con una matriz de ceros (posiciones permitidas)
    mask = np.zeros((seq_len, seq_len))
    # Asigna -inf a la parte triangular superior (posiciones futuras — prohibidas)
    mask[np.triu_indices(seq_len, k=1)] = -np.inf
    return mask
mascara = crear_mascara_causal(4)
print("Máscara causal (4×4):")
print(mascara)
```
### Preguntas de Reflexión
**Pregunta 1.1 (Concebir)**: ¿Por qué dividimos los scores por √d_k? ¿Qué ocurriría si no lo hiciéramos cuando d_k es grande (por ejemplo, d_k=512)?
**Pregunta 1.2 (Diseñar)**: ¿Cuál es la diferencia fundamental entre Self-Attention y la atención de Bahdanau utilizada en los seq2seq con RNN?
**Pregunta 1.3 (Implementar)**: La complejidad computacional de Self-Attention es O(n²·d). Para una secuencia de 1000 tokens con d_model=512, ¿cuántas operaciones de punto flotante implica solo el cálculo de Q·Kᵀ?
**Pregunta 1.4 (Operar)**: Si tienes una frase ambigua como "Vi a la estudiante con el telescopio", ¿cómo esperarías que se distribuyan los pesos de atención alrededor de la palabra "con"?
## 🔬 Parte 2: Multi-Head Attention (40 min)
### 2.1 Motivación: Múltiples Perspectivas
Una sola cabeza de atención solo puede enfocarse en un tipo de relación a la vez. **Multi-Head Attention** ejecuta *h* atenciones en paralelo, cada una en un subespacio diferente:
```
Cabeza 1: "¿Quién hace qué?" (relaciones sintácticas sujeto-verbo)
Cabeza 2: "¿Qué describe qué?" (adjetivos y sustantivos)
Cabeza 3: "¿Qué viene antes/después?" (dependencias posicionales)
Cabeza 4: "¿Qué co-refiere?" (pronombres y sus antecedentes)
```
### 2.2 Implementación de Multi-Head Attention
```python
class MultiHeadAttentionNumPy:
    """
    Multi-Head Attention desde cero con NumPy.
    
    Args:
        d_model: Dimensión del modelo
        num_heads: Número de cabezas de atención
    """
    def __init__(self, d_model, num_heads):
        assert d_model % num_heads == 0, \
            f"d_model ({d_model}) debe ser divisible por num_heads ({num_heads})"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads    # dimensión por cabeza
        
        # Pesos para Q, K, V de TODAS las cabezas (concatenados)
        scale = np.sqrt(2.0 / (d_model + self.d_k))
        self.W_Q = np.random.randn(d_model, d_model) * scale   # (d_model, d_model)
        self.W_K = np.random.randn(d_model, d_model) * scale
        self.W_V = np.random.randn(d_model, d_model) * scale
        self.W_O = np.random.randn(d_model, d_model) * scale   # proyección final
    
    def split_heads(self, X, seq_len):
        """
        Reorganiza X de (seq_len, d_model) a (num_heads, seq_len, d_k)
        """
        X = X.reshape(seq_len, self.num_heads, self.d_k)
        return X.transpose(1, 0, 2)    # (num_heads, seq_len, d_k)
    
    def forward(self, Q_in, K_in, V_in, mask=None):
        """
        Args:
            Q_in, K_in, V_in: (seq_len, d_model)
            mask: (seq_len, seq_len) opcional
        Returns:
            output:  (seq_len, d_model)
            weights: (num_heads, seq_len, seq_len)
        """
        seq_len = Q_in.shape[0]
        
        # Proyecciones lineales
        Q = Q_in @ self.W_Q    # (seq_len, d_model)
        K = K_in @ self.W_K
        V = V_in @ self.W_V
        
        # Dividir en cabezas: (num_heads, seq_len, d_k)
        Q = self.split_heads(Q, seq_len)
        K = self.split_heads(K, seq_len)
        V = self.split_heads(V, seq_len)
        
        # Atención por cabeza
        all_heads = []
        all_weights = []
        for i in range(self.num_heads):
            head_out, head_w = scaled_dot_product_attention(Q[i], K[i], V[i], mask)
            all_heads.append(head_out)       # (seq_len, d_k)
            all_weights.append(head_w)       # (seq_len, seq_len)
        
        # Concatenar cabezas: (seq_len, d_model)
        concatenado = np.concatenate(all_heads, axis=-1)
        
        # Proyección final
        output = concatenado @ self.W_O    # (seq_len, d_model)
        
        return output, np.array(all_weights)    # weights: (h, seq_len, seq_len)
# ─── Demostración ───
np.random.seed(7)
d_model = 32
num_heads = 4
seq_len = 5
mha = MultiHeadAttentionNumPy(d_model=d_model, num_heads=num_heads)
X = np.random.randn(seq_len, d_model)
output, weights = mha.forward(X, X, X)
print(f"Input shape:            {X.shape}")
print(f"Output shape:           {output.shape}")
print(f"Weights shape (h,s,s):  {weights.shape}")
print(f"\nPesos de la cabeza 0:\n{np.round(weights[0], 3)}")
print(f"\nPesos de la cabeza 1:\n{np.round(weights[1], 3)}")
```
**Actividad 2.1**: Compara los mapas de atención de las diferentes cabezas. ¿Observas patrones distintos entre ellas? ¿Qué sugiere esto sobre lo que aprende cada cabeza?
**Actividad 2.2**: Implementa la versión **vectorizada** del bucle por cabezas usando `np.einsum` o reordenando los tensores para evitar el loop explícito.
### 2.3 Visualización de Mapas de Atención
```python
import matplotlib.pyplot as plt
def visualizar_atencion(weights, tokens, titulo="Mapa de Atención", num_heads_mostrar=4):
    """
    Visualiza mapas de atención como heat maps.
    
    Args:
        weights: (num_heads, seq_len, seq_len)
        tokens:  lista de strings con los tokens
        titulo:  título del gráfico
    """
    h = min(num_heads_mostrar, weights.shape[0])
    fig, axes = plt.subplots(1, h, figsize=(4 * h, 4))
    if h == 1:
        axes = [axes]
    
    for i, ax in enumerate(axes):
        im = ax.imshow(weights[i], cmap='Blues', vmin=0, vmax=1)
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha='right')
        ax.set_yticklabels(tokens)
        ax.set_title(f'Cabeza {i+1}')
        plt.colorbar(im, ax=ax)
    
    plt.suptitle(titulo, fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('atencion_multihead.png', dpi=120, bbox_inches='tight')
    plt.show()
    print("✅ Guardado: atencion_multihead.png")
# Ejemplo de uso con tokens representativos
tokens_ejemplo = ["El", "gato", "bebe", "la", "leche"]
np.random.seed(42)
mha_demo = MultiHeadAttentionNumPy(d_model=16, num_heads=4)
X_demo = np.random.randn(5, 16)
_, w_demo = mha_demo.forward(X_demo, X_demo, X_demo)
visualizar_atencion(w_demo, tokens_ejemplo,
                    titulo="Multi-Head Attention — 4 Cabezas")
```
### Preguntas de Reflexión
**Pregunta 2.1 (Concebir)**: ¿Por qué se divide d_model entre el número de cabezas para obtener d_k? ¿Qué ocurriría si cada cabeza tuviera d_k = d_model completo?
**Pregunta 2.2 (Diseñar)**: En la práctica, ¿cuántas cabezas tienen BERT-base (12 capas) y GPT-2 (12 capas)? ¿Cuál es d_k en cada caso?
**Pregunta 2.3 (Implementar)**: El número total de parámetros en Multi-Head Attention es 4 × d_model². Para BERT-base (d_model=768), ¿cuántos parámetros tiene una sola capa de atención?
**Pregunta 2.4 (Operar)**: Visualiza los pesos de atención de tus 4 cabezas. ¿Alguna muestra un patrón diagonal (cada token atiende principalmente a sí mismo)? ¿Qué implicaciones tiene eso?
## 🔬 Parte 3: Positional Encoding (30 min)
### 3.1 El Problema de la Invariancia al Orden
Self-Attention es **equivariante a permutaciones**: si desordenamos los tokens de entrada, las representaciones de salida también se desordenan de la misma manera, pero no hay información sobre el orden original. Para el modelo, "El gato persigue al perro" y "El perro persigue al gato" serían equivalentes sin Positional Encoding.
### 3.2 Implementación de Positional Encoding Sinusoidal
```python
def positional_encoding_sinusoidal(seq_len, d_model):
    """
    Genera Positional Encoding sinusoidal (Vaswani et al., 2017).
    
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    Args:
        seq_len: longitud máxima de secuencia
        d_model: dimensión del modelo
    
    Returns:
        PE: (seq_len, d_model)
    """
    PE = np.zeros((seq_len, d_model))
    
    posiciones = np.arange(seq_len).reshape(-1, 1)     # (seq_len, 1)
    indices_dim = np.arange(0, d_model, 2)              # 0, 2, 4, ..., d_model-2
    
    # Calcular los divisores: 10000^(2i/d_model)
    divisores = np.power(10000.0, indices_dim / d_model)    # (d_model/2,)
    
    # Asignar senos a dimensiones pares
    PE[:, 0::2] = np.sin(posiciones / divisores)
    # Asignar cosenos a dimensiones impares
    PE[:, 1::2] = np.cos(posiciones / divisores)
    
    return PE
# ─── Generación y visualización ───
seq_len = 50
d_model = 64
PE = positional_encoding_sinusoidal(seq_len, d_model)
print(f"Positional Encoding shape: {PE.shape}")
print(f"PE[0, :6] (pos=0):         {np.round(PE[0, :6], 4)}")
print(f"PE[1, :6] (pos=1):         {np.round(PE[1, :6], 4)}")
# Verificar propiedad: rango de valores siempre entre -1 y 1
print(f"\nRango de valores: [{PE.min():.2f}, {PE.max():.2f}]")
# Similitud entre posiciones consecutivas vs. distantes
dot_consec = PE[0] @ PE[1]
dot_lejos  = PE[0] @ PE[25]
print(f"\nProducto punto pos (0,1):  {dot_consec:.2f}  (posiciones cercanas)")
print(f"Producto punto pos (0,25): {dot_lejos:.2f}  (posiciones lejanas)")
```
### 3.3 Visualización del Positional Encoding
```python
def visualizar_positional_encoding(PE, titulo="Positional Encoding Sinusoidal"):
    """Visualiza el mapa de calor del Positional Encoding."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Mapa de calor completo
    im = axes[0].imshow(PE, cmap='RdBu', aspect='auto', vmin=-1, vmax=1)
    axes[0].set_xlabel('Dimensión del embedding', fontsize=12)
    axes[0].set_ylabel('Posición en la secuencia', fontsize=12)
    axes[0].set_title('Mapa de calor completo')
    plt.colorbar(im, ax=axes[0])
    
    # Primeras 4 dimensiones a lo largo de la secuencia
    for dim in range(4):
        etiqueta = f'dim {dim} ({"sin" if dim % 2 == 0 else "cos"})'
        axes[1].plot(PE[:, dim], label=etiqueta)
    axes[1].set_xlabel('Posición', fontsize=12)
    axes[1].set_ylabel('Valor de encoding', fontsize=12)
    axes[1].set_title('Primeras 4 dimensiones')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.suptitle(titulo, fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('positional_encoding.png', dpi=120, bbox_inches='tight')
    plt.show()
    print("✅ Guardado: positional_encoding.png")
PE_vis = positional_encoding_sinusoidal(100, 128)
visualizar_positional_encoding(PE_vis)
```
**Actividad 3.1**: Calcula la **similitud de coseno** entre todos los pares de posiciones del encoding. ¿Qué observas? ¿Las posiciones cercanas son más similares que las lejanas?
**Actividad 3.2**: Compara el encoding sinusoidal con un **encoding aprendible** (embeddings de posición aleatorios que se entrenarían). ¿Qué ventajas tiene el sinusoidal para secuencias más largas que las vistas en entrenamiento?
### Preguntas de Reflexión
**Pregunta 3.1 (Concebir)**: ¿Por qué la base 10000 en la función sinusoidal? Prueba con base 100 y base 1000000 y visualiza la diferencia.
**Pregunta 3.2 (Diseñar)**: BERT usa embeddings de posición **aprendibles** (no sinusoidales). ¿Qué implicación tiene esto para secuencias más largas que la longitud máxima de entrenamiento (512 tokens)?
**Pregunta 3.3 (Implementar)**: El Positional Encoding se **suma** a los embeddings de tokens (no se concatena). ¿Por qué suma y no concatenación? ¿Qué dimensionalidad se perdería con la concatenación?
**Pregunta 3.4 (Operar)**: Observa el mapa de calor del encoding. ¿Qué tipo de frecuencias corresponden a las primeras dimensiones vs. las últimas dimensiones?
## 🔬 Parte 4: Transformer Encoder Block con PyTorch (45 min)
### 4.1 De NumPy a PyTorch
Hasta ahora implementaste los componentes desde cero con NumPy. En esta parte construirás un Transformer Encoder Block completo y diferenciable con PyTorch, listo para entrenamiento con backpropagation.
### 4.2 Transformer Encoder Block
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
class MultiHeadAttentionPyTorch(nn.Module):
    """
    Multi-Head Attention implementado con PyTorch.
    Utiliza nn.Linear para las proyecciones Q, K, V y O.
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Proyecciones lineales (sin bias para simplificar; en práctica se usa bias)
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
    
    def split_heads(self, x, batch_size):
        """(batch, seq, d_model) → (batch, heads, seq, d_k)"""
        x = x.view(batch_size, -1, self.num_heads, self.d_k)
        return x.transpose(1, 2)    # (batch, heads, seq, d_k)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Proyecciones y split en cabezas
        Q = self.split_heads(self.W_Q(query), batch_size)    # (b, h, s, d_k)
        K = self.split_heads(self.W_K(key), batch_size)
        V = self.split_heads(self.W_V(value), batch_size)
        
        # Scaled dot-product attention
        scores = Q @ K.transpose(-2, -1) / (self.d_k ** 0.5)  # (b, h, s, s)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        pesos = F.softmax(scores, dim=-1)
        pesos = self.dropout(pesos)
        
        # Suma ponderada y concatenación de cabezas
        attn_out = pesos @ V                              # (b, h, s, d_k)
        attn_out = attn_out.transpose(1, 2).contiguous()  # (b, s, h, d_k)
        attn_out = attn_out.view(batch_size, -1, self.d_model)  # (b, s, d_model)
        
        return self.W_O(attn_out), pesos


class FeedForwardBlock(nn.Module):
    """
    Red Feed-Forward del Transformer: dos capas lineales con ReLU (o GELU).
    FFN(x) = max(0, x·W₁ + b₁)·W₂ + b₂
    La dimensión interna suele ser 4× la del modelo.
    """
    def __init__(self, d_model, d_ff=None, dropout=0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


class TransformerEncoderBlock(nn.Module):
    """
    Bloque Encoder del Transformer (Vaswani et al., 2017):
    
    1. Multi-Head Self-Attention
    2. Add & LayerNorm  (conexión residual)
    3. Feed-Forward Network
    4. Add & LayerNorm  (conexión residual)
    """
    def __init__(self, d_model, num_heads, d_ff=None, dropout=0.1):
        super().__init__()
        
        self.attention = MultiHeadAttentionPyTorch(d_model, num_heads, dropout)
        self.ffn       = FeedForwardBlock(d_model, d_ff, dropout)
        
        self.norm1   = nn.LayerNorm(d_model)
        self.norm2   = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Sub-capa 1: Self-Attention + residual + norm
        attn_out, _ = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Sub-capa 2: FFN + residual + norm
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        
        return x


class TransformerEncoder(nn.Module):
    """
    Encoder completo: apilamiento de N bloques encoder.
    Incluye embedding de tokens y positional encoding.
    """
    def __init__(self, vocab_size, d_model, num_heads, num_layers,
                 d_ff=None, max_seq_len=512, dropout=0.1):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Embedding(max_seq_len, d_model)    # aprendible
        self.dropout = nn.Dropout(dropout)
        
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        
        # Inicialización de pesos
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, token_ids, mask=None):
        batch, seq_len = token_ids.shape
        
        # Embeddings de tokens + posicionales
        posiciones = torch.arange(seq_len, device=token_ids.device).unsqueeze(0)
        x = self.dropout(self.embedding(token_ids) + self.pos_encoding(posiciones))
        
        # Pasar por N bloques encoder
        for capa in self.layers:
            x = capa(x, mask)
        
        return self.norm(x)
# ─── Prueba del modelo ───
torch.manual_seed(42)
vocab_size = 1000
d_model    = 64
num_heads  = 4
num_layers = 2
encoder = TransformerEncoder(
    vocab_size=vocab_size,
    d_model=d_model,
    num_heads=num_heads,
    num_layers=num_layers,
    max_seq_len=100
)
# Batch de 3 secuencias de 10 tokens cada una
batch_tokens = torch.randint(0, vocab_size, (3, 10))
representaciones = encoder(batch_tokens)
print(f"Input (tokens):  {batch_tokens.shape}   → (batch=3, seq_len=10)")
print(f"Output (repr.):  {representaciones.shape} → (batch=3, seq_len=10, d_model=64)")
# Contar parámetros
total_params = sum(p.numel() for p in encoder.parameters())
print(f"\nTotal de parámetros: {total_params:,}")
print("Desglose por módulo:")
for nombre, modulo in encoder.named_children():
    params = sum(p.numel() for p in modulo.parameters())
    print(f"  {nombre}: {params:,}")
```
**Actividad 4.1**: Construye un **TransformerDecoderBlock** que incluya (1) Masked Self-Attention, (2) Cross-Attention con las representaciones del encoder, y (3) FFN, cada uno seguido de Add & LayerNorm.
**Actividad 4.2**: Implementa un clasificador de texto simple añadiendo una capa lineal al final del encoder (sobre el token `[CLS]`) y entrénalo en un dataset de juguete con 2 clases.
### Preguntas de Reflexión
**Pregunta 4.1 (Concebir)**: Las conexiones residuales (Add) son fundamentales para el entrenamiento de redes profundas. ¿Qué problema resuelven concretamente y cómo lo hacen?
**Pregunta 4.2 (Diseñar)**: ¿Por qué se usa Layer Normalization en los Transformers en lugar de Batch Normalization? ¿Qué diferencia hay en qué dimensión se normaliza?
**Pregunta 4.3 (Implementar)**: El paper original usa la variante "Post-LN" (Add & Norm después de la sub-capa). Los modelos modernos usan "Pre-LN" (Norm antes de la sub-capa). ¿Cuál es más estable durante el entrenamiento y por qué?
**Pregunta 4.4 (Operar)**: Experimenta con distintos valores de dropout (0.0, 0.1, 0.3). ¿Cómo afecta al overfitting cuando entrenas con pocos datos?
## 🔬 Parte 5: Fine-tuning de BERT con Hugging Face (50 min)
### 5.1 Transfer Learning con Transformers Pre-entrenados
Pre-entrenar un Transformer desde cero requiere recursos masivos (BERT-base fue entrenado durante 4 días en 64 TPUs de Google con 3.3 mil millones de palabras). En la práctica, usamos modelos pre-entrenados y los **ajustamos (fine-tuning)** para nuestra tarea específica.
```
PRE-TRAINING (una vez, muy costoso):
  BERT fue entrenado con:
  1. Masked Language Modeling (MLM): predecir tokens enmascarados
  2. Next Sentence Prediction (NSP): ¿estas dos frases son consecutivas?

FINE-TUNING (rápido, por tarea):
  Tomar BERT pre-entrenado + añadir capa de clasificación + entrenar con pocos datos
```
### 5.2 Análisis de Sentimientos con BERT
```python
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset
import torch
import numpy as np
# ─── 1. Cargar tokenizador y modelo pre-entrenado ───
model_name = "distilbert-base-uncased"    # versión ligera de BERT (40% más rápido)
tokenizer = AutoTokenizer.from_pretrained(model_name)
modelo = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2    # Positivo / Negativo
)
print(f"Modelo cargado: {model_name}")
print(f"Parámetros: {sum(p.numel() for p in modelo.parameters()):,}")
# ─── 2. Explorar el tokenizador ───
ejemplos = [
    "This movie was absolutely fantastic!",
    "I hated every minute of this film.",
    "The plot was interesting but the acting was poor."
]
for texto in ejemplos:
    tokens = tokenizer(texto, return_tensors='pt', truncation=True, max_length=64)
    print(f"\nTexto: {texto[:50]}...")
    print(f"  IDs: {tokens['input_ids'][0][:8].tolist()} ...")
    print(f"  Tokens: {tokenizer.convert_ids_to_tokens(tokens['input_ids'][0][:8])}")
# ─── 3. Preparar dataset ───
dataset = load_dataset("imdb", split={'train': 'train[:2000]', 'test': 'test[:500]'})
def tokenizar(ejemplos_batch):
    return tokenizer(
        ejemplos_batch['text'],
        truncation=True,
        max_length=256,
        padding='max_length'
    )
dataset_tokenizado = dataset.map(tokenizar, batched=True)
dataset_tokenizado = dataset_tokenizado.rename_column("label", "labels")
dataset_tokenizado.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
print(f"\nDataset tokenizado:")
print(f"  Train: {len(dataset_tokenizado['train'])} muestras")
print(f"  Test:  {len(dataset_tokenizado['test'])} muestras")
# ─── 4. Configurar entrenamiento ───
def calcular_metricas(eval_pred):
    logits, labels = eval_pred
    predicciones = np.argmax(logits, axis=-1)
    accuracy = (predicciones == labels).mean()
    return {"accuracy": accuracy}
training_args = TrainingArguments(
    output_dir="./bert_sentimiento",
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    learning_rate=2e-5,            # LR pequeño para fine-tuning
    warmup_ratio=0.1,
    weight_decay=0.01,
    fp16=torch.cuda.is_available(),
    logging_steps=50,
    report_to="none"
)
trainer = Trainer(
    model=modelo,
    args=training_args,
    train_dataset=dataset_tokenizado['train'],
    eval_dataset=dataset_tokenizado['test'],
    compute_metrics=calcular_metricas
)
# ─── 5. Entrenar ───
print("\n🚀 Iniciando fine-tuning...")
trainer.train()
print("\n✅ Fine-tuning completado!")
# ─── 6. Evaluación e inferencia ───
resultados = trainer.evaluate()
print(f"\n📊 Resultados en test:")
print(f"  Accuracy: {resultados['eval_accuracy']:.4f}")
# Inferencia en nuevas frases
def predecir_sentimiento(textos, modelo, tokenizer):
    modelo.eval()
    codificado = tokenizer(textos, return_tensors='pt',
                           truncation=True, max_length=256, padding=True)
    with torch.no_grad():
        logits = modelo(**codificado).logits
    predicciones = torch.argmax(logits, dim=-1)
    etiquetas = ['Negativo', 'Positivo']
    return [(t, etiquetas[p.item()], torch.softmax(logits, dim=-1)[i].max().item())
            for i, (t, p) in enumerate(zip(textos, predicciones))]
nuevas_frases = [
    "The special effects were mind-blowing and the story was compelling.",
    "I fell asleep halfway through. Complete waste of time.",
    "Decent film, nothing extraordinary but enjoyable enough."
]
print("\n🔍 Predicciones en nuevas frases:")
for texto, etiqueta, confianza in predecir_sentimiento(nuevas_frases, modelo, tokenizer):
    print(f"  '{texto[:60]}...'")
    print(f"  → {etiqueta} ({confianza:.2%} confianza)")
```
**Actividad 5.1**: Experimenta con distintos learning rates (5e-5, 2e-5, 1e-5, 5e-6). ¿Qué ocurre con un LR demasiado alto durante el fine-tuning de BERT?
**Actividad 5.2**: Prueba **congelar los primeros 6 layers** del modelo y solo entrenar los últimos 6 + la capa de clasificación. ¿Cómo afecta al rendimiento y al tiempo de entrenamiento?
### Preguntas de Reflexión
**Pregunta 5.1 (Concebir)**: ¿Por qué se usa un learning rate mucho más pequeño (2e-5) para fine-tuning que para entrenar desde cero (1e-3)? ¿Qué podría ocurrir con un LR grande?
**Pregunta 5.2 (Diseñar)**: ¿Cuál es la diferencia entre DistilBERT, BERT-base y BERT-large en términos de número de capas, cabezas y parámetros? ¿Cuándo elegirías cada uno?
**Pregunta 5.3 (Implementar)**: El token `[CLS]` al inicio de cada secuencia en BERT acumula información global. ¿Cómo podrías usar la representación de `[CLS]` directamente sin la API de Hugging Face?
**Pregunta 5.4 (Operar)**: Evalúa tu modelo con una frase ambigua como "It's not bad, just not what I expected." ¿Qué etiqueta asigna? ¿Tiene sentido? ¿Qué dice esto sobre las limitaciones del modelo?
## 🔬 Parte 6: Generación de Texto con GPT-2 (40 min)
### 6.1 Arquitectura Decoder: GPT vs. BERT
GPT es un modelo **decoder-only**: usa atención causal (máscara triangular), de modo que cada token solo puede "ver" los tokens anteriores. Esto lo hace ideal para generación autoregresiva.
```
BERT (Encoder — bidireccional):
  x₁ ↔ x₂ ↔ x₃ ↔ x₄   (cada token ve todos los demás)
  → Ideal para comprensión

GPT (Decoder — causal/unidireccional):
  x₁ → x₂ → x₃ → x₄   (cada token solo ve los anteriores)
  → Ideal para generación
```
### 6.2 Generación de Texto con GPT-2 y Hugging Face
```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
# ─── 1. Cargar GPT-2 ───
tokenizer_gpt2 = GPT2Tokenizer.from_pretrained('gpt2')
modelo_gpt2    = GPT2LMHeadModel.from_pretrained('gpt2')
modelo_gpt2.eval()
print(f"GPT-2 cargado. Parámetros: {sum(p.numel() for p in modelo_gpt2.parameters()):,}")
print(f"Tamaño del vocabulario: {tokenizer_gpt2.vocab_size:,}")
# ─── 2. Generación básica (greedy) ───
def generar_texto_greedy(prompt, max_new_tokens=50):
    """Generación greedy: en cada paso elige el token más probable."""
    ids = tokenizer_gpt2.encode(prompt, return_tensors='pt')
    
    with torch.no_grad():
        output_ids = modelo_gpt2.generate(
            ids,
            max_new_tokens=max_new_tokens,
            do_sample=False    # greedy
        )
    
    return tokenizer_gpt2.decode(output_ids[0], skip_special_tokens=True)
# ─── 3. Generación con sampling ───
def generar_texto_sampling(prompt, max_new_tokens=100,
                            temperature=0.8, top_k=50, top_p=0.92):
    """
    Generación con muestreo:
    - temperature: controla la aleatoriedad (< 1 = más conservador)
    - top_k: considera solo los k tokens más probables en cada paso
    - top_p (nucleus sampling): considera tokens hasta cubrir probabilidad p
    """
    ids = tokenizer_gpt2.encode(prompt, return_tensors='pt')
    
    with torch.no_grad():
        output_ids = modelo_gpt2.generate(
            ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=1.2,    # penaliza repeticiones
            pad_token_id=tokenizer_gpt2.eos_token_id
        )
    
    return tokenizer_gpt2.decode(output_ids[0], skip_special_tokens=True)
# ─── 4. Demostración ───
prompts = [
    "Artificial intelligence will transform",
    "The history of neural networks began",
    "In the future, language models will"
]
print("=" * 60)
for prompt in prompts:
    print(f"\n📝 Prompt: '{prompt}'")
    print(f"\n🤖 Greedy:")
    print(generar_texto_greedy(prompt, max_new_tokens=40))
    print(f"\n🎲 Sampling (temp=0.8):")
    print(generar_texto_sampling(prompt, max_new_tokens=60))
    print("-" * 60)
```
### 6.3 Análisis de Probabilidades de Tokens
```python
def analizar_probabilidades(prompt, top_n=10):
    """
    Muestra las probabilidades del siguiente token más probable dado un prompt.
    Útil para entender cómo GPT-2 "razona".
    """
    ids = tokenizer_gpt2.encode(prompt, return_tensors='pt')
    
    with torch.no_grad():
        logits = modelo_gpt2(ids).logits
    
    # Probabilidades del último token (siguiente a generar)
    probs = torch.softmax(logits[0, -1, :], dim=-1)
    top_probs, top_ids = probs.topk(top_n)
    
    print(f"Prompt: '{prompt}'")
    print(f"Siguiente token más probable:")
    for prob, idx in zip(top_probs, top_ids):
        token = tokenizer_gpt2.decode([idx.item()])
        print(f"  '{token}': {prob.item():.4f} ({prob.item()*100:.2f}%)")
analizar_probabilidades("The Transformer architecture was introduced in")
analizar_probabilidades("Deep learning models require large amounts of")
```
**Actividad 6.1**: Experimenta con distintos valores de `temperature` (0.3, 0.7, 1.0, 1.5). ¿Cómo afecta a la coherencia y creatividad del texto generado?
**Actividad 6.2**: Implementa **beam search** con `num_beams=5` y compara la calidad del texto con la generación greedy y por sampling.
### Preguntas de Reflexión
**Pregunta 6.1 (Concebir)**: ¿Qué es el "hallucination problem" en LLMs? A partir de tus experimentos con GPT-2, ¿puedes identificar casos donde el modelo genera texto plausible pero factualmente incorrecto?
**Pregunta 6.2 (Diseñar)**: Diseña un sistema de clasificación de texto usando GPT-2 con **prompting** (sin fine-tuning). Por ejemplo: "La siguiente reseña de película es [positiva/negativa]: [texto]". ¿Qué ventajas y limitaciones tiene este enfoque vs. fine-tuning de BERT?
**Pregunta 6.3 (Implementar)**: La `temperature` en la generación se aplica dividiendo los logits antes del softmax: `probs = softmax(logits / T)`. ¿Qué ocurre matemáticamente con T→0 (greedy) y T→∞ (distribución uniforme)?
**Pregunta 6.4 (Operar)**: Mide el tiempo de generación de 100 tokens con GPT-2 en CPU vs. GPU (si disponible). ¿Cuánto más rápida es la GPU? ¿Qué implica esto para el despliegue de LLMs en producción?
## 📊 Análisis Final de Rendimiento y Complejidad (30 min)
### Comparativa: Transformers vs. RNNs
```python
import numpy as np
import matplotlib.pyplot as plt
import time
# ─── Complejidad teórica ───
def analizar_complejidad_teorica():
    """
    Compara complejidad O() de RNN vs. Transformer
    
    RNN:       O(n * d²)          tiempo, O(n) paralelo → NO
    Self-Attn: O(n² * d)          tiempo, O(1) paralelo → SÍ
    FFN:       O(n * d * d_ff)    tiempo, O(1) paralelo → SÍ
    """
    longitudes = np.arange(10, 1001, 10)
    d_model = 512
    d_ff = 2048
    
    # Operaciones de punto flotante (aprox.)
    ops_rnn     = longitudes * d_model ** 2        # secuencial
    ops_attn    = longitudes ** 2 * d_model        # cuadrático en longitud
    ops_ffn     = longitudes * d_model * d_ff      # lineal en longitud
    ops_total_t = ops_attn + ops_ffn               # Transformer total
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Gráfica 1: FLOPs vs longitud
    axes[0].plot(longitudes, ops_rnn / 1e9,     label='RNN (secuencial)', color='red', linewidth=2)
    axes[0].plot(longitudes, ops_total_t / 1e9, label='Transformer (paralelo)', color='blue', linewidth=2)
    axes[0].plot(longitudes, ops_attn / 1e9,    label='Solo Self-Attention', color='blue',
                 linestyle='--', linewidth=1.5)
    axes[0].set_xlabel('Longitud de secuencia (tokens)', fontsize=12)
    axes[0].set_ylabel('GFLOPs (aprox.)', fontsize=12)
    axes[0].set_title('Complejidad Computacional', fontsize=13)
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Gráfica 2: Memoria (attention matrix crece cuadráticamente)
    memoria_attn = longitudes ** 2 * d_model * 4 / (1024**2)  # MB (float32)
    memoria_rnn  = longitudes * d_model * 4 / (1024**2)        # MB
    
    axes[1].plot(longitudes, memoria_attn, label='Attention Matrix (Transformer)', color='blue', linewidth=2)
    axes[1].plot(longitudes, memoria_rnn,  label='Estado oculto (RNN)', color='red', linewidth=2)
    axes[1].set_xlabel('Longitud de secuencia (tokens)', fontsize=12)
    axes[1].set_ylabel('Memoria aproximada (MB)', fontsize=12)
    axes[1].set_title('Uso de Memoria', fontsize=13)
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.suptitle('Transformer vs. RNN — Complejidad y Escalabilidad', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('complejidad_transformer_rnn.png', dpi=120, bbox_inches='tight')
    plt.show()
    print("✅ Guardado: complejidad_transformer_rnn.png")
    
    # Tabla resumen
    print("\n" + "="*70)
    print(f"{'Métrica':<30} {'RNN/LSTM':<20} {'Transformer':<20}")
    print("="*70)
    metricas = [
        ("Complejidad tiempo/capa",  "O(n·d²)",     "O(n²·d)"),
        ("Paralelización",           "NO (secuenc.)", "SÍ (completa)"),
        ("Dependencias largas",      "Difícil",       "O(1) en profundidad"),
        ("Memoria (attention)",      "O(n·d)",        "O(n²·d)"),
        ("Escalabilidad con datos",  "Moderada",      "Muy alta"),
        ("Interpretabilidad",        "Difícil",       "Attention weights"),
    ]
    for metrica, rnn, transformer in metricas:
        print(f"  {metrica:<28} {rnn:<20} {transformer:<20}")
    print("="*70)
analizar_complejidad_teorica()
```
### Análisis de Escalabilidad Empírica
```python
def benchmark_transformer_escalabilidad():
    """Benchmark empírico: tiempo de inferencia vs. longitud de secuencia."""
    import torch
    
    d_model = 128
    num_heads = 4
    
    class AttnSimple(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = MultiHeadAttentionPyTorch(d_model, num_heads, dropout=0.0)
        def forward(self, x):
            return self.attn(x, x, x)[0]
    
    modelo_bench = AttnSimple().eval()
    longitudes = [32, 64, 128, 256, 512]
    tiempos = []
    
    for seq_len in longitudes:
        x = torch.randn(1, seq_len, d_model)
        # Warm-up
        with torch.no_grad():
            for _ in range(5):
                modelo_bench(x)
        # Benchmark
        inicio = time.time()
        with torch.no_grad():
            for _ in range(50):
                modelo_bench(x)
        tiempo_medio = (time.time() - inicio) / 50 * 1000  # ms
        tiempos.append(tiempo_medio)
        print(f"  seq_len={seq_len:4d}: {tiempo_medio:.2f} ms")
    
    # Graficar
    plt.figure(figsize=(8, 4))
    plt.plot(longitudes, tiempos, 'o-', color='blue', linewidth=2, markersize=8)
    plt.xlabel('Longitud de secuencia', fontsize=12)
    plt.ylabel('Tiempo por inferencia (ms)', fontsize=12)
    plt.title('Tiempo de inferencia de Self-Attention vs. Longitud (CPU)', fontsize=13)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('benchmark_atencion.png', dpi=120, bbox_inches='tight')
    plt.show()
    
    # Verificar si crece cuadráticamente
    ratios = [tiempos[i+1] / tiempos[i] for i in range(len(tiempos)-1)]
    print(f"\nRatios de tiempo al doblar la secuencia: {[f'{r:.2f}x' for r in ratios]}")
    print(f"Si O(n²): esperaríamos ratios de ~4x")
    print(f"Ratio observado promedio: {np.mean(ratios):.2f}x")
print("\n📊 Benchmark de escalabilidad:")
benchmark_transformer_escalabilidad()
```
## 🎯 EJERCICIOS PROPUESTOS
### Ejercicio 1 — Básico (30 min): Self-Attention Manual
Implementa Self-Attention completamente desde cero **sin usar las clases del laboratorio**, solo NumPy y la fórmula:
$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$
**Requisitos:**
1. Crea matrices Q, K, V aleatorias de forma (5, 8)
2. Calcula los attention scores paso a paso
3. Aplica scaling con √d_k
4. Calcula softmax manualmente (sin `np.exp` de scipy, solo NumPy)
5. Multiplica por V para obtener el output
6. Verifica que los pesos de atención suman 1.0 por fila
7. Documenta cada paso con comentarios claros
**Entregable:** Función `mi_self_attention(Q, K, V)` completamente documentada.
### Ejercicio 2 — Intermedio (60 min): Transformer Block desde Cero
Implementa un **Transformer Encoder Block completo** usando únicamente NumPy (sin PyTorch):
**Requisitos:**
1. Clase `LayerNormNumPy` con parámetros γ y β aprendibles
2. Clase `FeedForwardNumPy` con dos capas lineales y activación ReLU
3. Clase `TransformerBlockNumPy` que integre:
   - Multi-Head Attention (usa tu implementación de la Parte 2)
   - Conexión residual + LayerNorm
   - FFN + conexión residual + LayerNorm
4. Prueba el bloque con una secuencia de 8 tokens y d_model=32
5. Verifica que las dimensiones de entrada y salida son idénticas
6. Implementa un paso de forward y backward **manual** para un parámetro (gradiente por definición)
**Entregable:** Clase `TransformerBlockNumPy` con tests de dimensiones y un diagrama ASCII del flujo de datos.
### Ejercicio 3 — Avanzado (90 min): Fine-tuning para Clasificación Multiclase
Fine-tunea **BERT** (o DistilBERT) para clasificación de **20 categorías** de noticias (dataset `20newsgroups`):
**Requisitos:**
1. Carga el dataset con Hugging Face `datasets` o `sklearn`
2. Preprocesa y tokeniza correctamente (max_length=256, truncation, padding)
3. Fine-tunea por mínimo 3 épocas con learning rate scheduling
4. Reporta accuracy, F1-macro y matriz de confusión
5. Identifica las **3 categorías** con peor rendimiento y analiza por qué
6. Compara con un baseline de TF-IDF + Regresión Logística
**Entregable:** Notebook con experimentación, tabla comparativa de métricas y análisis crítico de errores.
### Ejercicio 4 — Desafío (2-3 horas): Transformer Miniatura para Clasificación de Secuencias
Construye un **mini-Transformer desde cero con PyTorch** para clasificar secuencias sintéticas:
**Requisitos:**
1. Genera un dataset sintético: secuencias de números donde la etiqueta depende de patrones (e.g., "¿hay un 5 seguido de un 7 en la secuencia?")
2. Implementa un Transformer Encoder con:
   - Positional encoding sinusoidal (no aprendible)
   - 2 bloques encoder (d_model=64, num_heads=4, d_ff=256)
   - Capa de clasificación sobre el token `[CLS]`
3. Entrena con Adam, learning rate scheduling con warm-up
4. Grafica curvas de pérdida y accuracy en train/val
5. Visualiza los **mapas de atención** de ambas capas para 5 ejemplos de test
6. Compara contra un LSTM equivalente en mismo número de parámetros
**Entregable:** Código limpio, gráficas comparativas y análisis de qué aprende cada cabeza de atención.
### Ejercicio 5 — Proyecto (4+ horas): Sistema de Preguntas y Respuestas con BERT
Construye un sistema de **Question Answering (QA) extractivo** usando BERT:
**Requisitos:**
1. Usa el dataset SQuAD (Stanford Question Answering Dataset) con Hugging Face
2. Fine-tunea `bert-base-uncased` para la tarea de QA extractiva:
   - El modelo debe predecir la posición inicio/fin de la respuesta en el contexto
3. Implementa la lógica de inferencia: dado un contexto y una pregunta, extraer el span de respuesta
4. Evalúa con métricas EM (Exact Match) y F1 token-level
5. Implementa una **interfaz de demostración interactiva** (CLI o Gradio)
6. Analiza casos donde el modelo falla (preguntas que requieren razonamiento vs. extracción directa)
**Entregable completo:**
- Código de entrenamiento y evaluación
- Métricas EM y F1 en el set de validación
- Interfaz de demostración funcional
- Reporte de 3 páginas con análisis de errores y propuestas de mejora
## 📝 Entregables
Para completar este laboratorio, debes entregar:
### 1. Código Implementado (60%)
- Archivo `transformers_scratch.py` con implementaciones NumPy (Partes 1-3)
- Archivo `transformer_pytorch.py` con el Transformer en PyTorch (Parte 4)
- Archivo `bert_finetune.py` con el fine-tuning de BERT (Parte 5)
- Archivo `gpt2_generation.py` con generación de texto (Parte 6)
- Todas las funciones y clases con docstrings completos
- Código limpio, modular y con manejo de errores
### 2. Notebook de Experimentación (25%)
- `practica_lab12.ipynb` con:
  - Todas las actividades completadas
  - Visualizaciones de mapas de atención claramente etiquetadas
  - Gráficas de positional encoding
  - Curvas de entrenamiento del fine-tuning
  - Ejemplos de texto generado con GPT-2
  - Respuestas a todas las Preguntas de Reflexión
### 3. Reporte Técnico (15%)
- Documento PDF (3-4 páginas) que incluya:
  - Explicación del mecanismo de Self-Attention con fórmulas
  - Comparativa Transformers vs. RNNs (ventajas y limitaciones)
  - Resultados del fine-tuning (accuracy, curvas de entrenamiento)
  - Análisis de mapas de atención
  - Reflexiones sobre el impacto de los Transformers en IA
### Formato de Entrega
```
Lab12_NombreApellido/
├── codigo/
│   ├── transformers_scratch.py     # NumPy: Partes 1-3
│   ├── transformer_pytorch.py      # PyTorch: Parte 4
│   ├── bert_finetune.py            # BERT fine-tuning: Parte 5
│   └── gpt2_generation.py          # GPT-2: Parte 6
├── notebooks/
│   ├── practica_lab12.ipynb        # Notebook principal
│   └── ejercicios.ipynb            # Ejercicios propuestos
├── imagenes/
│   ├── atencion_multihead.png
│   ├── positional_encoding.png
│   ├── complejidad_transformer_rnn.png
│   └── benchmark_atencion.png
├── modelos/
│   └── bert_sentimiento/           # Modelo fine-tuneado (opcional)
├── reporte/
│   └── reporte_lab12.pdf
└── README.md
```
## 🎯 Criterios de Evaluación (CDIO)
### Concebir (25%)
- ✅ Comprende el mecanismo de Self-Attention (Q, K, V) y su intuición
- ✅ Explica por qué se necesita el escalado por √d_k
- ✅ Identifica las diferencias entre Encoder (BERT) y Decoder (GPT)
- ✅ Reconoce las limitaciones O(n²) de los Transformers y sus variantes eficientes
- ✅ Justifica cuándo usar fine-tuning vs. prompting vs. entrenamiento desde cero
**Evidencia**: Respuestas a preguntas de reflexión, introducción del reporte, análisis comparativo
### Diseñar (25%)
- ✅ Diseña la arquitectura apropiada para una tarea NLP dada (encoder vs. decoder vs. encoder-decoder)
- ✅ Elige hiperparámetros justificados (d_model, num_heads, num_layers, learning rate)
- ✅ Diseña el pipeline de tokenización, padding y masking correctamente
- ✅ Planifica el proceso de fine-tuning considerando el riesgo de catastrophic forgetting
- ✅ Selecciona el modelo pre-entrenado apropiado para la tarea
**Evidencia**: Decisiones de diseño documentadas, tabla de comparación de arquitecturas, justificación de hiperparámetros
### Implementar (30%)
- ✅ Implementa Scaled Dot-Product Attention correctamente desde cero (NumPy)
- ✅ Construye Multi-Head Attention con las dimensiones correctas
- ✅ Implementa Positional Encoding sinusoidal verificando propiedades matemáticas
- ✅ Construye el Transformer Encoder Block completo en PyTorch (atención + FFN + residual + norm)
- ✅ Fine-tunea BERT con el pipeline de Hugging Face correctamente
- ✅ Genera texto con GPT-2 usando diferentes estrategias de decodificación
- ✅ Código modular, documentado y con tests de dimensiones
**Evidencia**: Código fuente, dimensiones verificadas, output de ejecución sin errores
### Operar (20%)
- ✅ Analiza críticamente los mapas de atención generados
- ✅ Evalúa el fine-tuning con métricas apropiadas (accuracy, F1, confusion matrix)
- ✅ Identifica y documenta casos de fallo del modelo
- ✅ Compara empíricamente el rendimiento de Transformers vs. RNNs
- ✅ Propone mejoras concretas basadas en los resultados observados
- ✅ Resuelve problemas de memoria/GPU con estrategias prácticas
**Evidencia**: Reporte técnico, visualizaciones anotadas, análisis de errores, propuestas de mejora
### Rúbrica Detallada
| Criterio | Excelente (100%) | Bueno (80%) | Aceptable (60%) | Insuficiente (<60%) |
|----------|------------------|-------------|-----------------|---------------------|
| **Self-Attention** | Implementación perfecta, verifica todas las propiedades matemáticas | Funciona correctamente, pequeños detalles | Funciona con limitaciones | Incompleto o incorrecto |
| **Multi-Head Attention** | Implementación vectorizada, visualización clara de múltiples cabezas | Funciona con loop, visualización básica | Funciona parcialmente | No funciona |
| **Positional Encoding** | Implementación correcta, análisis de propiedades, visualización clara | Implementación correcta, visualización básica | Implementación con errores menores | Ausente o incorrecto |
| **Transformer PyTorch** | Bloque completo, conexiones residuales, LN, entrenamiento demostrado | Bloque funcional, algunos componentes faltantes | Estructura parcial | No funcional |
| **Fine-tuning BERT** | Accuracy > 90%, curvas de entrenamiento, análisis de errores | Accuracy > 85%, métricas básicas | Accuracy > 75%, sin análisis | No se ejecuta |
| **Generación GPT-2** | Múltiples estrategias comparadas, análisis de calidad | Generación funcional, una estrategia | Generación básica | No funciona |
| **Análisis** | Profundo, crítico, con comparativa Transformer vs. RNN | Completo y correcto | Superficial | Ausente |
| **Documentación** | Excelente, profesional, con fórmulas y referencias | Buena, entendible | Básica | Pobre o ausente |
## 📚 Referencias Adicionales
### Libros
1. **Vaswani, A. et al.** (2017). "Attention Is All You Need"
   - Artículo original del Transformer — lectura obligatoria
   - https://arxiv.org/abs/1706.03762
2. **Goodfellow, I., Bengio, Y., & Courville, A.** (2016). "Deep Learning"
   - Capítulo 10: Sequence Modeling: Recurrent and Recursive Nets
   - Capítulo 12: Applications (NLP)
   - http://www.deeplearningbook.org
3. **Jurafsky, D., & Martin, J.H.** (2023). "Speech and Language Processing" (3rd ed.)
   - Capítulo 9: Transformers and Pre-Trained Language Models
   - Disponible gratuitamente: https://web.stanford.edu/~jurafsky/slp3/
4. **Lewis, T., Fergus, R., & Conneau, A.** — Hugging Face Course
   - https://huggingface.co/course — gratuito e interactivo
### Artículos Académicos
1. **Vaswani, A. et al.** (2017). "Attention Is All You Need" — *NeurIPS 2017*
   - Introducción de la arquitectura Transformer
2. **Devlin, J. et al.** (2018). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" — *NAACL 2019*
   - https://arxiv.org/abs/1810.04805
3. **Brown, T. et al.** (2020). "Language Models are Few-Shot Learners" (GPT-3) — *NeurIPS 2020*
   - https://arxiv.org/abs/2005.14165
4. **Dosovitskiy, A. et al.** (2020). "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (ViT)
   - https://arxiv.org/abs/2010.11929
5. **Bahdanau, D. et al.** (2014). "Neural Machine Translation by Jointly Learning to Align and Translate"
   - Primer mecanismo de atención — contexto histórico
   - https://arxiv.org/abs/1409.0473
6. **Raffel, C. et al.** (2019). "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer" (T5)
   - https://arxiv.org/abs/1910.10683
### Recursos Online
1. **"The Illustrated Transformer"** — Jay Alammar
   - http://jalammar.github.io/illustrated-transformer/
   - La mejor explicación visual del Transformer — **lectura recomendada**
2. **"The Illustrated BERT"** — Jay Alammar
   - http://jalammar.github.io/illustrated-bert/
   - Visualización del pre-entrenamiento y fine-tuning de BERT
3. **"The Annotated Transformer"** — Harvard NLP
   - http://nlp.seas.harvard.edu/2018/04/03/attention.html
   - Implementación línea a línea del paper original
4. **Stanford CS224N** — Natural Language Processing with Deep Learning
   - http://web.stanford.edu/class/cs224n/
   - Slides y videos de clase sobre Transformers
5. **Andrej Karpathy** — "Let's build GPT: from scratch, in code, spelled out"
   - https://www.youtube.com/watch?v=kCc8FmEb1nY
   - Video de 2h construyendo un GPT desde cero — **altamente recomendado**
### Tutoriales Interactivos
1. **Hugging Face Course** — Módulo 1: Transformer Models
   - https://huggingface.co/course/chapter1
   - Interactivo, con código ejecutable en la nube
2. **Bertviz** — Herramienta para visualizar atención en BERT/GPT
   - https://github.com/jessevig/bertviz
   - `pip install bertviz`
3. **Attention Playground** — Visualización interactiva de attention
   - https://poloclub.github.io/transformer-explainer/
4. **Google Colab Notebooks oficiales de Hugging Face**
   - Notebooks listos para ejecutar con GPU gratuita
   - https://github.com/huggingface/notebooks
### Documentación Técnica
1. **Hugging Face Transformers Documentation**
   - https://huggingface.co/docs/transformers
   - Referencia completa de la librería
2. **PyTorch nn.Transformer**
   - https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html
   - Implementación oficial en PyTorch
3. **Hugging Face Model Hub**
   - https://huggingface.co/models
   - Miles de modelos pre-entrenados disponibles
4. **Papers With Code** — Transformers
   - https://paperswithcode.com/methods/category/transformer-based-architectures
   - Estado del arte actualizado
## 🎓 Notas Finales
### Conceptos Clave para Recordar
1. **Self-Attention = Búsqueda Diferenciable**
   - Q (qué busco) · Kᵀ (qué ofrezco) → scores → softmax → pesos
   - Output = suma ponderada de Values
   - Complejidad O(n²·d) — cuello de botella para secuencias largas
2. **Multi-Head Attention = Múltiples Perspectivas**
   - h cabezas con d_k = d_model/h dimensiones cada una
   - Cada cabeza aprende relaciones diferentes
   - Concatenar + proyectar = misma dimensión que la entrada
3. **Positional Encoding = Inyectar Orden**
   - Self-Attention es invariante al orden → necesitamos indicar posición
   - Sinusoidal: deterministico, generaliza a secuencias largas
   - Aprendible (BERT): más flexible, limitado a longitud de entrenamiento
4. **Transformer Block = Atención + FFN + Residual + Norm**
   - Conexiones residuales: resuelven gradiente que desaparece
   - Layer Norm: estabiliza la distribución de activaciones
   - FFN: transforma las representaciones en el espacio de características
5. **Pre-Training + Fine-Tuning = Paradigma Dominante**
   - Pre-entrenamiento masivo captura conocimiento general del lenguaje
   - Fine-tuning con pocos datos adapta a la tarea específica
   - LR pequeño (2e-5 a 5e-5) previene el olvido catastrófico
6. **BERT vs. GPT = Comprensión vs. Generación**
   - BERT: encoder bidireccional, Masked LM, ideal para clasificación/QA
   - GPT: decoder causal, next token prediction, ideal para generación
   - T5: encoder-decoder, text-to-text para cualquier tarea
### 🎉 Preparación: ¡Has Completado los 12 Laboratorios!
¡**Felicitaciones**! Has completado exitosamente el ciclo completo de 12 laboratorios de Deep Learning. Este último laboratorio coronó tu formación con la arquitectura más influyente de la IA moderna.
**El viaje que recorriste:**
```
Lab 01 → Neuronas artificiales desde cero
Lab 02 → Primera red neuronal (forward pass)
Lab 03 → Funciones de activación (no linealidad)
Lab 04 → Funciones de pérdida (optimizar qué)
Lab 05 → Backpropagation (cómo aprende la red)
Lab 06 → Entrenamiento: SGD, Adam, regularización
Lab 07 → Métricas de evaluación y validación
Lab 08 → Frameworks: PyTorch y TensorFlow
Lab 09 → IA Generativa: GANs y VAEs
Lab 10 → CNNs para visión computacional
Lab 11 → RNNs/LSTMs para secuencias
Lab 12 → Transformers: la arquitectura del futuro ✅
```
**Tu próximo paso**: Con esta base sólida, estás listo para explorar:
- **Modelos de Lenguaje Grandes (LLMs)**: Llama, Mistral, fine-tuning con QLoRA/LoRA
- **IA Multimodal**: CLIP, DALL-E, Stable Diffusion, GPT-4V
- **Reinforcement Learning from Human Feedback (RLHF)**: cómo se entrena ChatGPT
- **Efficient Transformers**: FlashAttention, Longformer, Mamba (State Space Models)
- **Proyectos de investigación**: contribuye a Hugging Face, reproduce un paper
### Consejos de Estudio
1. **Lee "The Illustrated Transformer"**: Es la mejor introducción visual y toma menos de una hora. Hazlo antes de implementar.
2. **Construye desde cero**: Implementar Self-Attention manualmente con NumPy consolidará el concepto de forma que ninguna librería puede hacerlo.
3. **Visualiza los mapas de atención**: El Transformer es uno de los modelos más interpretables gracias a los attention weights. Úsalos.
4. **Experimenta con prompts**: Con GPT-2 o GPT-4 API, la ingeniería de prompts es una habilidad práctica inmediata.
5. **Hugging Face es tu aliado**: Domina la librería `transformers` — es el estándar de la industria para NLP.
6. **Conoce las limitaciones**: Los Transformers no son perfectos. Estudia Efficient Transformers (Longformer, BigBird, FlashAttention) para secuencias largas.
### Solución de Problemas Comunes
**Problema**: `CUDA out of memory` al cargar BERT o GPT-2
- **Causa**: GPU insuficiente o batch size muy grande
- **Solución 1**: Reducir `per_device_train_batch_size` a 8 o 4
- **Solución 2**: Usar `fp16=True` en TrainingArguments (requiere GPU con Tensor Cores)
- **Solución 3**: Usar `DistilBERT` en lugar de BERT-base (40% menos parámetros)
- **Solución 4**: Usar gradient checkpointing: `model.gradient_checkpointing_enable()`
**Problema**: Fine-tuning converge muy lento o no converge
- **Causa**: Learning rate inapropiado
- **Solución**: Usar learning rate entre 1e-5 y 5e-5 con warm-up (primeras 10% de iteraciones)
- **Verificar**: que `input_ids`, `attention_mask` y `labels` estén correctamente preparados
**Problema**: `RuntimeError: Expected all tensors to be on the same device`
- **Causa**: Modelo en GPU pero datos en CPU (o viceversa)
- **Solución**: `inputs = {k: v.to(device) for k, v in inputs.items()}`
**Problema**: Nan en loss durante el entrenamiento
- **Causa**: Learning rate demasiado alto o gradientes explosivos
- **Solución**: Gradient clipping: `max_grad_norm=1.0` en TrainingArguments
**Problema**: Tokenizador trunca textos importantes
- **Causa**: `max_length` por defecto (512 para BERT) es insuficiente
- **Solución**: Estrategia de sliding window para textos largos; usar modelos con contexto mayor (Longformer, BERT-large-512)
**Problema**: Attention weights contienen NaN
- **Causa**: Overflow en la exponencial de softmax con scores muy grandes
- **Solución**: Verificar que el scaling por √d_k está aplicado; usar estabilidad numérica (restar el máximo antes del exp)
**Problema**: Generación de texto con GPT-2 produce repeticiones
- **Causa**: Modo greedy o temperatura muy baja
- **Solución**: Usar `repetition_penalty=1.2` y `top_p=0.92` en `model.generate()`
### Comunidad y Soporte
- **Foro del curso**: Para preguntas técnicas sobre implementación
- **Horas de oficina**: Consultas sobre fine-tuning y proyectos avanzados
- **Hugging Face Forums**: https://discuss.huggingface.co — comunidad muy activa
- **Stack Overflow**: Tag `pytorch`, `huggingface-transformers`
- **Papers With Code**: https://paperswithcode.com — para comparar implementaciones
- **Discord de Hugging Face**: Para preguntas en tiempo real
### Lista de Verificación de Completitud
Has completado exitosamente el Lab 12 cuando puedes:
- [ ] Explicar el mecanismo de Self-Attention (Q, K, V) sin consultar notas
- [ ] Implementar Scaled Dot-Product Attention desde cero con NumPy
- [ ] Construir Multi-Head Attention y visualizar los mapas de cada cabeza
- [ ] Generar Positional Encoding sinusoidal y explicar su propiedad de periodicidad
- [ ] Construir un Transformer Encoder Block completo en PyTorch
- [ ] Fine-tunear DistilBERT/BERT para una tarea de clasificación
- [ ] Generar texto con GPT-2 usando temperature, top-k y top-p sampling
- [ ] Comparar la complejidad O(n²·d) de Transformers vs. O(n·d²) de RNNs
- [ ] Explicar la diferencia entre BERT (encoder) y GPT (decoder)
- [ ] Resolver al menos un ejercicio propuesto completo
---
**¡Felicitaciones por completar los 12 laboratorios de Deep Learning! 🎓🏆**

Has recorrido un camino desde la neurona artificial más simple hasta los Transformers que impulsan la IA más avanzada del mundo. Con esta base, estás preparado para investigar, desarrollar y contribuir al estado del arte en Deep Learning.

**"Attention Is All You Need"** — y ahora tú también sabes por qué. 🚀
---
*Última actualización: 2024*  
*Versión: 1.0*  
*Licencia: MIT - Uso educativo*
