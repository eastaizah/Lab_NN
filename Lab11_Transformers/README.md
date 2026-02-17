# Lab 11: Transformers

## 📋 Descripción

Este laboratorio cubre la arquitectura **Transformer** y el mecanismo de **Self-Attention**, que revolucionaron el Deep Learning y son la base de los modelos más avanzados de IA actual (GPT-4, BERT, DALL-E, ChatGPT, etc.).

## 🎯 Objetivos de Aprendizaje

Al completar este laboratorio, serás capaz de:

1. **Entender Self-Attention**: Comprender el mecanismo fundamental de atención con Q, K, V
2. **Implementar Multi-Head Attention**: Múltiples cabezas de atención en paralelo
3. **Aplicar Positional Encoding**: Codificar información de posición en secuencias
4. **Construir Transformer Blocks**: Ensamblar bloques encoder y decoder completos
5. **Fine-tunear BERT**: Adaptar BERT pre-entrenado para análisis de sentimiento
6. **Generar texto con GPT**: Usar GPT-2 para generación autoregresiva
7. **Visualizar Atención**: Interpretar y visualizar patrones de atención
8. **Comparar con RNNs**: Entender ventajas de Transformers sobre arquitecturas recurrentes

## 📚 Contenido

### 1. Teoría (`teoria.md`)

Documento teórico completo que cubre:

- **Motivación y limitaciones de RNNs/LSTMs**
- **Self-Attention Mechanism**: Q, K, V y Scaled Dot-Product
- **Multi-Head Attention**: Múltiples perspectivas en paralelo
- **Positional Encoding**: Sinusoidal y aprendido
- **Arquitectura Transformer Completa**: Encoder-Decoder
- **Variantes**: BERT (Encoder-only) vs GPT (Decoder-only)
- **Vision Transformers (ViT)**: Transformers para imágenes
- **Transfer Learning y Fine-tuning**: Pre-entrenamiento y adaptación
- **Visualización e Interpretabilidad**: Attention maps
- **Aplicaciones Modernas**: ChatGPT, DALL-E, AlphaFold, etc.

### 2. Código (`codigo/transformers.py`)

Implementaciones completas en Python:

#### Parte 1: Self-Attention (NumPy)
```python
class SelfAttentionNumPy:
    """Self-attention desde cero con NumPy"""
    
class MultiHeadAttentionNumPy:
    """Multi-head attention con múltiples cabezas"""
```

#### Parte 2: Positional Encoding
```python
class PositionalEncodingSinusoidal:
    """Positional encoding con funciones seno/coseno"""
```

#### Parte 3: Transformer Blocks (PyTorch)
```python
class TransformerEncoderBlock(nn.Module):
    """Bloque encoder: Self-Attention + FFN"""
    
class TransformerDecoderBlock(nn.Module):
    """Bloque decoder: Masked Attention + Cross-Attention + FFN"""
    
class TransformerModel(nn.Module):
    """Transformer completo (Encoder-Decoder)"""
```

#### Parte 4: Hugging Face
```python
class BERTSentimentClassifier:
    """Fine-tuning de BERT para sentimiento"""
    
class GPT2TextGenerator:
    """Generación de texto con GPT-2"""
```

### 3. Práctica (`practica.ipynb`)

Notebook Jupyter interactivo con:

- **Parte 1**: Self-Attention paso a paso
- **Parte 2**: Multi-Head Attention
- **Parte 3**: Positional Encoding y visualización
- **Parte 4**: Transformer Blocks completos
- **Parte 5**: BERT Fine-tuning para análisis de sentimiento
- **Parte 6**: GPT-2 Text Generation
- **Parte 7**: Comparación con RNNs/LSTMs
- **Parte 8**: Ejercicios y proyectos avanzados
- Clase `TransformerBlock`: Bloque completo
- Ejemplos con Hugging Face Transformers
- Fine-tuning para tareas específicas

## Cómo Usar Este Laboratorio

### Opción 1: Jupyter Notebook (Recomendado)

```bash
# Desde el directorio del repositorio
cd Lab11_Transformers
jupyter notebook practica.ipynb
```

### Opción 2: Script Python

```bash
# Ejecutar el código de ejemplo
python codigo/transformers.py
```

### Opción 3: Lectura y Experimentación

1. Lee `teoria.md` para entender los conceptos
2. Abre `practica.ipynb` en Jupyter
3. Ejecuta cada celda y experimenta con los parámetros
4. Completa los ejercicios propuestos
5. Revisa `codigo/transformers.py` como referencia

## Requisitos

```bash
pip install numpy matplotlib jupyter torch transformers datasets
```

## Conceptos Clave

- **Attention**: Mecanismo para enfocarse en partes relevantes del input
- **Self-Attention**: Atención sobre la misma secuencia
- **Query, Key, Value**: Tres proyecciones lineales para calcular atención
- **Multi-Head Attention**: Múltiples atenciones en paralelo
- **Positional Encoding**: Información de posición sin recurrencia
- **Feed-Forward Network**: Red densa después de atención
- **Layer Normalization**: Normalización para estabilidad
- **Encoder**: Procesa input (ej: BERT)
- **Decoder**: Genera output (ej: GPT)

## Ejercicios

### Ejercicio 11.1: Self-Attention Manual
Calcula attention scores manualmente para una secuencia pequeña.

### Ejercicio 11.2: Positional Encoding
Implementa y visualiza positional encoding sinusoidal.

### Ejercicio 11.3: Multi-Head Attention
Completa la implementación desde cero.

### Ejercicio 11.4: Fine-tuning BERT
Entrena BERT para clasificación de sentimientos.

### Ejercicio 11.5: Generación con GPT (Desafío)
Usa GPT-2 para generar texto coherente.

## Ventajas de Transformers sobre RNNs

1. **Paralelización**: Procesa toda la secuencia en paralelo (vs secuencial en RNN)
2. **Dependencias Largas**: Captura relaciones a cualquier distancia
3. **Menos Bias Inductivo**: Aprende estructura desde datos
4. **Escalabilidad**: Funciona mejor con más datos y parámetros
5. **Interpretabilidad**: Attention weights son visualizables

## Arquitecturas Transformer Famosas

### BERT (2018) - Google

**Características:**
- Encoder-only (bidireccional)
- Pre-training: Masked Language Modeling (MLM)
- 340M parámetros (BERT-large)
- Estado del arte en comprensión de lenguaje

**Aplicaciones:**
- Clasificación de texto
- Question Answering
- Named Entity Recognition
- Sentence similarity

### GPT (2018-2023) - OpenAI

**GPT-1/2/3/4:**
- Decoder-only (autoregresivo)
- Pre-training: Next token prediction
- GPT-3: 175B parámetros
- GPT-4: multimodal

**Aplicaciones:**
- Generación de texto
- Traducción
- Resumen
- Conversación (ChatGPT)

### T5 (2019) - Google

**Características:**
- Encoder-Decoder completo
- Text-to-Text framework
- Todas las tareas como text generation

### Vision Transformer (ViT) - 2020

**Innovación:**
- Aplica Transformers a imágenes
- Divide imagen en patches
- Trata patches como "tokens"
- Supera CNNs en grandes datasets

### Otras Variantes

- **RoBERTa**: BERT mejorado
- **ALBERT**: BERT más eficiente
- **ELECTRA**: Pre-training más eficiente
- **DeBERTa**: Disentangled attention
- **Llama**: Open-source de Meta
- **Claude**: Anthropic
- **Gemini**: Google multimodal

## Aplicaciones

### Procesamiento de Lenguaje Natural

1. **Comprensión**:
   - Clasificación de texto
   - Análisis de sentimientos
   - Named Entity Recognition
   - Question Answering

2. **Generación**:
   - Traducción automática
   - Resumen de texto
   - Generación creativa
   - Diálogo (chatbots)

3. **Representación**:
   - Embeddings contextuales
   - Similarity search
   - Clustering semántico

### Visión Computacional

- **ViT**: Clasificación de imágenes
- **DETR**: Detección de objetos
- **Segmenter**: Segmentación semántica
- **CLIP**: Visión-lenguaje

### Multimodal

- **CLIP**: Imagen + Texto
- **DALL-E**: Texto → Imagen
- **Flamingo**: Visión + Lenguaje
- **GPT-4**: Multimodal completo

### Audio

- **Whisper**: Speech recognition
- **AudioLM**: Generación de audio
- **MusicGen**: Generación de música

### Ciencia

- **AlphaFold**: Predicción de proteínas
- **ESM**: Modelado de secuencias de proteínas
- **Molecule generation**: Diseño de fármacos

## Mecanismo de Atención

### Scaled Dot-Product Attention

```python
Attention(Q, K, V) = softmax(Q @ K^T / √d_k) @ V

Donde:
- Q (queries): qué buscar
- K (keys): qué se ofrece
- V (values): qué contenido devolver
- d_k: dimensión de keys (para escalar)
```

**Ejemplo intuitivo:**
```
"El gato bebió la leche porque estaba hambriento"

Al procesar "estaba", attention se enfoca en "gato" (no "leche")
porque el modelo aprendió que "hambriento" se refiere al sujeto.
```

### Multi-Head Attention

```python
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) @ W_O

head_i = Attention(Q @ W_Q^i, K @ W_K^i, V @ W_V^i)
```

**Ventaja:**
- Cada "head" puede enfocarse en diferentes aspectos
- Head 1: sintaxis
- Head 2: semántica
- Head 3: contexto largo
- etc.

## Arquitectura Transformer Completa

### Encoder (ej: BERT)

```
Input Tokens
    ↓
Token Embeddings + Positional Encoding
    ↓
[Multi-Head Self-Attention
    ↓
Add & Norm
    ↓
Feed-Forward Network
    ↓
Add & Norm] × N layers
    ↓
Output Representations
```

### Decoder (ej: GPT)

```
Output Tokens (shifted)
    ↓
Token Embeddings + Positional Encoding
    ↓
[Masked Multi-Head Self-Attention
    ↓
Add & Norm
    ↓
Feed-Forward Network
    ↓
Add & Norm] × N layers
    ↓
Linear → Softmax
    ↓
Next Token Probabilities
```

### Encoder-Decoder (ej: T5)

```
Encoder              Decoder
Input → [Enc Blocks] → [Dec Blocks] → Output
                ↓
        Cross-Attention
```

## Positional Encoding

**Problema:** Self-attention es permutation invariant (sin orden).

**Solución:** Agregar información de posición.

**Sinusoidal (original):**
```python
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**Aprendido (alternativa):**
- Embeddings de posición entrenables
- Usado en BERT, GPT

## Transfer Learning con Transformers

### 1. Pre-training (costoso, una vez)

**BERT:**
- Masked Language Modeling (MLM)
- Next Sentence Prediction (NSP)

**GPT:**
- Next token prediction

### 2. Fine-tuning (rápido, por tarea)

```python
# Cargar modelo pre-entrenado
model = BertForSequenceClassification.from_pretrained('bert-base')

# Fine-tune en datos específicos
train(model, task_data)
```

### 3. Prompt Engineering (sin fine-tuning)

**Few-shot learning:**
```
Clasifica sentimiento:
"La película fue increíble" → Positivo
"No me gustó nada" → Negativo
"Estuvo bien" → ?
```

## Limitaciones de Transformers

1. **Costo Computacional**: O(n²) en longitud de secuencia
2. **Memoria**: Attention matrix crece cuadráticamente
3. **Datos**: Requieren grandes cantidades para pre-training
4. **Interpretabilidad**: Modelos muy grandes son cajas negras
5. **Sesgo**: Heredan sesgos de datos de entrenamiento

## Mejoras Recientes

### Efficient Transformers

- **Linformer**: O(n) complexity
- **Performer**: Kernel approximation
- **Longformer**: Sparse attention
- **Big Bird**: Sparse + global attention

### Scaling Laws

- Más parámetros → mejor rendimiento (hasta ~)
- Compute-optimal: balance datos/parámetros
- Chinchilla scaling laws

## Notas Importantes

⚠️ **GPU Requerida**: Transformers grandes requieren GPUs potentes.

💡 **Hugging Face**: Biblioteca estándar para usar modelos pre-entrenados.

🚀 **Fine-tuning > Training from Scratch**: Casi siempre mejor usar modelo pre-entrenado.

⚡ **Prompting**: Para modelos muy grandes (GPT-4), prompting puede ser suficiente sin fine-tuning.

## Próximo Paso

Transformers son la base de modelos generativos modernos:

👉 **Vuelve a [Lab 08: IA Generativa](../Lab08_IA_Generativa/)** con nuevo contexto sobre Transformers para entender mejor modelos como DALL-E, GPT, etc.

## Recursos Adicionales

### Papers Fundamentales
- **Attention Is All You Need** (2017) - Paper original de Transformers
- **BERT**: Pre-training of Deep Bidirectional Transformers (2018)
- **Language Models are Few-Shot Learners** (GPT-3, 2020)
- **An Image is Worth 16x16 Words** (ViT, 2020)

### Tutoriales
- [The Illustrated Transformer - Jay Alammar](http://jalammar.github.io/illustrated-transformer/)
- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
- [Hugging Face Course](https://huggingface.co/course)
- [Stanford CS224N](http://web.stanford.edu/class/cs224n/)

### Herramientas
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Hugging Face Datasets](https://huggingface.co/datasets)
- [Weights & Biases](https://wandb.ai/) - Para tracking experiments

## Preguntas Frecuentes

**P: ¿Por qué Transformers son mejores que RNNs?**  
R: Procesan en paralelo (más rápido), capturan dependencias largas mejor, y escalan mejor con más datos/parámetros.

**P: ¿Cuándo usar BERT vs GPT?**  
R: BERT para comprensión (clasificación, Q&A). GPT para generación (texto, traducción, diálogo).

**P: ¿Puedo entrenar Transformers desde cero?**  
R: Posible pero costoso. Para proyectos, usar modelos pre-entrenados y hacer fine-tuning.

**P: ¿Qué es mejor: más heads o más layers?**  
R: Más layers típicamente ayuda más. Heads: 8-16 es estándar, más no siempre ayuda.

**P: ¿Transformers solo para NLP?**  
R: No. ViT para imágenes, Transformers para audio, video, proteínas, y más. Es una arquitectura general.

## Verificación de Conocimientos

- [ ] Entiendo el mecanismo de Self-Attention (Q, K, V)
- [ ] Puedo explicar Multi-Head Attention
- [ ] Entiendo por qué se necesita Positional Encoding
- [ ] Conozco la diferencia entre Encoder (BERT) y Decoder (GPT)
- [ ] Puedo implementar Self-Attention desde cero
- [ ] Sé usar Hugging Face para fine-tuning
- [ ] Entiendo las ventajas de Transformers sobre RNNs
- [ ] Conozco aplicaciones más allá de NLP

## Conclusión

**Transformers han revolucionado el deep learning:**

- ✅ Estado del arte en NLP
- ✅ Emergiendo en visión computacional
- ✅ Modelos multimodales potentes
- ✅ Base para IA generativa moderna
- ✅ Arquitectura general para muchos dominios

**"Attention is All You Need"** - y tenían razón! 🚀

---

**¡Has completado los laboratorios de Neural Networks!** 🎓

Ahora tienes las bases para entender y aplicar:
- Redes neuronales desde cero
- CNNs para visión
- RNNs/LSTMs para secuencias
- Transformers para todo
- IA Generativa moderna
