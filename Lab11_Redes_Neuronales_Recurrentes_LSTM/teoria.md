# Teoría: Redes Neuronales Recurrentes (RNN) y LSTM

## 1. Introducción

Las Redes Neuronales Recurrentes (RNN) son una clase de redes neuronales diseñadas específicamente para procesar datos secuenciales, donde el orden de los elementos importa.

### ¿Por qué RNNs para Secuencias?

**Problema con Redes Densas:**
- No capturan dependencias temporales
- Tamaño de entrada fijo (no pueden manejar secuencias de longitud variable)
- No comparten parámetros a través del tiempo
- No tienen memoria de eventos pasados

**Ejemplos de Datos Secuenciales:**
- Texto: palabras en una oración
- Series temporales: precio de acciones, temperatura
- Audio: señales de voz
- Video: secuencias de frames
- ADN: secuencias de nucleótidos

**Solución: RNNs**
- Procesan secuencias de cualquier longitud
- Mantienen un "estado oculto" como memoria
- Comparten parámetros a través del tiempo
- Capturan patrones temporales y dependencias

## 2. Arquitectura de RNN

### 2.1 Estructura Básica

Una RNN procesa una secuencia paso a paso, manteniendo un estado oculto que se actualiza en cada paso temporal.

**Ecuaciones Fundamentales:**

```
h_t = tanh(W_hh · h_{t-1} + W_xh · x_t + b_h)
y_t = W_hy · h_t + b_y
```

Donde:
- `x_t`: entrada en el tiempo t
- `h_t`: estado oculto en el tiempo t
- `y_t`: salida en el tiempo t
- `W_hh`: pesos recurrentes (hidden-to-hidden)
- `W_xh`: pesos de entrada (input-to-hidden)
- `W_hy`: pesos de salida (hidden-to-output)
- `b_h, b_y`: sesgos

**Visualización:**

```
Forma desplegada (unfolded):

x_0 → [RNN] → h_0 → y_0
        ↓
x_1 → [RNN] → h_1 → y_1
        ↓
x_2 → [RNN] → h_2 → y_2
        ↓
x_3 → [RNN] → h_3 → y_3

Forma compacta:
       ↻ (bucle recurrente)
x_t → [RNN] → y_t
```

### 2.2 Tipos de Secuencias

**One-to-One**: Entrada → Salida (red neuronal estándar)
- Ejemplo: Clasificación de imagen

**One-to-Many**: Una entrada → Secuencia de salidas
- Ejemplo: Generación de texto a partir de una imagen (image captioning)

**Many-to-One**: Secuencia de entradas → Una salida
- Ejemplo: Análisis de sentimiento (toda la oración → positivo/negativo)

**Many-to-Many (sincronizada)**: Secuencia → Secuencia (misma longitud)
- Ejemplo: Etiquetado de partes del discurso (POS tagging)

**Many-to-Many (asíncrona)**: Secuencia → Secuencia (diferentes longitudes)
- Ejemplo: Traducción automática (Encoder-Decoder)

## 3. Backpropagation Through Time (BPTT)

### 3.1 Concepto

Backpropagation Through Time es el algoritmo para entrenar RNNs. Consiste en "desplegar" la red a través del tiempo y aplicar backpropagation estándar.

**Proceso:**
1. **Forward Pass**: Procesar toda la secuencia hacia adelante
2. **Calcular pérdida**: Sumar pérdidas en todos los pasos temporales
3. **Backward Pass**: Calcular gradientes retrocediendo en el tiempo
4. **Actualizar pesos**: Aplicar optimización

### 3.2 Cálculo de Gradientes

La pérdida total es la suma de pérdidas en cada paso temporal:

```
L = Σ L_t    (suma sobre todos los pasos de tiempo)
```

Para calcular ∂L/∂W_hh, necesitamos aplicar la regla de la cadena a través del tiempo:

```
∂L/∂W_hh = Σ ∂L_t/∂W_hh
```

El gradiente fluye hacia atrás en el tiempo:

```
∂L/∂h_t = ∂L/∂y_t · ∂y_t/∂h_t + ∂L/∂h_{t+1} · ∂h_{t+1}/∂h_t
```

### 3.3 Truncated BPTT

Para secuencias muy largas, BPTT completo es costoso. **Truncated BPTT** limita el número de pasos hacia atrás:

- Procesar secuencia en bloques de k pasos
- Solo retropropagar k pasos hacia atrás
- Reduce memoria y computación
- Sacrifica algo de información de largo plazo

## 4. Problema del Gradiente Desvaneciente/Explosivo

### 4.1 El Problema

**Gradiente Desvaneciente:**

Al retropropagar a través de muchos pasos temporales, los gradientes se multiplican repetidamente:

```
∂h_t/∂h_0 = ∏ ∂h_i/∂h_{i-1}    (producto de muchos términos)
```

Si cada término < 1 (común con tanh, sigmoid):
- Producto tiende a 0 exponencialmente
- Gradientes desaparecen
- No se aprenden dependencias de largo plazo

**Gradiente Explosivo:**

Si los términos son > 1:
- Producto crece exponencialmente
- Gradientes explotan (→ ∞)
- Inestabilidad numérica
- Pesos oscilan violentamente

### 4.2 Análisis Matemático

Derivada de tanh:

```
∂tanh(x)/∂x = 1 - tanh²(x)
```

Para |tanh(x)| cercano a 1: derivada ≈ 0 (saturación)

Si W_hh tiene autovalores λ:
- |λ| < 1: gradientes se desvanecen
- |λ| > 1: gradientes explotan
- Necesitamos |λ| ≈ 1 para estabilidad

### 4.3 Soluciones

**Para Gradientes Explosivos:**
- **Gradient Clipping**: Limitar la norma del gradiente
  ```python
  if ||g|| > threshold:
      g = threshold * g / ||g||
  ```

**Para Gradientes Desvanecientes:**
- **LSTM/GRU**: Arquitecturas especializadas (ver sección 5)
- **Inicialización cuidadosa**: Inicializar pesos cerca de identidad
- **ReLU**: Usar ReLU en lugar de tanh (menos común en RNN)
- **Skip connections**: Conexiones residuales

## 5. Long Short-Term Memory (LSTM)

### 5.1 Motivación

Los LSTM fueron diseñados específicamente para resolver el problema del gradiente desvaneciente y capturar dependencias de largo plazo.

**Idea Clave:** Añadir un "camino directo" para el flujo de información (cell state) que permite que la información fluya sin cambios a través del tiempo.

### 5.2 Arquitectura LSTM

Un LSTM tiene **tres puertas (gates)** y un **estado de celda (cell state)**:

1. **Forget Gate (f_t)**: Decide qué información olvidar del cell state
2. **Input Gate (i_t)**: Decide qué información nueva añadir al cell state
3. **Output Gate (o_t)**: Decide qué parte del cell state exponer como output

**Ecuaciones Completas:**

```
# Forget Gate: ¿Qué olvidar del estado anterior?
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)

# Input Gate: ¿Qué información nueva guardar?
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)

# Candidato a nuevo cell state
C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)

# Actualizar Cell State
C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t

# Output Gate: ¿Qué exponer como salida?
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)

# Hidden State (salida)
h_t = o_t ⊙ tanh(C_t)
```

Donde:
- σ: función sigmoide (salidas entre 0 y 1)
- ⊙: producto elemento a elemento (Hadamard)
- [h_{t-1}, x_t]: concatenación de vectores

### 5.3 Funcionamiento de las Puertas

**Forget Gate (f_t):**
```
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
```
- Salida: vector con valores entre 0 y 1
- 0 = olvidar completamente
- 1 = recordar completamente
- Decide qué información del C_{t-1} mantener

**Input Gate (i_t) y Candidato (C̃_t):**
```
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)
```
- i_t: Decide cuánta información nueva añadir
- C̃_t: Vector de candidatos a nuevos valores
- i_t ⊙ C̃_t: Información nueva filtrada

**Cell State Update:**
```
C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t
     └─ recordar ─┘   └─ actualizar ─┘
```
- Primera parte: Información antigua filtrada
- Segunda parte: Información nueva filtrada
- Suma: Estado actualizado

**Output Gate (o_t):**
```
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
h_t = o_t ⊙ tanh(C_t)
```
- o_t: Decide qué partes del cell state exponer
- tanh(C_t): Cell state normalizado a [-1, 1]
- h_t: Hidden state resultante (salida)

### 5.4 ¿Por qué LSTM Funciona Mejor?

**Flujo de Gradiente:**
- El cell state C_t proporciona un "camino directo" para gradientes
- ∂C_t/∂C_{t-1} = f_t (multiplicación simple, no matriz)
- Si f_t ≈ 1, el gradiente fluye sin cambios
- Evita las múltiples multiplicaciones de matrices que causan desvanecimiento

**Capacidad Adaptativa:**
- Las puertas aprenden cuándo recordar/olvidar
- Pueden mantener información relevante indefinidamente
- Pueden descartar información irrelevante rápidamente

### 5.5 Variantes de LSTM

**LSTM con Peephole Connections:**
Las puertas miran directamente al cell state:
```
f_t = σ(W_f · [C_{t-1}, h_{t-1}, x_t] + b_f)
i_t = σ(W_i · [C_{t-1}, h_{t-1}, x_t] + b_i)
o_t = σ(W_o · [C_t, h_{t-1}, x_t] + b_o)
```

**Coupled Forget-Input Gates:**
En lugar de forget e input independientes:
```
f_t = σ(...)
i_t = 1 - f_t  # Cuando olvida, añade; cuando recuerda, no añade
```

## 6. Gated Recurrent Unit (GRU)

### 6.1 Arquitectura

GRU es una simplificación de LSTM con **solo 2 puertas** (en lugar de 3):

**Ecuaciones:**

```
# Reset Gate: ¿Cuánto del pasado olvidar?
r_t = σ(W_r · [h_{t-1}, x_t] + b_r)

# Update Gate: ¿Cuánto del estado anterior mantener?
z_t = σ(W_z · [h_{t-1}, x_t] + b_z)

# Candidato a nuevo hidden state
h̃_t = tanh(W_h · [r_t ⊙ h_{t-1}, x_t] + b_h)

# Hidden state actualizado
h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
```

### 6.2 Diferencias con LSTM

**Simplificaciones:**
1. **No hay cell state separado**: Solo hidden state
2. **Menos puertas**: 2 en lugar de 3
3. **Menos parámetros**: Aproximadamente 25% menos que LSTM
4. **Más rápido**: Menos operaciones por paso

**Update Gate vs Forget+Input:**
- GRU combina forget y input en una sola puerta (z_t)
- z_t controla balance entre información pasada y nueva
- Cuando z_t → 1: usar h̃_t (información nueva)
- Cuando z_t → 0: mantener h_{t-1} (información vieja)

**Reset Gate:**
- Controla cuánta información pasada usar para h̃_t
- Permite "olvidar" información irrelevante al calcular candidato

### 6.3 LSTM vs GRU

| Aspecto | LSTM | GRU |
|---------|------|-----|
| **Puertas** | 3 (forget, input, output) | 2 (reset, update) |
| **Parámetros** | Más (4 transformaciones) | Menos (3 transformaciones) |
| **Cell State** | Separado del hidden state | Unificado |
| **Velocidad** | Más lento | Más rápido |
| **Memoria** | Mayor | Menor |
| **Rendimiento** | Mejor en tareas complejas | Similar en muchas tareas |

**¿Cuándo usar cada uno?**
- **LSTM**: Tareas con dependencias muy largas, datos abundantes
- **GRU**: Datasets pequeños, recursos limitados, prototipado rápido
- **Regla práctica**: Probar ambos, GRU es buen punto de partida

## 7. Arquitecturas Avanzadas

### 7.1 Bidirectional RNN/LSTM

**Motivación:** A veces necesitamos contexto futuro además del pasado.

**Arquitectura:**
```
Forward RNN:  x_1 → x_2 → x_3 → x_4 → x_5
              h_1→ h_2→ h_3→ h_4→ h_5→

Backward RNN: x_1 ← x_2 ← x_3 ← x_4 ← x_5
              h_1← h_2← h_3← h_4← h_5←

Output: Concatenación [h_t→, h_t←]
```

**Ecuaciones:**
```
# Forward pass
h_t→ = RNN(x_t, h_{t-1}→)

# Backward pass  
h_t← = RNN(x_t, h_{t+1}←)

# Combinar
h_t = [h_t→; h_t←]  # Concatenación
```

**Aplicaciones:**
- NLP: POS tagging, NER (Named Entity Recognition)
- Traducción automática
- Speech recognition
- Cualquier tarea donde todo el contexto está disponible

**Limitación:** No se puede usar en tiempo real (necesita toda la secuencia)

### 7.2 Stacked (Deep) RNN/LSTM

**Motivación:** Más capas = más capacidad de abstracción

**Arquitectura:**
```
Capa 3:  [RNN] → [RNN] → [RNN] → [RNN]
           ↑       ↑       ↑       ↑
Capa 2:  [RNN] → [RNN] → [RNN] → [RNN]
           ↑       ↑       ↑       ↑
Capa 1:  [RNN] → [RNN] → [RNN] → [RNN]
           ↑       ↑       ↑       ↑
Input:   x_1     x_2     x_3     x_4
```

**Ecuaciones:**
```
h_t^1 = RNN^1(x_t, h_{t-1}^1)
h_t^2 = RNN^2(h_t^1, h_{t-1}^2)
h_t^3 = RNN^3(h_t^2, h_{t-1}^3)
...
y_t = Output(h_t^L)
```

**Ventajas:**
- Jerarquía de representaciones
- Mayor capacidad de modelado
- Mejora rendimiento en tareas complejas

**Desventajas:**
- Más lento
- Más propenso a overfitting
- Más difícil de entrenar

**Práctica común:**
- 2-4 capas en la mayoría de aplicaciones
- Añadir dropout entre capas
- Más capas no siempre es mejor

### 7.3 Encoder-Decoder (Seq2Seq)

**Arquitectura para traducción y generación:**

```
Encoder (comprime input):
x_1 → x_2 → x_3 → x_4 → [Context Vector]

Decoder (genera output):
[Context Vector] → y_1 → y_2 → y_3 → y_4 → y_5
```

**Componentes:**
1. **Encoder**: RNN/LSTM que procesa secuencia de entrada
2. **Context Vector**: Representación fija del input (último hidden state)
3. **Decoder**: RNN/LSTM que genera secuencia de salida

**Ecuaciones:**
```
# Encoder
h_t^enc = Encoder(x_t, h_{t-1}^enc)
context = h_T^enc  # Último estado

# Decoder
h_0^dec = context  # Inicializar con contexto
h_t^dec = Decoder(y_{t-1}, h_{t-1}^dec)
y_t = Output(h_t^dec)
```

**Limitación:** Cuello de botella en context vector fijo
**Solución:** Mecanismo de atención (ver Lab11: Transformers)

## 8. Aplicaciones

### 8.1 Procesamiento de Lenguaje Natural (NLP)

**Clasificación de Texto:**
- Análisis de sentimiento (positivo/negativo/neutral)
- Detección de spam
- Categorización de documentos
- Arquitectura: Many-to-One

**Generación de Texto:**
- Autocompletado
- Generación de historias
- Chatbots
- Arquitectura: Many-to-Many

**Traducción Automática:**
- Traducir entre idiomas
- Arquitectura: Encoder-Decoder
- Ejemplo: "Hello world" → "Hola mundo"

**Named Entity Recognition (NER):**
- Identificar nombres, lugares, organizaciones
- Arquitectura: Many-to-Many (sincronizada)
- Ejemplo: "Juan vive en Madrid" → [PERSONA, O, O, LUGAR]

**POS Tagging:**
- Etiquetar partes del discurso
- Arquitectura: Many-to-Many (sincronizada)
- Ejemplo: "El gato come" → [DET, NOUN, VERB]

### 8.2 Series Temporales

**Predicción de Series Temporales:**
- Precio de acciones
- Demanda de energía
- Tráfico web
- Clima
- Arquitectura: Many-to-One o Many-to-Many

**Detección de Anomalías:**
- Fraude en transacciones
- Fallos en equipos
- Comportamiento inusual en redes
- Arquitectura: Autoencoder con RNN/LSTM

**Clasificación de Series Temporales:**
- Actividad humana (acelerómetro)
- Arritmias cardíacas (ECG)
- Patrones de comportamiento
- Arquitectura: Many-to-One

### 8.3 Audio y Voz

**Speech Recognition:**
- Audio → Texto
- Asistentes virtuales (Siri, Alexa)
- Subtítulos automáticos
- Arquitectura: Encoder-Decoder con atención

**Music Generation:**
- Generar música nota por nota
- Continuar melodías
- Arquitectura: Many-to-Many

**Speaker Identification:**
- Identificar quién está hablando
- Arquitectura: Many-to-One

### 8.4 Video y Secuencias de Imágenes

**Video Classification:**
- Clasificar acciones en videos
- Detección de eventos
- Arquitectura: CNN + RNN/LSTM

**Video Captioning:**
- Describir qué sucede en un video
- Arquitectura: CNN (extracción) + RNN (descripción)

**Action Recognition:**
- Detectar acciones humanas
- Análisis deportivo
- Arquitectura: 3D CNN + LSTM

## 9. Técnicas de Entrenamiento

### 9.1 Gradient Clipping

```python
# Limitar la norma del gradiente
max_norm = 5.0
if gradient_norm > max_norm:
    gradient = gradient * (max_norm / gradient_norm)
```

**Beneficios:**
- Previene explosión de gradientes
- Estabiliza entrenamiento
- Permite usar learning rates más altos

### 9.2 Teacher Forcing

En Seq2Seq, durante entrenamiento usar salida real (no predicha) como entrada del siguiente paso:

```python
# Con Teacher Forcing
decoder_input = target_sequence[t]  # Ground truth

# Sin Teacher Forcing
decoder_input = decoder_output[t]  # Predicción
```

**Ventajas:**
- Entrenamiento más rápido y estable
- Gradientes más informativos

**Desventajas:**
- Discrepancia entre entrenamiento e inferencia
- Puede causar errores que se acumulan en producción

**Solución:** Scheduled sampling (mezclar ambos enfoques)

### 9.3 Regularización

**Dropout en RNN:**
```python
# Aplicar dropout SOLO en conexiones no recurrentes
# NO en conexiones recurrentes (causa problemas)
dropout_input = Dropout(x_t)
dropout_output = Dropout(h_t)
```

**Recurrent Dropout:**
- Aplicar la misma máscara de dropout en todos los pasos temporales
- Evita interferir con flujo de información temporal

**L2 Regularization:**
- Penalizar pesos grandes
- Reducir overfitting

### 9.4 Optimización

**Optimizadores recomendados:**
- **Adam**: Buen punto de partida (adaptativo)
- **RMSprop**: Bueno para RNNs
- **SGD con momentum**: Si tienes tiempo para tuning

**Learning Rate:**
- Típicamente 0.001 - 0.01 para Adam
- Usar learning rate decay
- Learning rate warmup para estabilidad inicial

**Batch Size:**
- Secuencias variables: usar padding y packing
- Ordenar secuencias por longitud para eficiencia
- Típicamente 32-128

## 10. Limitaciones y Consideraciones

### 10.1 Limitaciones de RNN/LSTM

**Procesamiento Secuencial:**
- No se puede paralelizar fácilmente a través del tiempo
- Más lento que arquitecturas paralelas (Transformers)
- Limitación en hardware moderno (GPUs)

**Dependencias Muy Largas:**
- Incluso LSTM tiene límites prácticos (~200-300 pasos)
- Para secuencias muy largas, considerar Transformers
- O truncar/muestrear secuencia

**Memoria:**
- Almacenar estados de todos los pasos temporales
- BPTT requiere mucha memoria
- Usar truncated BPTT para secuencias largas

### 10.2 Cuándo NO Usar RNN/LSTM

**Usar CNN si:**
- Datos tienen estructura espacial (imágenes)
- No hay dependencia temporal fuerte
- Necesitas velocidad de procesamiento

**Usar Transformers si:**
- Secuencias muy largas (>500 tokens)
- Necesitas paralelización
- Tareas de NLP modernas
- Tienes recursos computacionales abundantes

**Usar redes densas si:**
- Datos tabulares sin estructura secuencial
- Features independientes
- Clasificación simple

### 10.3 Buenas Prácticas

**Preprocesamiento:**
- Normalizar datos de entrada
- Padding inteligente (pack_padded_sequence en PyTorch)
- Tokenización apropiada para texto

**Arquitectura:**
- Empezar simple (1 capa, vanilla LSTM)
- Añadir complejidad gradualmente
- Probar tanto LSTM como GRU

**Entrenamiento:**
- Monitorear gradientes (detectar explosión/desvanecimiento)
- Usar gradient clipping
- Validación cruzada temporal (no aleatoria)
- Early stopping basado en validation loss

**Debugging:**
- Verificar que loss decrece en dataset pequeño
- Visualizar hidden states
- Verificar que gradientes fluyen
- Comparar con baseline simple

## 11. Matemáticas Detalladas

### 11.1 Dimensiones en LSTM

Para un LSTM con:
- Input dimension: `d_x`
- Hidden dimension: `d_h`
- Batch size: `B`
- Sequence length: `T`

**Parámetros:**
```
x_t: (B, d_x)
h_t: (B, d_h)
C_t: (B, d_h)

W_f, W_i, W_C, W_o: (d_h, d_x + d_h)
b_f, b_i, b_C, b_o: (d_h,)
```

**Total de parámetros por LSTM cell:**
```
4 * (d_h * (d_x + d_h) + d_h) = 4 * d_h * (d_x + d_h + 1)
```

**Ejemplo:** d_x=100, d_h=256
```
Total = 4 * 256 * (100 + 256 + 1) = 365,568 parámetros
```

### 11.2 Derivadas para BPTT

**Gradiente de la función de pérdida respecto al hidden state:**

```
∂L/∂h_t = ∂L_t/∂h_t + ∂L_{t+1}/∂h_t

donde:
∂L_{t+1}/∂h_t = ∂L_{t+1}/∂h_{t+1} · ∂h_{t+1}/∂h_t
```

**Para RNN vanilla:**
```
∂h_t/∂h_{t-1} = W_hh^T · diag(1 - tanh²(a_t))

donde a_t = W_hh·h_{t-1} + W_xh·x_t + b_h
```

**Para LSTM (gradiente a través del cell state):**
```
∂C_t/∂C_{t-1} = f_t

Más directo, evita multiplicación de matrices
```

### 11.3 Análisis de Estabilidad

Para que gradientes no se desvanezcan ni exploten:

```
||∂h_t/∂h_{t-k}|| ≈ ||W_hh||^k · ∏ ||diag(σ')||

Necesitamos: ||W_hh|| ≈ 1
```

**Normas espectrales:**
- λ_max(W_hh) < 1: gradientes se desvanecen
- λ_max(W_hh) > 1: gradientes explotan
- λ_max(W_hh) ≈ 1: flujo estable (difícil de mantener)

**LSTM resuelve esto:**
- Cell state tiene flujo aditivo (no multiplicativo)
- ∂C_t/∂C_0 = ∏ f_i (producto de valores 0-1, más controlable)
- Gates aprenden cuándo mantener gradiente ≈ 1

## 12. Resumen y Comparación

### 12.1 Tabla Comparativa

| Modelo | Ventajas | Desventajas | Uso Típico |
|--------|----------|-------------|------------|
| **RNN Vanilla** | Simple, pocos parámetros | Gradiente desvaneciente | Secuencias cortas, baselines |
| **LSTM** | Captura dependencias largas, estable | Lento, muchos parámetros | NLP complejo, series temporales |
| **GRU** | Más rápido que LSTM, menos parámetros | Menos expresivo que LSTM | Recursos limitados, prototipos |
| **Bidirectional** | Contexto completo | No en tiempo real | POS tagging, NER |
| **Stacked** | Más capacidad | Lento, overfitting | Tareas complejas con datos |

### 12.2 Evolución Histórica

```
1986: RNN Vanilla (Rumelhart)
      ↓
1997: LSTM (Hochreiter & Schmidhuber) - Resuelve gradiente desvaneciente
      ↓
2000s: Bidirectional RNN, aplicaciones en speech
      ↓
2014: GRU (Cho) - Simplificación de LSTM
      ↓
2014: Seq2Seq (Sutskever) - Traducción automática
      ↓
2015: Attention Mechanism - Mejora Seq2Seq
      ↓
2017: Transformer - Reemplaza RNN en muchas tareas
```

### 12.3 Estado Actual (2024)

**RNN/LSTM siguen siendo útiles para:**
- Series temporales (finanzas, sensores)
- Secuencias cortas y medianas (<500 pasos)
- Recursos computacionales limitados
- Datos secuenciales pequeños
- Aplicaciones en tiempo real (streaming)

**Transformers dominan en:**
- NLP (BERT, GPT, T5)
- Secuencias muy largas
- Tareas con datos masivos
- Estado del arte en benchmarks

**Híbridos y Nuevas Arquitecturas:**
- CNN + LSTM para video
- Transformer + LSTM para audio largo
- Mamba, RWKV (2023-2024): Alternativas eficientes a Transformers

## 13. Referencias y Recursos

### 13.1 Papers Fundamentales

1. **LSTM Original**: Hochreiter & Schmidhuber (1997)
   - "Long Short-Term Memory"

2. **GRU**: Cho et al. (2014)
   - "Learning Phrase Representations using RNN Encoder-Decoder"

3. **Seq2Seq**: Sutskever et al. (2014)
   - "Sequence to Sequence Learning with Neural Networks"

4. **Gradient Problems**: Bengio et al. (1994)
   - "Learning Long-Term Dependencies with Gradient Descent is Difficult"

### 13.2 Recursos Adicionales

- **Libros**: 
  - "Deep Learning" (Goodfellow, Bengio, Courville) - Capítulo 10
  - "Neural Network Methods for NLP" (Goldberg)

- **Tutoriales**:
  - Understanding LSTM Networks (colah.github.io)
  - The Unreasonable Effectiveness of RNNs (Karpathy)

- **Implementaciones**:
  - PyTorch: nn.LSTM, nn.GRU
  - TensorFlow/Keras: LSTM, GRU layers

### 13.3 Datasets para Práctica

- **Texto**: IMDB reviews, Penn Treebank, WikiText
- **Series Temporales**: Stock prices, weather data, sensor data
- **Audio**: LibriSpeech, Common Voice
- **Secuencias**: MNIST secuencial, Video datasets

---

## Conclusión

Las RNNs y especialmente los LSTMs representaron un avance crucial en el procesamiento de secuencias. Aunque los Transformers han ganado prominencia en NLP, RNN/LSTM siguen siendo herramientas valiosas, especialmente para:

1. Series temporales y predicción
2. Aplicaciones con recursos limitados
3. Problemas donde el procesamiento secuencial es natural
4. Situaciones con datos limitados

Comprender LSTM es fundamental para:
- Entender la evolución hacia Transformers
- Apreciar el problema de dependencias largas
- Diseñar arquitecturas híbridas
- Trabajar con datos secuenciales de manera efectiva

En el próximo laboratorio (Lab11), estudiaremos **Transformers**, que resuelven las limitaciones de paralelización de RNN/LSTM mediante mecanismos de atención.
