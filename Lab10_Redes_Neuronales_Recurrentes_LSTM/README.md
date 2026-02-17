# Lab 10: Redes Neuronales Recurrentes y LSTM

## Descripción

Este laboratorio introduce las Redes Neuronales Recurrentes (RNN) y Long Short-Term Memory (LSTM), arquitecturas especializadas para procesar datos secuenciales como texto, series de tiempo y audio. Implementaremos desde cero los componentes fundamentales y exploraremos aplicaciones prácticas.

## Objetivos de Aprendizaje

Al completar este laboratorio, podrás:

1. ✅ Comprender cómo las RNNs procesan secuencias
2. ✅ Implementar una RNN simple desde cero
3. ✅ Entender el problema del gradiente que desaparece/explota
4. ✅ Comprender la arquitectura LSTM y sus componentes (gates)
5. ✅ Implementar LSTM desde cero
6. ✅ Aplicar RNNs/LSTMs a problemas de clasificación de texto
7. ✅ Entender bidirectional RNNs y stacked RNNs
8. ✅ Conocer variantes: GRU, Bidirectional LSTM

## Contenido

### 📖 Teoría (`teoria.md`)

Documento completo con los fundamentos teóricos:
- ¿Por qué RNNs para secuencias?
- Arquitectura de RNN vanilla
- Backpropagation Through Time (BPTT)
- Problema del gradiente que desaparece
- LSTM: arquitectura y gates
- GRU: alternativa más simple
- Aplicaciones de RNNs

### 💻 Práctica (`practica.ipynb`)

Jupyter Notebook interactivo con:
- Implementación de RNN desde cero
- Construcción de LSTM paso a paso
- Entrenamiento en datos secuenciales
- Predicción de series de tiempo
- Clasificación de sentimientos en texto
- Generación de texto
- Ejercicios progresivos

### 🔧 Código de Ejemplo (`codigo/rnn_lstm.py`)

Script Python con implementaciones completas:
- Clase `RNNCell`: Celda RNN básica
- Clase `RNN`: Red recurrente completa
- Clase `LSTMCell`: Celda LSTM con gates
- Clase `LSTM`: LSTM completa
- Clase `GRU`: Gated Recurrent Unit
- Ejemplos con PyTorch

## Cómo Usar Este Laboratorio

### Opción 1: Jupyter Notebook (Recomendado)

```bash
# Desde el directorio del repositorio
cd Lab10_Redes_Neuronales_Recurrentes_LSTM
jupyter notebook practica.ipynb
```

### Opción 2: Script Python

```bash
# Ejecutar el código de ejemplo
python codigo/rnn_lstm.py
```

### Opción 3: Lectura y Experimentación

1. Lee `teoria.md` para entender los conceptos
2. Abre `practica.ipynb` en Jupyter
3. Ejecuta cada celda y experimenta con las secuencias
4. Completa los ejercicios propuestos
5. Revisa `codigo/rnn_lstm.py` como referencia

## Requisitos

```bash
pip install numpy matplotlib jupyter torch
```

## Conceptos Clave

- **Secuencia**: Datos ordenados en el tiempo (texto, series temporales)
- **Estado Oculto (Hidden State)**: Memoria de la RNN sobre el pasado
- **Celda Recurrente**: Unidad que procesa un paso de tiempo
- **BPTT**: Backpropagation Through Time - entrenamiento de RNNs
- **Gates**: Mecanismos que controlan flujo de información (forget, input, output)
- **Cell State**: Memoria a largo plazo en LSTM
- **GRU**: Versión simplificada de LSTM con menos parámetros

## Ejercicios

### Ejercicio 10.1: RNN para Suma
Implementa una RNN que sume una secuencia de números.

### Ejercicio 10.2: LSTM desde Cero
Completa la implementación de LSTM con todos los gates.

### Ejercicio 10.3: Predicción de Series
Entrena LSTM para predecir valores futuros de una serie de tiempo.

### Ejercicio 10.4: Clasificación de Sentimientos
Usa RNN/LSTM para clasificar reseñas como positivas o negativas.

### Ejercicio 10.5: Generación de Texto (Desafío)
Genera texto caracter por caracter usando LSTM.

## Ventajas de RNNs/LSTMs

1. **Procesan Secuencias de Longitud Variable**: Flexible para diferentes inputs
2. **Memoria Temporal**: Mantienen contexto del pasado
3. **Compartición de Parámetros**: Mismos pesos en cada paso temporal
4. **Capturan Dependencias Temporales**: Entienden orden y contexto

## Diferencias RNN vs LSTM

### RNN Vanilla

**Ventajas:**
- Simple, fácil de entender
- Menos parámetros

**Desventajas:**
- Gradientes que desaparecen/explotan
- No captura dependencias largas
- Difícil de entrenar

### LSTM

**Ventajas:**
- Captura dependencias a largo plazo
- Resuelve gradientes que desaparecen
- Más estable en entrenamiento

**Desventajas:**
- Más parámetros (4× que RNN)
- Más lento de entrenar
- Más complejo

### GRU

**Ventajas:**
- Menos parámetros que LSTM (3× que RNN)
- Más rápido que LSTM
- Rendimiento similar a LSTM

**Desventajas:**
- Más complejo que RNN vanilla

## Aplicaciones

### Procesamiento de Lenguaje Natural (NLP)
- **Clasificación de Texto**: Sentimientos, spam, categorías
- **Traducción Automática**: seq2seq con encoder-decoder
- **Generación de Texto**: Completar oraciones, escribir historias
- **Named Entity Recognition**: Identificar personas, lugares
- **Question Answering**: Responder preguntas sobre texto

### Series de Tiempo
- **Predicción de Stock**: Precios de acciones
- **Predicción de Clima**: Temperatura, lluvia
- **Predicción de Demanda**: Ventas, tráfico
- **Detección de Anomalías**: En señales temporales

### Audio y Música
- **Reconocimiento de Voz**: Speech-to-text
- **Generación de Música**: Componer melodías
- **Clasificación de Audio**: Géneros musicales

### Video
- **Descripción de Video**: Generar captions
- **Reconocimiento de Acciones**: Detectar actividades

## Arquitecturas Avanzadas

### Bidirectional RNN/LSTM
```
→ → → →  (forward)
←序列数据 ←  (backward)
```
- Procesa secuencia en ambas direcciones
- Mejor contexto para cada elemento
- Útil cuando toda la secuencia está disponible

### Stacked (Deep) RNN/LSTM
```
LSTM Layer 3
    ↑
LSTM Layer 2
    ↑
LSTM Layer 1
    ↑
  Input
```
- Múltiples capas RNN/LSTM apiladas
- Captura jerarquía de características
- Primera capa: características simples
- Capas superiores: características complejas

### Encoder-Decoder
```
Encoder RNN → Context Vector → Decoder RNN
   input           |              output
 sequence          |            sequence
```
- Usado en traducción automática
- Encoder comprime input a vector
- Decoder genera output desde vector

### Attention Mechanism
- Permite enfocarse en partes relevantes del input
- Mejora significativa sobre encoder-decoder básico
- Base para Transformers (Lab 11)

## Notas Importantes

⚠️ **Gradient Clipping**: Esencial para evitar explosión de gradientes. Limita norma de gradientes (ej: clip a 5.0).

💡 **Secuencia de Tamaño**: LSTMs funcionan bien hasta ~200-300 pasos. Para más largo, considera Transformers.

🚀 **Embeddings**: Para texto, usa embeddings (Word2Vec, GloVe) antes de RNN/LSTM.

⚡ **Bidireccional**: Útil para tareas donde el futuro importa (clasificación), no para predicción en tiempo real.

## Fórmulas Importantes

### RNN Vanilla
```python
h_t = tanh(W_hh @ h_{t-1} + W_xh @ x_t + b_h)
y_t = W_hy @ h_t + b_y
```

### LSTM
```python
f_t = σ(W_f @ [h_{t-1}, x_t] + b_f)  # forget gate
i_t = σ(W_i @ [h_{t-1}, x_t] + b_i)  # input gate
C̃_t = tanh(W_C @ [h_{t-1}, x_t] + b_C)  # candidate
C_t = f_t * C_{t-1} + i_t * C̃_t  # cell state
o_t = σ(W_o @ [h_{t-1}, x_t] + b_o)  # output gate
h_t = o_t * tanh(C_t)  # hidden state
```

### GRU
```python
r_t = σ(W_r @ [h_{t-1}, x_t])  # reset gate
z_t = σ(W_z @ [h_{t-1}, x_t])  # update gate
h̃_t = tanh(W @ [r_t * h_{t-1}, x_t])  # candidate
h_t = (1 - z_t) * h_{t-1} + z_t * h̃_t  # hidden state
```

## Número de Parámetros

Para hidden_size=h, input_size=x, output_size=y:

- **RNN**: 4 matrices → (h×h + x×h + h + h×y + y)
- **LSTM**: 4× RNN → 4(h×h + x×h + h) + (h×y + y)
- **GRU**: 3× RNN → 3(h×h + x×h + h) + (h×y + y)

## Próximo Paso

Una vez completes este laboratorio, continúa con:

👉 **[Lab 11: Transformers](../Lab11_Transformers/)**

Exploraremos la arquitectura que revolucionó el NLP y está transformando todo el deep learning.

## Recursos Adicionales

- [Understanding LSTM Networks - colah's blog](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [The Unreasonable Effectiveness of RNNs - Andrej Karpathy](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
- [PyTorch RNN Tutorial](https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html)
- [Sequence Models - Coursera](https://www.coursera.org/learn/nlp-sequence-models)
- [RNN Cheatsheet - Stanford CS230](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-recurrent-neural-networks)

## Preguntas Frecuentes

**P: ¿Cuándo usar RNN vs LSTM vs GRU?**  
R: RNN para secuencias cortas y simples. LSTM cuando necesitas memoria a largo plazo. GRU como alternativa más rápida a LSTM con rendimiento similar.

**P: ¿Por qué los gradientes desaparecen en RNNs?**  
R: Al hacer backprop a través de muchos pasos temporales, multiplicamos derivadas <1, haciendo que el gradiente → 0 exponencialmente.

**P: ¿LSTM siempre es mejor que RNN?**  
R: No siempre. Para tareas simples, RNN puede ser suficiente y más rápido. LSTM brilla en dependencias largas.

**P: ¿Bidirectional LSTM para generación de texto?**  
R: No. Bidirectional requiere toda la secuencia. Para generación (predicción del futuro), usa LSTM unidireccional.

**P: ¿Cuántas capas usar en Stacked LSTM?**  
R: Típicamente 2-3 capas. Más de 4 raramente ayuda y aumenta overfitting.

## Verificación de Conocimientos

- [ ] Entiendo cómo las RNNs procesan secuencias paso a paso
- [ ] Puedo explicar el problema del gradiente que desaparece
- [ ] Entiendo los 3 gates de LSTM y su propósito
- [ ] Sé implementar RNN y LSTM desde cero
- [ ] Conozco la diferencia entre RNN, LSTM y GRU
- [ ] Puedo aplicar RNNs/LSTMs a problemas de NLP y series de tiempo
- [ ] Entiendo cuándo usar bidirectional vs unidirectional
