# Guía Completa del Curso de Redes Neuronales

## 📚 Descripción General

Este repositorio contiene un curso completo de Redes Neuronales, Deep Learning e Inteligencia Artificial Generativa, diseñado para aprender desde cero con un enfoque muy didáctico, basado en el libro "Neural Networks from Scratch in Python".

## 🎯 Objetivos del Curso

Al completar este curso, serás capaz de:

1. ✅ Comprender los fundamentos matemáticos de las redes neuronales
2. ✅ Implementar redes neuronales completamente desde cero en Python
3. ✅ Entrenar modelos para problemas reales de clasificación y regresión
4. ✅ Dominar arquitecturas especializadas: CNNs, RNNs/LSTMs y Transformers
5. ✅ Procesar imágenes con Redes Neuronales Convolucionales
6. ✅ Trabajar con datos secuenciales usando RNNs y LSTMs
7. ✅ Entender y aplicar mecanismos de atención y Transformers
8. ✅ Usar frameworks modernos como PyTorch y TensorFlow
9. ✅ Crear modelos de IA Generativa (VAE, GAN)
10. ✅ Aplicar buenas prácticas en el desarrollo de modelos de ML

## 📋 Estructura del Curso

### Módulo 1: Fundamentos (Labs 01-02)

#### [Lab 01: Introducción a las Neuronas](Lab01_Introduccion_Neuronas/)
**Duración estimada**: 2-3 horas

**Aprenderás**:
- Qué es una neurona artificial
- Pesos, bias y producto punto
- Implementación desde cero con NumPy
- Procesamiento en batch

**Archivos**:
- `teoria.md`: Fundamentos teóricos completos
- `practica.ipynb`: Ejercicios interactivos
- `codigo/neurona.py`: Implementación completa con ejemplos

**Conceptos clave**: Neurona, Pesos, Bias, Forward Pass, NumPy

---

#### [Lab 02: Primera Red Neuronal](Lab02_Primera_Red_Neuronal/)
**Duración estimada**: 3-4 horas

**Aprenderás**:
- Arquitectura de redes neuronales multicapa
- Conectar capas de neuronas
- Forward propagation
- Diseño de arquitecturas

**Archivos**:
- `teoria.md`: Arquitecturas y dimensiones
- `practica.ipynb`: Construcción de redes
- `codigo/red_neuronal.py`: Red neuronal completa

**Conceptos clave**: Capas, Arquitectura, Forward Propagation, Parámetros

---

### Módulo 2: Componentes Esenciales (Labs 03-04)

#### [Lab 03: Funciones de Activación](Lab03_Funciones_Activacion/)
**Duración estimada**: 3-4 horas

**Aprenderás**:
- ReLU, Sigmoid, Tanh, Softmax
- Por qué necesitamos no-linealidad
- Derivadas de funciones de activación
- Cuándo usar cada función

**Archivos**:
- `teoria.md`: Matemáticas y casos de uso
- `practica.ipynb`: Comparación visual
- `codigo/activaciones.py`: Todas las funciones implementadas

**Conceptos clave**: No-linealidad, ReLU, Sigmoid, Softmax, Gradientes

---

#### [Lab 04: Funciones de Pérdida](Lab04_Funciones_Perdida/)
**Duración estimada**: 3-4 horas

**Aprenderás**:
- MSE, MAE, Cross-Entropy
- Cómo medir el error de una red
- Descenso de gradiente básico
- Optimización

**Archivos**:
- `teoria.md`: Funciones de pérdida explicadas
- `practica.ipynb`: Comparación de loss functions
- `codigo/perdida.py`: Implementaciones completas

**Conceptos clave**: Loss Function, MSE, Cross-Entropy, Gradient Descent

---

### Módulo 3: Entrenamiento (Labs 05-06)

#### [Lab 05: Backpropagation](Lab05_Backpropagation/)
**Duración estimada**: 4-5 horas

**Aprenderás**:
- Regla de la cadena
- Grafos computacionales
- Algoritmo de backpropagation completo
- Cálculo de gradientes

**Archivos**:
- `teoria.md`: Matemáticas del backprop
- `practica.ipynb`: Implementación paso a paso
- `codigo/backprop.py`: Backprop completo

**Conceptos clave**: Chain Rule, Gradientes, Backward Pass, Derivadas

---

#### [Lab 06: Entrenamiento de Redes](Lab06_Entrenamiento/)
**Duración estimada**: 4-5 horas

**Aprenderás**:
- Loop de entrenamiento completo
- Epochs, batches, learning rate
- Validación y overfitting
- Entrenar en datos reales

**Archivos**:
- `teoria.md`: Proceso de entrenamiento
- `practica.ipynb`: Entrenamiento real
- `codigo/entrenamiento.py`: Sistema completo

**Conceptos clave**: Training Loop, Epochs, Batches, Validation, Overfitting

---

### Módulo 4: Evaluación y Métricas (Lab 07)

#### [Lab 07: Métricas de Evaluación y Matriz de Confusión](Lab07_Metricas_Evaluacion/)
**Duración estimada**: 3-4 horas

**Aprenderás**:
- Matriz de confusión y sus componentes
- Métricas: Accuracy, Precision, Recall, F1-Score
- Validación cruzada (K-Fold)
- Evaluación en datasets balanceados y desbalanceados
- Optimización de umbrales de clasificación

**Archivos**:
- `teoria.md`: Fundamentos de evaluación de modelos
- `practica.ipynb`: Ejercicios con datasets reales
- `codigo/metricas.py`: Implementación de métricas desde cero

**Conceptos clave**: Matriz de confusión, TP/FP/FN/TN, Precision, Recall, F1-Score, Cross-Validation

---

### Módulo 5: Frameworks y Herramientas (Lab 08)

#### [Lab 08: Frameworks de Deep Learning](Lab08_Frameworks_DeepLearning/)
**Duración estimada**: 3-4 horas

**Aprenderás**:
- PyTorch básico
- TensorFlow/Keras básico
- Comparación de frameworks
- Migrar de código manual a frameworks

**Archivos**:
- `teoria.md`: Comparación PyTorch vs TensorFlow
- `practica.ipynb`: Mismo modelo en ambos frameworks
- `codigo/pytorch_ejemplo.py`: Ejemplo completo PyTorch
- `codigo/tensorflow_ejemplo.py`: Ejemplo completo TensorFlow

**Conceptos clave**: PyTorch, TensorFlow, High-level APIs, Autograd

---

### Módulo 6: Arquitecturas Especializadas (Labs 10-12)

#### [Lab 10: Redes Neuronales Convolucionales (CNN)](Lab10_Redes_Neuronales_Convolucionales/)
**Duración estimada**: 4-5 horas

**Aprenderás**:
- Operación de convolución y correlación cruzada
- Arquitectura de CNNs: capas convolucionales, pooling, fully connected
- Filtros y feature maps
- Aplicaciones en visión por computadora
- Implementación desde cero y con PyTorch/TensorFlow

**Archivos**:
- `teoria.md`: Matemáticas de convolución, arquitecturas CNN clásicas
- `practica.ipynb`: Construcción de CNN para clasificación de imágenes
- `codigo/cnn.py`: Implementación completa de CNN
- `codigo/cnn_pytorch.py`: CNN usando PyTorch
- `codigo/cnn_tensorflow.py`: CNN usando TensorFlow/Keras

**Conceptos clave**: Convolución, Filtros, Feature Maps, Pooling, Stride, Padding, VGG, ResNet

---

#### [Lab 11: Redes Neuronales Recurrentes y LSTM](Lab11_Redes_Neuronales_Recurrentes_LSTM/)
**Duración estimada**: 5-6 horas

**Aprenderás**:
- Arquitectura de RNNs para datos secuenciales
- Backpropagation Through Time (BPTT)
- Problema del vanishing gradient en RNNs
- LSTMs: puertas de olvido, entrada y salida
- GRU como alternativa simplificada
- Aplicaciones en procesamiento de texto y series temporales

**Archivos**:
- `teoria.md`: RNNs, LSTMs, GRUs y sus matemáticas
- `practica.ipynb`: Predicción de series temporales y generación de texto
- `codigo/rnn.py`: Implementación RNN desde cero
- `codigo/lstm.py`: Implementación LSTM completa
- `codigo/lstm_pytorch.py`: LSTM usando PyTorch
- `codigo/lstm_tensorflow.py`: LSTM usando TensorFlow/Keras

**Conceptos clave**: RNN, LSTM, GRU, Secuencias, Estado Oculto, Gates, BPTT, Vanishing Gradient

---

#### [Lab 12: Transformers y Mecanismos de Atención](Lab12_Transformers/)
**Duración estimada**: 6-7 horas

**Aprenderás**:
- Mecanismo de self-attention
- Queries, Keys y Values (Q, K, V)
- Multi-head attention
- Positional encoding
- Arquitectura completa del Transformer
- Diferencias entre modelos encoder, decoder y encoder-decoder
- Aplicaciones modernas: BERT, GPT, Vision Transformers

**Archivos**:
- `teoria.md`: Arquitectura Transformer, atención y positional encoding
- `practica.ipynb`: Construcción de Transformer paso a paso
- `codigo/attention.py`: Implementación de mecanismos de atención
- `codigo/transformer.py`: Transformer completo desde cero
- `codigo/transformer_pytorch.py`: Transformer usando PyTorch
- `codigo/transformer_tensorflow.py`: Transformer usando TensorFlow/Keras

**Conceptos clave**: Self-Attention, Multi-Head Attention, Q-K-V, Positional Encoding, Transformer, BERT, GPT, Encoder-Decoder

---

### Módulo 7: IA Generativa (Lab 09)

#### [Lab 09: Inteligencia Artificial Generativa](Lab09_IA_Generativa/)
**Duración estimada**: 4-5 horas

**Aprenderás**:
- Conceptos de IA Generativa
- VAE (Variational Autoencoders) básicos
- GAN (Generative Adversarial Networks) básicos
- Aplicaciones de modelos generativos

**Archivos**:
- `teoria.md`: Fundamentos de IA Generativa
- `practica.ipynb`: Modelos generativos simples
- `codigo/generativo.py`: VAE y GAN básicos

**Conceptos clave**: Generative AI, VAE, GAN, Latent Space, Generation

---

## 🚀 Cómo Empezar

### Requisitos Previos

**Conocimientos**:
- Python básico (variables, funciones, clases)
- Matemáticas básicas (álgebra, cálculo básico)
- Opcional: NumPy básico

**Software**:
- Python 3.8 o superior
- pip (gestor de paquetes)
- Jupyter Notebook
- Editor de código (VS Code, PyCharm, etc.)

### Instalación

1. **Clonar el repositorio**:
```bash
git clone https://github.com/eastaizah/Lab_NN.git
cd Lab_NN
```

2. **Crear entorno virtual** (recomendado):
```bash
# Linux/Mac
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

3. **Instalar dependencias**:
```bash
pip install -r requirements.txt
```

4. **Verificar instalación**:
```bash
python -c "import numpy, matplotlib, torch; print('✓ Todo instalado correctamente')"
```

### Ejecutar los Laboratorios

**Opción 1: Jupyter Notebooks** (Recomendado para aprender)
```bash
jupyter notebook
# Navega a cada laboratorio y abre practica.ipynb
```

**Opción 2: Scripts Python** (Para ver ejemplos completos)
```bash
# Ejecutar ejemplo de Lab 01
python Lab01_Introduccion_Neuronas/codigo/neurona.py

# Ejecutar ejemplo de Lab 02
python Lab02_Primera_Red_Neuronal/codigo/red_neuronal.py

# Y así sucesivamente...
```

## 📖 Metodología de Aprendizaje

### Para cada laboratorio:

1. **Leer la teoría** (30-40 min)
   - Abre `teoria.md`
   - Lee cuidadosamente los conceptos
   - Toma notas de dudas

2. **Practicar con el notebook** (60-90 min)
   - Abre `practica.ipynb` en Jupyter
   - Ejecuta cada celda
   - Experimenta modificando valores
   - Completa los ejercicios

3. **Revisar el código completo** (20-30 min)
   - Abre los archivos en `codigo/`
   - Estudia las implementaciones
   - Compara con tus ejercicios

4. **Experimentar y profundizar** (30-60 min)
   - Modifica parámetros
   - Prueba diferentes arquitecturas
   - Resuelve los desafíos

5. **Reflexionar** (10-15 min)
   - Responde las preguntas de reflexión
   - Anota conceptos clave
   - Identifica áreas para revisar

## 📊 Progreso Recomendado

### Semana 1: Fundamentos
- **Día 1-2**: Lab 01 - Neuronas
- **Día 3-4**: Lab 02 - Redes Neuronales
- **Día 5**: Revisión y práctica adicional

### Semana 2: Componentes
- **Día 1-2**: Lab 03 - Funciones de Activación
- **Día 3-4**: Lab 04 - Funciones de Pérdida
- **Día 5**: Proyecto integrador 1

### Semana 3: Entrenamiento y Evaluación
- **Día 1-3**: Lab 05 - Backpropagation
- **Día 4-5**: Lab 06 - Entrenamiento

### Semana 4: Métricas y Frameworks
- **Día 1-2**: Lab 07 - Métricas y Evaluación
- **Día 3-5**: Lab 08 - PyTorch/TensorFlow

### Semana 5: Visión por Computadora
- **Día 1-3**: Lab 10 - CNNs
- **Día 4-5**: Proyectos con imágenes

### Semana 6: Procesamiento Secuencial
- **Día 1-4**: Lab 11 - RNNs y LSTMs
- **Día 5**: Proyectos con series temporales/texto

### Semana 7: Arquitecturas Modernas
- **Día 1-5**: Lab 12 - Transformers y Atención

### Semana 8: IA Generativa y Proyecto Final
- **Día 1-3**: Lab 09 - IA Generativa
- **Día 4-5**: Proyecto final integrador

## 🛤️ Camino de Aprendizaje

### Progresión Pedagógica

El curso sigue una progresión cuidadosamente diseñada:

**Fase 1: Fundamentos (Labs 01-02)**
```
Neurona individual → Capas de neuronas → Redes neuronales densas
```

**Fase 2: Componentes Core (Labs 03-04)**
```
Funciones de activación → Funciones de pérdida → Optimización básica
```

**Fase 3: Mecanismos de Aprendizaje (Labs 05-07)**
```
Backpropagation → Entrenamiento completo → Evaluación y métricas
```

**Fase 4: Herramientas Profesionales (Lab 08)**
```
Código manual → PyTorch/TensorFlow → Desarrollo profesional
```

**Fase 5: Arquitecturas Especializadas (Labs 10-12)**
```
Visión (CNNs) → Secuencias (RNNs/LSTMs) → Atención (Transformers)
```

**Fase 6: Generación (Lab 09)**
```
Modelos discriminativos → Modelos generativos → VAE y GAN
```

### ¿Por qué este orden?

1. **Labs 01-06**: Base sólida antes de especializaciones
2. **Lab 07**: Evaluación y métricas - esencial antes de frameworks
3. **Lab 08**: Frameworks antes de arquitecturas complejas
4. **Lab 10 (CNNs)**: Más intuitivo, introduce convolución
5. **Lab 11 (RNNs/LSTMs)**: Secuencias y memoria
6. **Lab 12 (Transformers)**: Combina conceptos de CNNs y RNNs
7. **Lab 09 (IA Generativa)**: Culminación, usa todas las técnicas anteriores

## 🎓 Evaluación y Proyectos

### Proyectos Sugeridos

**Proyecto 1** (Después de Lab 02):
- Crear una red para clasificar flores Iris
- Implementar desde cero sin frameworks

**Proyecto 2** (Después de Lab 04):
- Red para reconocer dígitos MNIST
- Incluir funciones de activación y pérdida

**Proyecto 3** (Después de Lab 06):
- Sistema de clasificación completo
- Con entrenamiento, validación y evaluación

**Proyecto 4** (Después de Lab 07):
- Reimplementar proyectos anteriores usando PyTorch o TensorFlow
- Comparar rendimiento y facilidad de uso

**Proyecto 5** (Después de Lab 09):
- Clasificador de imágenes con CNN
- Usar CIFAR-10 o ImageNet subset
- Experimentar con data augmentation

**Proyecto 6** (Después de Lab 10):
- Predictor de series temporales (precio de acciones, clima)
- O generador de texto con LSTM
- Analizar análisis de sentimiento

**Proyecto 7** (Después de Lab 11):
- Implementar mini-GPT o mini-BERT
- Tarea de NLP: clasificación, QA o generación
- Explorar fine-tuning de modelos pre-entrenados

**Proyecto Final** (Después de Lab 08):
- Modelo generativo para crear imágenes (GAN)
- O sistema de text-to-image simplificado
- O chatbot usando Transformers
- Integrar múltiples conceptos del curso

## 📚 Recursos Adicionales

### Libros
- **"Neural Networks from Scratch in Python"** - Harrison Kinsley & Daniel Kukieła
- **"Deep Learning"** - Ian Goodfellow, Yoshua Bengio, Aaron Courville
- **"Neural Networks and Deep Learning"** - Michael Nielsen
- **"Dive into Deep Learning"** - Aston Zhang, Zachary C. Lipton, Mu Li, Alexander J. Smola
- **"Attention Is All You Need"** - Paper original de Transformers (Vaswani et al., 2017)

### Cursos Online
- [3Blue1Brown - Neural Networks](https://www.youtube.com/watch?v=aircAruvnKk)
- [Fast.ai - Practical Deep Learning](https://www.fast.ai/)
- [Stanford CS231n - CNNs for Visual Recognition](http://cs231n.stanford.edu/)
- [Stanford CS224n - NLP with Deep Learning](http://web.stanford.edu/class/cs224n/)
- [DeepLearning.AI - Coursera](https://www.coursera.org/specializations/deep-learning)

### Herramientas Interactivas
- [TensorFlow Playground](http://playground.tensorflow.org/)
- [CNN Explainer](https://poloclub.github.io/cnn-explainer/)
- [Distill.pub](https://distill.pub/)
- [Transformer Explainer](https://poloclub.github.io/transformer-explainer/)
- [LSTMVis](http://lstm.seas.harvard.edu/)

### Papers Fundamentales
- **AlexNet** (2012): ImageNet Classification with Deep CNNs
- **VGGNet** (2014): Very Deep CNNs
- **ResNet** (2015): Deep Residual Learning
- **LSTM** (1997): Long Short-Term Memory
- **Attention** (2014): Neural Machine Translation by Jointly Learning to Align and Translate
- **Transformer** (2017): Attention Is All You Need
- **BERT** (2018): Pre-training of Deep Bidirectional Transformers
- **GPT** series (2018-2023): Language Models are Unsupervised Multitask Learners
- **Vision Transformer** (2020): An Image is Worth 16x16 Words

## ❓ FAQ (Preguntas Frecuentes)

**P: ¿Necesito saber matemáticas avanzadas?**
R: No. El curso explica los conceptos matemáticos necesarios. Álgebra y cálculo básico son suficientes.

**P: ¿Cuánto tiempo toma completar el curso?**
R: Aproximadamente 6-8 semanas dedicando 2-3 horas diarias para el curso completo (11 labs). Puedes ir a tu propio ritmo. El curso básico (Labs 01-07) toma 4-5 semanas.

**P: ¿Puedo saltar laboratorios?**
R: No recomendado. Cada lab construye sobre los anteriores. El orden es importante.

**P: ¿Qué hago si me atasco?**
R: 
1. Revisa la teoría nuevamente
2. Estudia el código de ejemplo
3. Busca en los recursos adicionales
4. Abre un issue en GitHub

**P: ¿Necesito una GPU?**
R: No para Labs 01-07. Labs 09-11 funcionan en CPU pero GPU acelera significativamente. Lab 08 (GANs) se beneficia de GPU. Google Colab ofrece GPUs gratuitas.

**P: ¿Cuál es la diferencia entre CNNs, RNNs y Transformers?**
R: CNNs son ideales para datos espaciales (imágenes). RNNs/LSTMs procesan secuencias (texto, series temporales). Transformers usan atención, son más rápidos y potentes que RNNs para secuencias largas.

**P: ¿Debo aprender todos los labs en orden?**
R: Sí para Labs 01-07 (fundamentos). Labs 09-11 se pueden hacer en orden diferente si ya dominas los fundamentos, pero el orden recomendado es pedagógicamente óptimo.

## 🤝 Contribuir

¿Encontraste un error? ¿Tienes una sugerencia?
1. Abre un issue describiendo el problema/sugerencia
2. O envía un pull request con la mejora

## 📄 Licencia

Este proyecto está bajo la licencia MIT. Ver archivo LICENSE para más detalles.

## 🙏 Agradecimientos

Inspirado en:
- "Neural Networks from Scratch in Python" por Harrison Kinsley y Daniel Kukieła
- La comunidad de deep learning y open source
- Todos los recursos educativos mencionados

---

## 📞 Contacto

Para preguntas, sugerencias o colaboraciones:
- GitHub Issues: [Lab_NN Issues](https://github.com/eastaizah/Lab_NN/issues)
- Discusiones: [Lab_NN Discussions](https://github.com/eastaizah/Lab_NN/discussions)

---

**¡Feliz aprendizaje! 🚀🧠**

*Última actualización: Diciembre 2024*
