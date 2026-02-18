# Laboratorio de Redes Neuronales y Deep Learning

Conjunto de guías y prácticas de laboratorio en Python sobre redes neuronales, deep learning e inteligencia artificial generativa. Este curso está diseñado para iniciarse desde cero, con un enfoque muy didáctico basado en el libro "Neural Networks from Scratch in Python".

## 📚 Contenido del Curso

### [Lab 01: Introducción a las Neuronas](Lab01_Introduccion_Neuronas/)
Fundamentos de las redes neuronales. Implementación de una neurona desde cero.
- Teoría: Conceptos básicos de neuronas artificiales
- Práctica: Implementación de una neurona simple
- Código: Neurona con pesos y bias

### [Lab 02: Primera Red Neuronal](Lab02_Primera_Red_Neuronal/)
Construcción de la primera red neuronal completa desde cero.
- Teoría: Arquitectura de redes neuronales
- Práctica: Capas de neuronas
- Código: Red neuronal multicapa

### [Lab 03: Funciones de Activación](Lab03_Funciones_Activacion/)
Exploración de diferentes funciones de activación.
- Teoría: Propósito y tipos de funciones de activación
- Práctica: ReLU, Sigmoid, Softmax, Tanh
- Código: Implementación desde cero

### [Lab 04: Funciones de Pérdida y Optimización](Lab04_Funciones_Perdida/)
Medición del error y optimización de redes neuronales.
- Teoría: Funciones de costo y optimización
- Práctica: Cross-Entropy, MSE, MAE
- Código: Cálculo de pérdida

### [Lab 05: Backpropagation](Lab05_Backpropagation/)
Algoritmo de retropropagación para entrenar redes neuronales.
- Teoría: Derivadas y regla de la cadena
- Práctica: Cálculo de gradientes
- Código: Backpropagation desde cero

### [Lab 06: Entrenamiento de Redes Neuronales](Lab06_Entrenamiento/)
Proceso completo de entrenamiento de una red neuronal.
- Teoría: Descenso de gradiente, learning rate, epochs
- Práctica: Entrenamiento con datos reales
- Código: Loop de entrenamiento completo

### [Lab 07: Métricas de Evaluación y Matriz de Confusión](Lab07_Metricas_Evaluacion/)
Evaluación rigurosa de modelos de clasificación.
- Teoría: Matriz de confusión, Accuracy, Precision, Recall, F1-Score
- Práctica: Validación cruzada, datasets balanceados y desbalanceados
- Código: Implementación de métricas desde cero, optimización de umbrales

### [Lab 08: Frameworks de Deep Learning](Lab08_Frameworks_DeepLearning/)
Introducción a PyTorch y TensorFlow.
- Teoría: Ventajas de usar frameworks
- Práctica: Comparación de implementaciones
- Código: Redes neuronales con PyTorch y TensorFlow

### [Lab 09: Inteligencia Artificial Generativa](Lab09_IA_Generativa/)
Introducción a modelos generativos modernos.
- Teoría: VAE, GAN, Diffusion Models, aplicaciones con Transformers
- Práctica: Tipos de modelos generativos, generación de contenido
- Código: Modelo generativo simple, integración con arquitecturas modernas

### [Lab 10: Redes Neuronales Convolucionales (CNN)](Lab10_Redes_Neuronales_Convolucionales/)
Arquitecturas especializadas para procesamiento de imágenes y visión computacional.
- Teoría: Convolución, pooling, arquitecturas famosas (LeNet, ResNet, VGG)
- Práctica: Implementación de CNN desde cero, clasificación de imágenes
- Código: Capas convolucionales, filtros, CNN completa en PyTorch

### [Lab 11: Redes Neuronales Recurrentes y LSTM](Lab11_Redes_Neuronales_Recurrentes_LSTM/)
Arquitecturas para datos secuenciales como texto y series de tiempo.
- Teoría: RNN, LSTM, GRU, problema del gradiente que desaparece
- Práctica: Procesamiento de secuencias, predicción de series temporales
- Código: RNN y LSTM desde cero, clasificación de texto, generación

### [Lab 12: Transformers](Lab12_Transformers/)
Arquitectura revolucionaria basada en atención para NLP y más.
- Teoría: Self-Attention, Multi-Head Attention, BERT, GPT, Vision Transformers
- Práctica: Implementación de Transformers, fine-tuning de modelos
- Código: Attention desde cero, Hugging Face, aplicaciones modernas

## 🚀 Cómo Empezar

### Requisitos Previos
- Python 3.8 o superior
- Conocimientos básicos de programación en Python
- Conocimientos básicos de matemáticas (álgebra lineal, cálculo)

### Instalación

1. Clonar el repositorio:
```bash
git clone https://github.com/eastaizah/Lab_NN.git
cd Lab_NN
```

2. Crear un entorno virtual (recomendado):
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. Instalar dependencias:
```bash
pip install -r requirements.txt
```

### Uso

Cada laboratorio contiene:
- `teoria.md`: Documento con fundamentos teóricos
- `practica.ipynb`: Jupyter notebook con ejercicios prácticos
- `codigo/`: Directorio con implementaciones de ejemplo

Se recomienda seguir los laboratorios en orden, ya que cada uno construye sobre los conceptos del anterior.

## 📖 Metodología

Este curso sigue una filosofía didáctica basada en "Neural Networks from Scratch in Python" y expandida a arquitecturas modernas:
1. **Entender los fundamentos**: Implementar todo desde cero antes de usar librerías
2. **Aprendizaje práctico**: Código ejecutable en cada laboratorio
3. **Progresión gradual**: De conceptos simples a arquitecturas complejas
4. **Visualización**: Gráficos y ejemplos visuales en cada tema
5. **Del fundamento a la práctica**: Desde implementaciones NumPy hasta modelos de producción

## 📋 Ruta de Aprendizaje

El curso está organizado en **tres módulos pedagógicos**:

### Módulo 1: Fundamentos (Labs 01-07)
Construcción de redes neuronales desde cero con NumPy
- Neuronas y arquitecturas básicas
- Funciones de activación y pérdida
- Backpropagation y optimización
- Entrenamiento completo
- **Evaluación y métricas de clasificación**

### Módulo 2: Frameworks y Arquitecturas Modernas (Labs 08, 10-12)
Arquitecturas especializadas y herramientas profesionales
- PyTorch y TensorFlow
- **CNNs** para visión computacional
- **RNNs/LSTMs** para secuencias y texto
- **Transformers** para NLP y aplicaciones multimodales

### Módulo 3: IA Generativa (Lab 09)
Modelos generativos modernos
- VAEs y GANs
- Diffusion Models
- Integración con Transformers (GPT, DALL-E)

## 🤝 Contribuir

Las contribuciones son bienvenidas. Por favor, abre un issue o pull request para sugerencias o mejoras.

## 📄 Licencia

Este proyecto es de código abierto y está disponible bajo la licencia MIT.

## 📚 Referencias

- Harrison Kinsley & Daniel Kukieła. "Neural Networks from Scratch in Python"
- Ian Goodfellow, Yoshua Bengio, Aaron Courville. "Deep Learning"
- Michael Nielsen. "Neural Networks and Deep Learning"
