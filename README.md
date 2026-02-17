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

### [Lab 07: Frameworks de Deep Learning](Lab07_Frameworks_DeepLearning/)
Introducción a PyTorch y TensorFlow.
- Teoría: Ventajas de usar frameworks
- Práctica: Comparación de implementaciones
- Código: Redes neuronales con PyTorch y TensorFlow

### [Lab 08: Inteligencia Artificial Generativa](Lab08_IA_Generativa/)
Introducción a modelos generativos.
- Teoría: Conceptos de IA generativa
- Práctica: Tipos de modelos generativos
- Código: Modelo generativo simple

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

Este curso sigue la filosofía del libro "Neural Networks from Scratch in Python":
1. **Entender los fundamentos**: Implementar todo desde cero antes de usar librerías
2. **Aprendizaje práctico**: Código ejecutable en cada laboratorio
3. **Progresión gradual**: De conceptos simples a complejos
4. **Visualización**: Gráficos y ejemplos visuales en cada tema

## 🤝 Contribuir

Las contribuciones son bienvenidas. Por favor, abre un issue o pull request para sugerencias o mejoras.

## 📄 Licencia

Este proyecto es de código abierto y está disponible bajo la licencia MIT.

## 📚 Referencias

- Harrison Kinsley & Daniel Kukieła. "Neural Networks from Scratch in Python"
- Ian Goodfellow, Yoshua Bengio, Aaron Courville. "Deep Learning"
- Michael Nielsen. "Neural Networks and Deep Learning"
