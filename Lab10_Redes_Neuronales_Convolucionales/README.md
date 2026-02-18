# Lab 10: Redes Neuronales Convolucionales (CNN)

## Descripción

Este laboratorio introduce las Redes Neuronales Convolucionales (CNN), arquitecturas especializadas para procesamiento de datos con estructura de cuadrícula como imágenes. Implementaremos desde cero los componentes fundamentales de una CNN y exploraremos sus aplicaciones en visión computacional.

## Objetivos de Aprendizaje

Al completar este laboratorio, podrás:

1. ✅ Comprender la arquitectura de una CNN y sus componentes
2. ✅ Implementar capas convolucionales desde cero
3. ✅ Entender pooling y sus variantes (max pooling, average pooling)
4. ✅ Construir arquitecturas CNN completas
5. ✅ Aplicar CNNs a problemas de clasificación de imágenes
6. ✅ Entender conceptos de padding, stride, y receptive field
7. ✅ Conocer arquitecturas CNN famosas (LeNet, AlexNet, VGG, ResNet)

## Contenido

### 📖 Teoría (`teoria.md`)

Documento completo con los fundamentos teóricos:
- ¿Por qué CNNs para imágenes?
- Operación de convolución
- Filtros y feature maps
- Capas de pooling
- Arquitecturas CNN completas
- CNNs vs Redes totalmente conectadas
- Arquitecturas CNN famosas

### 💻 Práctica (`practica.ipynb`)

Jupyter Notebook interactivo con:
- Implementación de convolución 2D desde cero
- Construcción de capas CNN
- Visualización de filtros y activaciones
- Entrenamiento de CNN en MNIST
- Comparación con redes densas
- Ejercicios progresivos

### 🔧 Código de Ejemplo (`codigo/cnn.py`)

Script Python con implementaciones completas:
- Función `convolve2d()`: Operación de convolución
- Clase `CapaConvolucional`: Capa CNN completa
- Clase `CapaPooling`: Max y average pooling
- Clase `CNN`: Red convolucional completa
- Ejemplos de arquitecturas

## Cómo Usar Este Laboratorio

### Opción 1: Jupyter Notebook (Recomendado)

```bash
# Desde el directorio del repositorio
cd Lab09_Redes_Neuronales_Convolucionales
jupyter notebook practica.ipynb
```

### Opción 2: Script Python

```bash
# Ejecutar el código de ejemplo
python codigo/cnn.py
```

### Opción 3: Lectura y Experimentación

1. Lee `teoria.md` para entender los conceptos
2. Abre `practica.ipynb` en Jupyter
3. Ejecuta cada celda y experimenta con los parámetros
4. Completa los ejercicios propuestos
5. Revisa `codigo/cnn.py` como referencia

## Requisitos

```bash
pip install numpy matplotlib jupyter torch torchvision
```

## Conceptos Clave

- **Convolución**: Operación que aplica filtros para detectar características
- **Filtro/Kernel**: Matriz de pesos que se desliza sobre la entrada
- **Feature Map**: Resultado de aplicar un filtro a la entrada
- **Pooling**: Reducción de dimensionalidad espacial
- **Stride**: Paso del desplazamiento del filtro
- **Padding**: Relleno de bordes para controlar tamaño de salida
- **Receptive Field**: Región de la entrada que afecta a una neurona

## Ejercicios

### Ejercicio 9.1: Convolución Manual
Implementa una convolución 2D sin usar bucles, solo operaciones NumPy.

### Ejercicio 9.2: Filtros Personalizados
Crea filtros para detectar bordes horizontales, verticales y diagonales.

### Ejercicio 9.3: CNN en MNIST
Construye y entrena una CNN simple en el dataset MNIST.

### Ejercicio 9.4: Visualización de Activaciones
Visualiza qué características aprende cada capa de la CNN.

### Ejercicio 9.5: Arquitectura Personalizada (Desafío)
Diseña tu propia arquitectura CNN para clasificar CIFAR-10.

## Ventajas de las CNNs

1. **Invariancia a Traslación**: Detectan características sin importar posición
2. **Compartición de Parámetros**: Menos parámetros que redes densas
3. **Jerarquía de Características**: Aprenden desde bordes hasta objetos
4. **Eficiencia Computacional**: Aprovechan estructura local de imágenes

## Arquitecturas CNN Famosas

### LeNet-5 (1998)
- Primera CNN exitosa
- MNIST: 99%+ precisión
- Arquitectura: CONV → POOL → CONV → POOL → FC

### AlexNet (2012)
- Ganadora ImageNet 2012
- Popularizó deep learning
- 8 capas, ReLU, Dropout

### VGG (2014)
- Capas convolucionales 3x3 apiladas
- Arquitectura muy profunda (16-19 capas)
- Simple pero efectiva

### ResNet (2015)
- Conexiones residuales (skip connections)
- Permite entrenar redes muy profundas (>100 capas)
- Soluciona problema de gradientes que desaparecen

## Aplicaciones

- **Clasificación de Imágenes**: Reconocer objetos en fotos
- **Detección de Objetos**: YOLO, Faster R-CNN
- **Segmentación Semántica**: U-Net, Mask R-CNN
- **Reconocimiento Facial**: FaceNet, DeepFace
- **Diagnóstico Médico**: Detección de tumores en radiografías
- **Vehículos Autónomos**: Detección de señales, peatones
- **Arte y Estilo**: Neural Style Transfer

## Notas Importantes

⚠️ **Dimensiones**: Presta atención a las dimensiones de entrada/salida en cada capa.

💡 **Visualización**: Visualizar filtros y activaciones ayuda a entender qué aprende la red.

🚀 **Transfer Learning**: En la práctica, se suelen usar redes pre-entrenadas y hacer fine-tuning.

## Fórmulas Importantes

### Tamaño de salida de convolución:
```
Output_size = (Input_size - Kernel_size + 2*Padding) / Stride + 1
```

### Número de parámetros en capa convolucional:
```
Params = (Kernel_height * Kernel_width * Input_channels + 1) * Num_filters
```

## Próximo Paso

Una vez completes este laboratorio, continúa con:

👉 **[Lab 11: Redes Neuronales Recurrentes y LSTM](../Lab11_Redes_Neuronales_Recurrentes_LSTM/)**

Exploraremos arquitecturas especializadas para datos secuenciales como texto y series de tiempo.

## Recursos Adicionales

- [CS231n: Convolutional Neural Networks](http://cs231n.stanford.edu/)
- [Visualización de CNNs](https://poloclub.github.io/cnn-explainer/)
- [PyTorch CNN Tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
- [Distill.pub - Feature Visualization](https://distill.pub/2017/feature-visualization/)
- [Neural Style Transfer](https://www.tensorflow.org/tutorials/generative/style_transfer)

## Preguntas Frecuentes

**P: ¿Por qué las CNNs funcionan mejor que redes densas para imágenes?**  
R: Aprovechan la estructura espacial de las imágenes, usan menos parámetros gracias a la compartición de pesos, y son invariantes a la traslación.

**P: ¿Qué tamaño de kernel es mejor?**  
R: Kernels 3x3 son los más comunes por su balance entre campo receptivo y parámetros. A veces se usan 1x1 para cambiar dimensionalidad.

**P: ¿Cuándo usar padding?**  
R: "same" padding mantiene tamaño espacial, útil en redes profundas. "valid" (sin padding) reduce tamaño, útil para reducir dimensionalidad.

**P: ¿Max pooling o average pooling?**  
R: Max pooling es más común porque preserva características más fuertes. Average pooling suaviza pero pierde información.

## Verificación de Conocimientos

- [ ] Entiendo cómo funciona la operación de convolución
- [ ] Puedo calcular dimensiones de salida de capas CNN
- [ ] Sé implementar convolución y pooling desde cero
- [ ] Entiendo la diferencia entre CNNs y redes densas
- [ ] Conozco arquitecturas CNN famosas y sus innovaciones
- [ ] Puedo construir y entrenar una CNN con PyTorch/TensorFlow
