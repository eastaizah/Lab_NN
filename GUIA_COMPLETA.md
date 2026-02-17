# Guía Completa del Curso de Redes Neuronales

## 📚 Descripción General

Este repositorio contiene un curso completo de Redes Neuronales, Deep Learning e Inteligencia Artificial Generativa, diseñado para aprender desde cero con un enfoque muy didáctico, basado en el libro "Neural Networks from Scratch in Python".

## 🎯 Objetivos del Curso

Al completar este curso, serás capaz de:

1. ✅ Comprender los fundamentos matemáticos de las redes neuronales
2. ✅ Implementar redes neuronales completamente desde cero en Python
3. ✅ Entrenar modelos para problemas reales de clasificación y regresión
4. ✅ Usar frameworks modernos como PyTorch y TensorFlow
5. ✅ Entender los conceptos básicos de IA Generativa (VAE, GAN)
6. ✅ Aplicar buenas prácticas en el desarrollo de modelos de ML

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

### Módulo 4: Frameworks y IA Generativa (Labs 07-08)

#### [Lab 07: Frameworks de Deep Learning](Lab07_Frameworks_DeepLearning/)
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

#### [Lab 08: Inteligencia Artificial Generativa](Lab08_IA_Generativa/)
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

### Semana 3: Entrenamiento
- **Día 1-3**: Lab 05 - Backpropagation
- **Día 4-5**: Lab 06 - Entrenamiento

### Semana 4: Frameworks y Generativa
- **Día 1-2**: Lab 07 - PyTorch/TensorFlow
- **Día 3-4**: Lab 08 - IA Generativa
- **Día 5**: Proyecto final

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

**Proyecto Final** (Después de Lab 08):
- Modelo generativo para crear imágenes
- O clasificador usando PyTorch/TensorFlow

## 📚 Recursos Adicionales

### Libros
- **"Neural Networks from Scratch in Python"** - Harrison Kinsley & Daniel Kukieła
- **"Deep Learning"** - Ian Goodfellow, Yoshua Bengio, Aaron Courville
- **"Neural Networks and Deep Learning"** - Michael Nielsen

### Cursos Online
- [3Blue1Brown - Neural Networks](https://www.youtube.com/watch?v=aircAruvnKk)
- [Fast.ai - Practical Deep Learning](https://www.fast.ai/)
- [Stanford CS231n](http://cs231n.stanford.edu/)

### Herramientas Interactivas
- [TensorFlow Playground](http://playground.tensorflow.org/)
- [CNN Explainer](https://poloclub.github.io/cnn-explainer/)
- [Distill.pub](https://distill.pub/)

## ❓ FAQ (Preguntas Frecuentes)

**P: ¿Necesito saber matemáticas avanzadas?**
R: No. El curso explica los conceptos matemáticos necesarios. Álgebra y cálculo básico son suficientes.

**P: ¿Cuánto tiempo toma completar el curso?**
R: Aproximadamente 4-6 semanas dedicando 2-3 horas diarias. Puedes ir a tu propio ritmo.

**P: ¿Puedo saltar laboratorios?**
R: No recomendado. Cada lab construye sobre los anteriores. El orden es importante.

**P: ¿Qué hago si me atasco?**
R: 
1. Revisa la teoría nuevamente
2. Estudia el código de ejemplo
3. Busca en los recursos adicionales
4. Abre un issue en GitHub

**P: ¿Necesito una GPU?**
R: No para Labs 01-06. Labs 07-08 funcionan en CPU, aunque GPU acelera el entrenamiento.

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

*Última actualización: Febrero 2026*
