# Lab 01: Introducción a las Neuronas

## Descripción

Este laboratorio introduce los conceptos fundamentales de las neuronas artificiales, la unidad básica de las redes neuronales. Implementaremos neuronas desde cero siguiendo la filosofía del libro "Neural Networks from Scratch in Python".

## Objetivos de Aprendizaje

Al completar este laboratorio, podrás:

1. ✅ Comprender qué es una neurona artificial y sus componentes
2. ✅ Implementar una neurona simple desde cero
3. ✅ Entender y utilizar el producto punto (dot product)
4. ✅ Usar NumPy para cálculos eficientes
5. ✅ Crear capas de neuronas
6. ✅ Procesar múltiples muestras en batch

## Contenido

### 📖 Teoría (`teoria.md`)

Documento completo con los fundamentos teóricos:
- ¿Qué es una neurona artificial?
- Componentes: entradas, pesos, bias
- Función de suma ponderada
- Producto punto vectorial
- Limitaciones y potencial de las neuronas

### 💻 Práctica (`practica.ipynb`)

Jupyter Notebook interactivo con:
- Implementación paso a paso de una neurona
- Ejercicios prácticos progresivos
- Visualizaciones de resultados
- Desafíos de programación

### 🔧 Código de Ejemplo (`codigo/neurona.py`)

Script Python con implementaciones completas:
- Función `neurona_simple()`: Implementación básica
- Función `neurona_numpy()`: Versión optimizada con NumPy
- Clase `Neurona`: Encapsulación orientada a objetos
- Clase `CapaNeuronal`: Múltiples neuronas trabajando juntas

## Cómo Usar Este Laboratorio

### Opción 1: Jupyter Notebook (Recomendado)

```bash
# Desde el directorio del repositorio
cd Lab01_Introduccion_Neuronas
jupyter notebook practica.ipynb
```

### Opción 2: Script Python

```bash
# Ejecutar el código de ejemplo
python codigo/neurona.py
```

### Opción 3: Lectura y Experimentación

1. Lee `teoria.md` para entender los conceptos
2. Abre `practica.ipynb` en Jupyter
3. Ejecuta cada celda y experimenta con los valores
4. Completa los ejercicios propuestos
5. Revisa `codigo/neurona.py` como referencia

## Requisitos

```bash
pip install numpy matplotlib jupyter
```

## Conceptos Clave

- **Neurona**: Unidad básica que procesa información
- **Pesos (Weights)**: Parámetros que determinan la importancia de cada entrada
- **Bias**: Parámetro que permite ajustar el umbral de activación
- **Producto Punto**: Operación fundamental para calcular salidas
- **Forward Pass**: Cálculo de la salida dadas las entradas

## Ejercicios

### Ejercicio 1.1: Experimentación
Modifica pesos y bias para observar cómo cambian las salidas.

### Ejercicio 1.2: Función Personalizada
Implementa tu propia función `calcular_neurona()`.

### Ejercicio 1.3: Capa Aleatoria
Crea una capa de 4 neuronas con valores aleatorios.

### Ejercicio 1.4: Clase CapaNeuronal (Desafío)
Implementa una clase completa para una capa de neuronas.

## Notas Importantes

⚠️ **Sin Funciones de Activación**: En este laboratorio trabajamos sin funciones de activación para enfocarnos en los conceptos básicos. Las introduciremos en Lab 03.

💡 **Por qué NumPy**: NumPy es fundamental porque:
- Operaciones vectorizadas son ~100x más rápidas
- Código más limpio y legible
- Estándar en la industria de Machine Learning

## Próximo Paso

Una vez completes este laboratorio, continúa con:

👉 **[Lab 02: Primera Red Neuronal](../Lab02_Primera_Red_Neuronal/)**

Combinaremos múltiples capas de neuronas para crear nuestra primera red neuronal completa.

## Recursos Adicionales

- [NumPy Documentation](https://numpy.org/doc/)
- [Neural Networks from Scratch - YouTube](https://www.youtube.com/watch?v=Wo5dMEP_BbI)
- [3Blue1Brown - Neural Networks](https://www.youtube.com/watch?v=aircAruvnKk)

## Preguntas Frecuentes

**P: ¿Por qué multiplicamos por 0.01 al inicializar pesos?**  
R: Valores iniciales pequeños ayudan en el entrenamiento. Lo explicaremos en detalle en Lab 05.

**P: ¿Puedo usar una neurona para cualquier problema?**  
R: Una sola neurona solo puede resolver problemas linealmente separables. Necesitaremos redes para problemas complejos.

**P: ¿Qué es el "forward pass"?**  
R: Es el proceso de calcular la salida de la red dadas las entradas. Lo complementaremos con "backward pass" en Lab 05.
