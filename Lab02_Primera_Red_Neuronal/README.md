# Lab 02: Primera Red Neuronal

## Descripción

En este laboratorio construimos nuestra primera red neuronal completa conectando múltiples capas de neuronas. Implementamos forward propagation y exploramos diferentes arquitecturas.

## Objetivos de Aprendizaje

Al completar este laboratorio, podrás:

1. ✅ Construir redes neuronales con múltiples capas
2. ✅ Implementar forward propagation
3. ✅ Entender cómo fluyen los datos a través de la red
4. ✅ Diseñar arquitecturas para diferentes problemas
5. ✅ Calcular el número de parámetros de una red
6. ✅ Comprender el problema de la linealidad

## Contenido

### 📖 Teoría (`teoria.md`)

Fundamentos teóricos completos:
- Arquitectura de redes neuronales
- Capas: entrada, ocultas, salida
- Forward propagation
- Dimensiones de matrices
- Inicialización de pesos
- Redes profundas vs anchas
- El problema sin funciones de activación

### 💻 Práctica (`practica.ipynb`)

Jupyter Notebook interactivo con:
- Construcción de redes multicapa
- Experimentación con arquitecturas
- Visualización de activaciones
- Ejercicios prácticos
- Demostración del problema de linealidad

### 🔧 Código de Ejemplo (`codigo/red_neuronal.py`)

Implementación completa:
- Clase `CapaDensa`: Capa individual
- Clase `RedNeuronal`: Red completa
- Función `visualizar_activaciones()`: Visualización de flujo de datos
- Múltiples ejemplos de uso

## Cómo Usar Este Laboratorio

### Prerequisitos

Completa primero el [Lab 01: Introducción a las Neuronas](../Lab01_Introduccion_Neuronas/)

### Opción 1: Jupyter Notebook (Recomendado)

```bash
cd Lab02_Primera_Red_Neuronal
jupyter notebook practica.ipynb
```

### Opción 2: Script Python

```bash
python codigo/red_neuronal.py
```

### Opción 3: Estudio Guiado

1. Lee `teoria.md` para comprender los conceptos
2. Abre y ejecuta `practica.ipynb`
3. Completa los ejercicios
4. Experimenta modificando arquitecturas
5. Revisa `codigo/red_neuronal.py` como referencia

## Conceptos Clave

### Arquitectura de Red

```
[n_entrada] → [n_capa1] → [n_capa2] → ... → [n_salida]
```

**Ejemplo**: `[784, 128, 64, 10]`
- 784 características de entrada
- 128 neuronas en capa oculta 1
- 64 neuronas en capa oculta 2
- 10 clases de salida

### Forward Propagation

```python
activacion = X
for cada capa:
    activacion = capa.forward(activacion)
return activacion
```

### Número de Parámetros

Para cada capa:
```
parámetros = (n_entradas × n_neuronas) + n_neuronas
```

## Ejemplos Incluidos

### 1. Red Simple
Red básica de 2 capas para entender el flujo de datos.

### 2. Red para MNIST
Arquitectura típica para clasificación de dígitos: `[784, 128, 64, 10]`

### 3. Comparación de Arquitecturas
Diferentes configuraciones y su impacto en parámetros.

### 4. Visualización
Gráficos mostrando cómo se transforman los datos.

### 5. Profunda vs Ancha
Comparación de diferentes estrategias de diseño.

## Ejercicios

### Ejercicio 2.1: Seguimiento de Dimensiones
Traza las dimensiones de los tensores a través de la red.

### Ejercicio 2.2: Contar Parámetros
Calcula manualmente los parámetros de una red `[10, 20, 15, 5]`.

### Ejercicio 2.3: Diseño de Arquitectura (Desafío)
Diseña dos redes con ~10,000 parámetros pero arquitecturas diferentes.

## Notas Importantes

⚠️ **Limitación Sin Activaciones**: Sin funciones de activación no lineales, cualquier red profunda es matemáticamente equivalente a una red de una sola capa.

💡 **Por qué Importa**:
- Capas múltiples solo son útiles con no-linealidad
- En Lab 03 añadiremos funciones de activación
- Entonces veremos el verdadero poder de las redes profundas

## Visualizaciones

El notebook incluye visualizaciones que muestran:
- Cómo cambia la dimensionalidad en cada capa
- Valores de activación por neurona
- Comparación de diferentes arquitecturas

## Decisiones de Diseño

### ¿Cuántas capas?
- **Problemas simples**: 1-2 capas ocultas
- **Problemas complejos**: 3-5+ capas ocultas
- **Deep Learning**: 10-100+ capas (con técnicas especiales)

### ¿Cuántas neuronas por capa?
- Generalmente, disminuir hacia la salida
- Depende de la complejidad del problema
- Experimentación es clave

### Inicialización de Pesos
- ❌ No todo ceros (simetría)
- ❌ No valores muy grandes (gradientes explotan)
- ✅ Valores pequeños aleatorios
- ✅ Xavier/He initialization (veremos en Lab 05)

## Próximo Paso

Una vez completes este laboratorio, continúa con:

👉 **[Lab 03: Funciones de Activación](../Lab03_Funciones_Activacion/)**

Aprenderemos sobre ReLU, Sigmoid, Tanh y Softmax, que permitirán a nuestras redes aprender patrones no lineales.

## Recursos Adicionales

- [Visualizing Neural Networks](http://playground.tensorflow.org/)
- [CS231n: Neural Networks Part 1](https://cs231n.github.io/neural-networks-1/)
- [Deep Learning Book - Chapter 6](https://www.deeplearningbook.org/contents/mlp.html)

## Preguntas Frecuentes

**P: ¿Más capas siempre es mejor?**  
R: No necesariamente. Más capas pueden llevar a overfitting y son más difíciles de entrenar. El balance es importante.

**P: ¿Cómo sé cuántas neuronas usar?**  
R: Es parte del "arte" del ML. Se determina mediante experimentación y validación. Generalmente se empieza con valores estándar y se ajusta.

**P: ¿Por qué la red sin activación es lineal?**  
R: Porque la composición de funciones lineales es lineal. Necesitamos no-linealidad para resolver problemas complejos.

**P: ¿Puedo tener capas de diferentes tamaños?**  
R: ¡Sí! No hay restricción. Puedes aumentar, disminuir, o mantener el tamaño entre capas según tu necesidad.
