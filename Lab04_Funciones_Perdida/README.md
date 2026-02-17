# Lab 04: Funciones de Pérdida y Optimización

## Objetivos de Aprendizaje

Al completar este laboratorio, serás capaz de:

1. Comprender qué son las funciones de pérdida y por qué son necesarias
2. Implementar desde cero MSE, MAE, Binary Cross-Entropy y Categorical Cross-Entropy
3. Entender las diferencias entre pérdidas para regresión y clasificación
4. Implementar gradient descent básico
5. Reconocer overfitting y sus causas
6. Elegir la función de pérdida apropiada para diferentes problemas

## Estructura del Laboratorio

```
Lab04_Funciones_Perdida/
├── README.md                # Esta guía
├── teoria.md               # Fundamentos teóricos
├── practica.ipynb         # Notebook interactivo
└── codigo/
    └── perdida.py         # Implementaciones completas
```

## Requisitos Previos

- Completar Labs 01-03
- Comprensión de funciones de activación
- Conocimientos básicos de cálculo (derivadas)

## Contenido Teórico

El archivo `teoria.md` cubre:

- **Función de pérdida**: Qué es y por qué la necesitamos
- **MSE y MAE**: Para problemas de regresión
- **Cross-Entropy**: Para clasificación binaria y multiclase
- **Gradient Descent**: Algoritmo de optimización fundamental
- **Learning Rate**: Su importancia y cómo elegirlo
- **Overfitting**: Qué es y cómo detectarlo
- **Regularización**: Técnicas básicas

## Práctica

### Parte 1: Ejecutar Código Principal (20 min)

```bash
cd codigo/
python perdida.py
```

Esto generará:
- Comparación MSE vs MAE con outliers
- Visualización de Binary Cross-Entropy
- Demostración de Gradient Descent con diferentes learning rates
- Ejemplo de overfitting

### Parte 2: Notebook Interactivo (60 min)

Abre `practica.ipynb` y completa los ejercicios:

```bash
jupyter notebook practica.ipynb
```

Incluye:
1. Implementación de funciones de pérdida
2. Comparación de pérdidas en diferentes escenarios
3. Gradient descent paso a paso
4. Experimentos con learning rates
5. Detección de overfitting

### Parte 3: Experimentos (30 min)

1. **Experimento 1**: Comparar MSE vs MAE con diferentes niveles de outliers
2. **Experimento 2**: Probar gradient descent con learning rates extremos
3. **Experimento 3**: Entrenar un modelo hasta que ocurra overfitting

## Conceptos Clave

### 1. Elección de Función de Pérdida

```
Tipo de Problema          | Función de Pérdida
------------------------- | ------------------------
Regresión                 | MSE, MAE
Clasificación Binaria     | Binary Cross-Entropy
Clasificación Multiclase  | Categorical Cross-Entropy
```

### 2. Combinaciones Ideales

| Problema | Activación | Pérdida | Por qué |
|----------|-----------|---------|---------|
| Regresión | Lineal | MSE | Natural para valores continuos |
| Binaria | Sigmoid | Binary CE | Derivada simple |
| Multiclase | Softmax | Categorical CE | Derivada simple |

### 3. Learning Rate

```python
α muy pequeño (0.0001):  Lento pero estable
α moderado (0.01):       Balance ideal (usual)
α muy grande (1.0):      Rápido pero inestable
```

## Ejercicios

### Ejercicio 1: Implementar Huber Loss

Combina MSE y MAE:
```python
Huber(y, ŷ) = 0.5 * (y - ŷ)²     si |y - ŷ| ≤ δ
             = δ * |y - ŷ| - 0.5δ²  en otro caso
```

### Ejercicio 2: Learning Rate Adaptativo

Implementa un scheduler que reduce el learning rate:
```python
lr_new = lr_initial * 0.95^epoch
```

### Ejercicio 3: Early Stopping

Implementa early stopping para prevenir overfitting:
- Monitorea pérdida de validación
- Detén si no mejora en N épocas

### Ejercicio 4: Comparación de Optimizadores

Compara:
- Batch Gradient Descent
- Stochastic Gradient Descent
- Mini-batch Gradient Descent

## Preguntas de Reflexión

1. **¿Por qué MSE penaliza más los errores grandes?**
   
   Pista: Piensa en el término cuadrático.

2. **¿Por qué usamos Cross-Entropy en lugar de MSE para clasificación?**
   
   Pista: Considera la interpretación probabilística.

3. **¿Qué pasa si el learning rate es demasiado grande?**
   
   Pista: Piensa en términos de convergencia.

4. **¿Cómo detectas overfitting?**
   
   Pista: Compara pérdida de train vs validación.

## Verificación de Comprensión

Después de completar el laboratorio, deberías poder:

- [ ] Explicar qué mide una función de pérdida
- [ ] Implementar MSE, MAE y Cross-Entropy desde cero
- [ ] Elegir la pérdida correcta para un problema dado
- [ ] Implementar gradient descent básico
- [ ] Comprender el efecto del learning rate
- [ ] Identificar overfitting en gráficas de entrenamiento
- [ ] Calcular derivadas de funciones de pérdida

## Errores Comunes

### Error 1: Usar MSE para Clasificación

**Problema**: MSE no es adecuada para clasificación

**Solución**: Usar Binary o Categorical Cross-Entropy

**Por qué**: Cross-Entropy tiene mejor interpretación probabilística

### Error 2: Learning Rate Muy Grande

**Síntoma**: Pérdida oscila o diverge

**Solución**: Reducir learning rate en órdenes de magnitud

**Típico**: Probar 0.1, 0.01, 0.001

### Error 3: No Normalizar Datos

**Síntoma**: Convergencia lenta o inestable

**Solución**: Normalizar/estandarizar entradas

```python
X = (X - mean) / std
```

### Error 4: Confundir Pérdida y Métrica

**Recordar**:
- **Pérdida**: Lo que optimizamos (ej: Cross-Entropy)
- **Métrica**: Lo que reportamos (ej: Accuracy)

## Visualizaciones Importantes

### 1. Curva de Aprendizaje

```
Pérdida |
        |  \
        |   \_____ convergencia
        |
        +----------------> Épocas
```

### 2. Overfitting

```
Pérdida |  train \___
        |              
        |  val    /‾‾‾
        +----------------> Épocas
```

### 3. Learning Rate

```
Pérdida |     lr grande \/\/\/
        |     
        |     lr pequeño \____
        +---------------------  ‾> Épocas
```

## Recursos Adicionales

### Lecturas

1. **Loss Functions**: Deep Learning Book, Chapter 5
2. **Optimization**: Deep Learning Book, Chapter 8
3. **Cross-Entropy**: "Pattern Recognition and Machine Learning" (Bishop)

### Papers

- "Adam: A Method for Stochastic Optimization" (Kingma & Ba, 2014)
- "On the importance of initialization and momentum in deep learning" (Sutskever et al., 2013)

### Herramientas Interactivas

- [Loss Landscape Visualization](https://losslandscape.com)
- [TensorFlow Playground](https://playground.tensorflow.org)

## Solución de Problemas

### Pérdida no disminuye

**Posibles causas**:
1. Learning rate muy pequeño → Aumentar
2. Inicialización mala → Re-inicializar
3. Datos no normalizados → Normalizar
4. Arquitectura inadecuada → Revisar modelo

### Pérdida es NaN

**Posibles causas**:
1. Learning rate muy grande → Reducir
2. Overflow en exp() → Usar estabilización numérica
3. División por cero → Añadir epsilon

**Solución**:
```python
# Softmax estable
exp_x = np.exp(x - np.max(x))

# Evitar log(0)
epsilon = 1e-15
loss = -np.log(y_pred + epsilon)
```

### Convergencia lenta

**Soluciones**:
1. Aumentar learning rate
2. Usar momentum (lab posterior)
3. Mejor inicialización
4. Normalizar datos

## Próximo Laboratorio

En **Lab 05: Backpropagation**, aprenderemos:
- Chain rule para redes neuronales
- Grafos computacionales
- Implementación completa de backpropagation
- Cálculo eficiente de gradientes

Backpropagation es el algoritmo que hace posible el entrenamiento de redes profundas al calcular los gradientes necesarios para gradient descent.

## Notas Finales

Las funciones de pérdida son fundamentales porque:
1. Cuantifican el error del modelo
2. Guían la optimización
3. Permiten comparar modelos

La elección correcta de la función de pérdida puede hacer la diferencia entre un modelo que funciona y uno que no.

**Recuerda**: 
- MSE para regresión
- Cross-Entropy para clasificación
- Learning rate: empieza con 0.01
- Monitorea overfitting siempre

---

**¿Preguntas?** Revisa la teoría, experimenta con los notebooks, y recuerda: ¡la práctica hace al maestro!

**¡Éxito! 🎯**
