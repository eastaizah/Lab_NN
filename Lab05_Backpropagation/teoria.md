# Teoría: Backpropagation

## Introducción

**Backpropagation** (propagación hacia atrás) es el algoritmo fundamental que hace posible el entrenamiento de redes neuronales profundas. Es el método mediante el cual calculamos los gradientes de la función de pérdida con respecto a todos los parámetros de la red.

##¿Qué es Backpropagation?

Backpropagation es la aplicación inteligente de la **regla de la cadena del cálculo** para calcular gradientes de manera eficiente en redes neuronales.

**Sin backpropagation**: Tendríamos que calcular derivadas manualmente para cada parámetro → imposible para redes grandes

**Con backpropagation**: Calculamos gradientes automáticamente de forma eficiente

## El Problema que Resuelve

Queremos minimizar la pérdida L ajustando los pesos W:

```
min L(W, X, y)
```

Para usar gradient descent necesitamos:
```
∂L/∂W  (gradiente de la pérdida respecto a cada peso)
```

En una red con millones de parámetros, ¿cómo calculamos esto eficientemente?

**Respuesta**: Backpropagation

## La Regla de la Cadena

**Fundamento matemático** de backpropagation.

### Para una variable:

Si `y = f(g(x))`, entonces:
```
dy/dx = (dy/dg) * (dg/dx)
```

### Ejemplo simple:

```
y = (3x + 2)²

Descomponer:
  u = 3x + 2
  y = u²

Aplicar regla de la cadena:
  dy/dx = (dy/du) * (du/dx)
        = (2u) * (3)
        = 2(3x + 2) * 3
        = 6(3x + 2)
```

### Para múltiples variables:

Si `z = f(x, y)`, y `x = g(t)`, `y = h(t)`, entonces:
```
dz/dt = (∂z/∂x)*(dx/dt) + (∂z/∂y)*(dy/dt)
```

## Grafos Computacionales

Backpropagation se entiende mejor mediante **grafos computacionales**.

### Ejemplo: z = (x + y) * w

```
    x ─┐
        ├─→ [+] ─→ q ─┐
    y ─┘              │
                      ├─→ [*] ─→ z
                  w ──┘
```

**Forward pass** (izquierda → derecha):
```
1. q = x + y
2. z = q * w
```

**Backward pass** (derecha ← izquierda):
```
1. ∂z/∂z = 1 (empezamos aquí)
2. ∂z/∂q = w, ∂z/∂w = q
3. ∂z/∂x = (∂z/∂q)*(∂q/∂x) = w * 1 = w
4. ∂z/∂y = (∂z/∂q)*(∂q/∂y) = w * 1 = w
```

## Backpropagation en una Red Neuronal Simple

### Arquitectura:

```
Input (x) → [W1, b1] → ReLU → [W2, b2] → Sigmoid → Output (ŷ) → Loss (L)
```

### Forward Pass:

```python
# Capa 1
z1 = W1 @ x + b1
a1 = ReLU(z1)

# Capa 2
z2 = W2 @ a1 + b2
a2 = sigmoid(z2)

# Pérdida
L = binary_crossentropy(a2, y)
```

### Backward Pass:

Necesitamos: `∂L/∂W1`, `∂L/∂b1`, `∂L/∂W2`, `∂L/∂b2`

**Paso 1**: Gradiente de la pérdida
```
∂L/∂a2 = a2 - y  (para BCE + sigmoid)
```

**Paso 2**: Capa 2 (backward)
```
∂L/∂z2 = ∂L/∂a2 * sigmoid'(z2) = ∂L/∂a2 * a2 * (1 - a2)
∂L/∂W2 = ∂L/∂z2 @ a1.T
∂L/∂b2 = sum(∂L/∂z2)
∂L/∂a1 = W2.T @ ∂L/∂z2
```

**Paso 3**: Capa 1 (backward)
```
∂L/∂z1 = ∂L/∂a1 * ReLU'(z1)
∂L/∂W1 = ∂L/∂z1 @ x.T
∂L/∂b1 = sum(∂L/∂z1)
```

## Algoritmo de Backpropagation (General)

### Forward Pass:

```
Para cada capa l = 1, 2, ..., L:
    1. z[l] = W[l] @ a[l-1] + b[l]
    2. a[l] = activation[l](z[l])

Calcular pérdida L
```

### Backward Pass:

```
Inicializar: dL/da[L] = gradient de la pérdida

Para cada capa l = L, L-1, ..., 1 (hacia atrás):
    1. dL/dz[l] = dL/da[l] ⊙ activation'[l](z[l])
    2. dL/dW[l] = dL/dz[l] @ a[l-1].T
    3. dL/db[l] = sum(dL/dz[l])
    4. dL/da[l-1] = W[l].T @ dL/dz[l]
```

Donde `⊙` denota producto elemento a elemento (Hadamard).

## Derivadas de Funciones Comunes

### Activaciones:

**Sigmoid**:
```
σ'(x) = σ(x) * (1 - σ(x))
```

**Tanh**:
```
tanh'(x) = 1 - tanh²(x)
```

**ReLU**:
```
ReLU'(x) = 1 si x > 0
          = 0 si x ≤ 0
```

**Softmax** (con Cross-Entropy):
```
∂L/∂z = ŷ - y  (simplificación notable!)
```

### Pérdidas:

**MSE**:
```
∂MSE/∂ŷ = 2(ŷ - y) / n
```

**Binary Cross-Entropy** (con Sigmoid):
```
∂BCE/∂z = ŷ - y
```

**Categorical Cross-Entropy** (con Softmax):
```
∂CCE/∂z = ŷ - y
```

## Ejemplo Numérico Completo

### Setup:

```
Red: 2 → 3 → 1
Activaciones: ReLU → Sigmoid
Pérdida: Binary Cross-Entropy
```

### Datos:

```
x = [1, 2]
y = 1  (clase positiva)
```

### Pesos (iniciales):

```
W1 = [[0.1, 0.2],
      [0.3, 0.4],
      [0.5, 0.6]]
b1 = [0, 0, 0]

W2 = [[0.7],
      [0.8],
      [0.9]]
b2 = [0]
```

### Forward:

```
# Capa 1
z1 = W1 @ x + b1 = [[0.1*1 + 0.2*2],   = [0.5,
                     [0.3*1 + 0.4*2],      1.1,
                     [0.5*1 + 0.6*2]]      1.7]

a1 = ReLU(z1) = [0.5, 1.1, 1.7]

# Capa 2
z2 = W2.T @ a1 + b2 = 0.7*0.5 + 0.8*1.1 + 0.9*1.7 + 0 = 2.76

a2 = sigmoid(z2) = 0.941  (predicción)

# Pérdida
L = -[y*log(a2) + (1-y)*log(1-a2)]
  = -[1*log(0.941) + 0*log(0.059)]
  = 0.061
```

### Backward:

```
# Gradiente de pérdida
dL/da2 = a2 - y = 0.941 - 1 = -0.059

# Capa 2
dL/dz2 = dL/da2 * sigmoid'(z2) = -0.059 * 0.941 * (1-0.941) = -0.0033

dL/dW2 = dL/dz2 * a1 = -0.0033 * [0.5, 1.1, 1.7]
       = [-0.0017, -0.0036, -0.0056]

dL/db2 = dL/dz2 = -0.0033

dL/da1 = W2 * dL/dz2 = [0.7, 0.8, 0.9] * -0.0033
       = [-0.0023, -0.0026, -0.0030]

# Capa 1
dL/dz1 = dL/da1 ⊙ ReLU'(z1)
       = [-0.0023, -0.0026, -0.0030] ⊙ [1, 1, 1]
       = [-0.0023, -0.0026, -0.0030]

dL/dW1 = dL/dz1 @ x.T
       = [[-0.0023], [-0.0026], [-0.0030]] @ [[1, 2]]
       = [[-0.0023, -0.0046],
          [-0.0026, -0.0052],
          [-0.0030, -0.0060]]

dL/db1 = dL/dz1 = [-0.0023, -0.0026, -0.0030]
```

### Update (con α = 0.1):

```
W1 = W1 - 0.1 * dL/dW1
W2 = W2 - 0.1 * dL/dW2
... etc
```

## Eficiencia de Backpropagation

**Por qué es eficiente:**

1. **Reutilización de cálculos**: Los gradientes intermedios se reutilizan
2. **Un solo pase hacia atrás**: Calcula todos los gradientes simultáneamente
3. **Complejidad**: O(n) donde n es el número de parámetros

**Sin backprop**: Tendríamos que calcular cada derivada parcial independientemente → O(n²) o peor

## Problemas Comunes

### 1. Gradientes que Exploran (Exploding Gradients)

**Síntoma**: Gradientes se vuelven muy grandes
**Causa**: Multiplicación repetida de valores > 1
**Solución**: 
- Gradient clipping
- Mejores inicializaciones
- Batch normalization

### 2. Gradientes que Desaparecen (Vanishing Gradients)

**Síntoma**: Gradientes se vuelven muy pequeños
**Causa**: Multiplicación repetida de valores < 1 (ej: derivadas de sigmoid)
**Solución**:
- Usar ReLU en lugar de Sigmoid/Tanh
- Skip connections (ResNet)
- Batch normalization

### 3. Errores de Implementación

**Común**: Dimensiones incorrectas en multiplicaciones matriciales
**Solución**: Verificar dimensiones cuidadosamente

**Común**: Olvidar transponer matrices
**Solución**: Seguir el algoritmo paso a paso

## Vectorización

Para eficiencia, procesamos **batches** de datos:

### Sin vectorización (loop):
```python
for sample in batch:
    loss += calculate_loss(sample)
    grads += calculate_gradients(sample)
```

### Con vectorización:
```python
# Procesa todo el batch simultáneamente
loss = calculate_loss(batch)
grads = calculate_gradients(batch)
```

**Ventaja**: Órdenes de magnitud más rápido (usa operaciones matriciales optimizadas)

## Verificación de Gradientes

**Importante**: Siempre verifica que tu implementación sea correcta.

### Gradient Checking:

```python
def numerical_gradient(f, x, epsilon=1e-5):
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_plus[i] += epsilon
        x_minus = x.copy()
        x_minus[i] -= epsilon
        
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * epsilon)
    
    return grad

# Comparar
analytical_grad = backprop_gradient()
numerical_grad = numerical_gradient(loss_function, params)

difference = norm(analytical_grad - numerical_grad) / (norm(analytical_grad) + norm(numerical_grad))

if difference < 1e-7:
    print("✓ Gradientes correctos!")
else:
    print("✗ Error en gradientes")
```

## Conceptos Clave

1. **Regla de la cadena**: Fundamento matemático
2. **Forward pass**: Calcular predicciones y guardar valores intermedios
3. **Backward pass**: Calcular gradientes usando valores guardados
4. **Grafos computacionales**: Visualización útil
5. **Eficiencia**: Un pase hacia atrás calcula todos los gradientes
6. **Verificación**: Siempre compara con gradientes numéricos

## Resumen

Backpropagation es:
- El algoritmo que hace posible el deep learning
- Aplicación eficiente de la regla de la cadena
- Requiere guardar valores intermedios (cache)
- Calcula todos los gradientes en un solo pase
- Debe ser verificado numéricamente

## Próximos Pasos

En el siguiente laboratorio usaremos backpropagation para **entrenar** redes neuronales completas en problemas reales.
