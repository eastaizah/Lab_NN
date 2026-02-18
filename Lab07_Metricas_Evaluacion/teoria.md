# Teoría: Matriz de Confusión y Métricas de Evaluación

## Introducción

Después de entrenar un modelo de clasificación, necesitamos **evaluar su rendimiento** de manera rigurosa. No basta con saber que el modelo "funciona" - necesitamos entender **qué tan bien funciona**, **en qué se equivoca**, y **si es adecuado para nuestro problema específico**.

Las **métricas de evaluación** y la **matriz de confusión** son herramientas fundamentales para este análisis.

## La Matriz de Confusión

### Definición

La **matriz de confusión** es una tabla que muestra el rendimiento de un modelo de clasificación comparando las **predicciones** con las **etiquetas verdaderas**.

### Caso Binario (2 clases)

Para un problema de clasificación binaria (por ejemplo, detectar spam vs no-spam), la matriz de confusión tiene esta estructura:

```
                    Predicción
                 Positivo  Negativo
              ┌──────────┬──────────┐
Verdadero  P  │    TP    │    FN    │
           N  │    FP    │    TN    │
              └──────────┴──────────┘
```

Donde:
- **TP (True Positives)**: Positivos correctamente identificados
- **TN (True Negatives)**: Negativos correctamente identificados
- **FP (False Positives)**: Negativos incorrectamente identificados como positivos (Error Tipo I)
- **FN (False Negatives)**: Positivos incorrectamente identificados como negativos (Error Tipo II)

### Ejemplo Práctico

Imagina un modelo que detecta si un email es spam (positivo) o no spam (negativo) en 100 emails:

```
                    Predicción
                 Spam    No Spam
              ┌─────────┬─────────┐
Verdadero  S  │   40    │   10    │  (50 spam reales)
        No S  │    5    │   45    │  (50 no spam reales)
              └─────────┴─────────┘
```

Interpretación:
- **TP = 40**: 40 spam correctamente detectados
- **FN = 10**: 10 spam que se perdieron (falsos negativos)
- **FP = 5**: 5 emails normales marcados como spam (falsos positivos)
- **TN = 45**: 45 emails normales correctamente identificados

### Caso Multiclase (N clases)

Para problemas con más de 2 clases, la matriz se expande:

```
                Predicción
             Gato  Perro  Pájaro
          ┌──────┬──────┬──────┐
Real Gato │  35  │   3  │   2  │
     Perro│   2  │  38  │   0  │
   Pájaro │   1  │   1  │  18  │
          └──────┴──────┴──────┘
```

En este caso:
- Diagonal principal = predicciones correctas
- Fuera de la diagonal = errores
- Cada fila suma el total de muestras de esa clase real

## Métricas Derivadas de la Matriz de Confusión

### 1. Accuracy (Exactitud)

**Definición**: Proporción de predicciones correctas sobre el total.

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**Ejemplo**:
```
Accuracy = (40 + 45) / 100 = 0.85 = 85%
```

**Cuándo usar**:
- Dataset balanceado (clases con cantidad similar de muestras)
- Todos los errores tienen el mismo costo

**Limitaciones**:
- **Paradoja del Accuracy**: En datasets desbalanceados puede ser engañosa

**Ejemplo de la paradoja**:
```
Dataset: 95 no-spam, 5 spam
Modelo que predice todo como "no-spam":
Accuracy = 95/100 = 95% (¡pero no detecta ningún spam!)
```

### 2. Precision (Precisión)

**Definición**: De todas las predicciones positivas, ¿cuántas fueron correctas?

```
Precision = TP / (TP + FP)
```

**Ejemplo**:
```
Precision = 40 / (40 + 5) = 0.889 = 88.9%
```

**Interpretación**:
- "¿Qué tan confiable es cuando predice positivo?"
- "Si marca un email como spam, ¿qué probabilidad hay de que realmente sea spam?"

**Cuándo priorizar Precision**:
- Cuando los **falsos positivos son costosos**
- Ejemplo: Diagnósticos médicos que requieren tratamientos caros/peligrosos
- Ejemplo: Sistema de spam que no debe bloquear emails importantes

### 3. Recall (Sensibilidad, Sensitivity, True Positive Rate)

**Definición**: De todos los casos positivos reales, ¿cuántos detectamos?

```
Recall = TP / (TP + FN)
```

**Ejemplo**:
```
Recall = 40 / (40 + 10) = 0.80 = 80%
```

**Interpretación**:
- "¿Qué tan bueno es el modelo encontrando todos los positivos?"
- "De todos los spam que existen, ¿cuántos detectamos?"

**Cuándo priorizar Recall**:
- Cuando los **falsos negativos son costosos**
- Ejemplo: Detección de fraude (no queremos perder ningún caso)
- Ejemplo: Detección de cáncer (mejor un falso positivo que perder un caso real)

### 4. F1-Score

**Definición**: Media armónica entre Precision y Recall.

```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

**Ejemplo**:
```
F1 = 2 * (0.889 * 0.80) / (0.889 + 0.80) = 0.842 = 84.2%
```

**Por qué media armónica**:
- Penaliza valores extremos
- Si Precision O Recall son bajos, F1 será bajo
- Balance entre ambas métricas

**Cuándo usar F1**:
- Dataset desbalanceado
- Cuando queremos balance entre Precision y Recall
- Métrica única para comparar modelos

**Variantes**:

**F-beta Score**: Permite dar más peso a Precision o Recall

```
F_β = (1 + β²) * (Precision * Recall) / (β² * Precision + Recall)
```

- **β < 1**: Más peso a Precision
- **β > 1**: Más peso a Recall
- **β = 1**: F1-Score (balance)
- **β = 2**: F2-Score (favorece Recall)

### 5. Specificity (Especificidad)

**Definición**: De todos los casos negativos reales, ¿cuántos identificamos correctamente?

```
Specificity = TN / (TN + FP)
```

**Ejemplo**:
```
Specificity = 45 / (45 + 5) = 0.90 = 90%
```

**Interpretación**:
- "¿Qué tan bueno es el modelo identificando negativos?"
- "De todos los emails normales, ¿cuántos identificamos correctamente?"

### 6. Matthews Correlation Coefficient (MCC)

**Definición**: Correlación entre predicciones y realidad.

```
MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
```

**Rango**: -1 (peor) a +1 (perfecto), 0 = random

**Ventaja**: Funciona bien incluso con clases muy desbalanceadas

### Comparación de Métricas

| Métrica | Fórmula | Mejor para | Rango |
|---------|---------|------------|-------|
| Accuracy | (TP+TN)/Total | Clases balanceadas | 0-1 |
| Precision | TP/(TP+FP) | Minimizar FP | 0-1 |
| Recall | TP/(TP+FN) | Minimizar FN | 0-1 |
| F1-Score | 2PR/(P+R) | Balance P y R | 0-1 |
| Specificity | TN/(TN+FP) | Identificar negativos | 0-1 |
| MCC | Correlación | Clases desbalanceadas | -1 a 1 |

## Métricas para Clasificación Multiclase

### Macro-Average

**Definición**: Calcular métrica para cada clase y promediar.

```python
Precision_macro = (Precision_clase1 + Precision_clase2 + ... + Precision_claseN) / N
```

**Ventaja**: Trata todas las clases por igual (bueno para clases pequeñas)

### Micro-Average

**Definición**: Agregar todos los TP, FP, FN y calcular métrica global.

```python
Precision_micro = Σ(TP_i) / (Σ(TP_i) + Σ(FP_i))
```

**Ventaja**: Refleja rendimiento en dataset completo (dominado por clases grandes)

### Weighted-Average

**Definición**: Promedio ponderado por el número de muestras de cada clase.

```python
Precision_weighted = Σ(Precision_i * n_samples_i) / Σ(n_samples_i)
```

**Ventaja**: Balance entre macro y micro

## Curva ROC y AUC

### ROC (Receiver Operating Characteristic)

**Definición**: Gráfica que muestra el trade-off entre True Positive Rate (Recall) y False Positive Rate.

```
TPR = Recall = TP / (TP + FN)
FPR = FP / (FP + TN)
```

**Cómo se construye**:
1. Variar el umbral de clasificación (de 0 a 1)
2. Para cada umbral, calcular TPR y FPR
3. Graficar TPR vs FPR

```
TPR │         ┌─────
    │        /
    │       /
    │      /  Modelo bueno
    │     /
    │    /
    │   /
    │  /_____ Modelo aleatorio
    └──────────────── FPR
```

### AUC (Area Under the Curve)

**Definición**: Área bajo la curva ROC.

**Interpretación**:
- **AUC = 1.0**: Clasificador perfecto
- **AUC = 0.9-1.0**: Excelente
- **AUC = 0.8-0.9**: Bueno
- **AUC = 0.7-0.8**: Aceptable
- **AUC = 0.5**: Random (línea diagonal)
- **AUC < 0.5**: Peor que random (modelo invertido)

**Ventaja**:
- Métrica única que resume el rendimiento
- Independiente del umbral de clasificación
- Útil para comparar modelos

## Precision-Recall Curve

**Definición**: Gráfica que muestra el trade-off entre Precision y Recall.

```
Precision │  ────┐
          │      │
          │      │  Mejor modelo
          │      └─────
          │   
          │    ───┐
          │       └──── Peor modelo
          └─────────────── Recall
```

**Cuándo usar en vez de ROC**:
- Datasets muy desbalanceados
- Cuando la clase positiva es rara pero importante
- La curva Precision-Recall es más informativa en estos casos

## Validación del Modelo

### 1. Holdout Validation

**Concepto**: Dividir datos en train/val/test.

```
Train (70%): Entrenar modelo
Validation (15%): Ajustar hiperparámetros
Test (15%): Evaluación final
```

**Ventaja**: Simple y rápido
**Desventaja**: Puede depender de cómo se dividieron los datos

### 2. K-Fold Cross-Validation

**Concepto**: Dividir datos en K partes (folds), entrenar K veces.

```
Fold 1: [TEST][TRAIN][TRAIN][TRAIN][TRAIN]
Fold 2: [TRAIN][TEST][TRAIN][TRAIN][TRAIN]
Fold 3: [TRAIN][TRAIN][TEST][TRAIN][TRAIN]
Fold 4: [TRAIN][TRAIN][TRAIN][TEST][TRAIN]
Fold 5: [TRAIN][TRAIN][TRAIN][TRAIN][TEST]
```

**Proceso**:
1. Dividir datos en K folds
2. Para cada fold:
   - Entrenar con K-1 folds
   - Validar con el fold restante
3. Promediar resultados

**Ventajas**:
- Usa todos los datos para entrenar y validar
- Estimación más robusta del rendimiento
- Reduce varianza de la evaluación

**K típico**: 5 o 10

**Variante - Stratified K-Fold**: Mantiene proporción de clases en cada fold

### 3. Leave-One-Out Cross-Validation (LOOCV)

**Concepto**: K-Fold donde K = número de muestras.

**Ventajas**: Máximo uso de datos
**Desventajas**: Muy costoso computacionalmente

## Estrategias de Evaluación Según el Problema

### Datasets Balanceados
```
Métrica principal: Accuracy
Secundarias: Precision, Recall, F1
```

### Datasets Desbalanceados
```
Métricas principales: F1-Score, AUC, MCC
Evitar: Accuracy (puede ser engañoso)
```

### Costo Asimétrico de Errores

**Caso 1: FP muy costosos** (ej: diagnóstico que requiere cirugía)
```
Optimizar: Precision
Métrica secundaria: Specificity
```

**Caso 2: FN muy costosos** (ej: detección de fraude)
```
Optimizar: Recall
Métrica secundaria: F2-Score
```

**Caso 3: Balance** (ej: clasificación general)
```
Optimizar: F1-Score
Métricas secundarias: Precision, Recall
```

### Problemas Multiclase
```
Métrica principal: Macro F1-Score
Secundarias: Micro F1, Weighted F1
Análisis: Matriz de confusión completa
```

## Interpretación de Resultados

### Análisis de la Matriz de Confusión

**Patrón 1: Confusión entre clases específicas**
```
              Pred
        Gato  Perro  Pájaro
Real Gato  80    15      5
     Perro 12    85      3
   Pájaro  1     1     98
```
→ Problema: El modelo confunde gatos y perros
→ Solución: Añadir más datos diferenciadores entre estas clases

**Patrón 2: Clase difícil de detectar**
```
              Pred
        A    B    C
Real A  90   5    5
     B  10  85    5
     C  30  30   40  ← Clase problemática
```
→ Problema: Clase C tiene baja recall
→ Solución: Más datos de clase C, balanceo, o repensar features

### Métricas en Conjunto

**Escenario 1**:
```
Accuracy: 95%
Precision: 60%
Recall: 30%
```
→ Interpretación: Dataset desbalanceado, modelo conservador (predice poco la clase positiva)

**Escenario 2**:
```
Accuracy: 70%
Precision: 90%
Recall: 85%
F1: 87%
```
→ Interpretación: Buen balance, modelo confiable para la clase positiva

**Escenario 3**:
```
Precision: 95%
Recall: 40%
F1: 56%
```
→ Interpretación: Modelo muy conservador, alta confianza pero detecta pocos casos

## Mejora Iterativa Basada en Métricas

### Proceso de Optimización

1. **Establecer baseline**: Primera versión del modelo
2. **Identificar problema**: Analizar matriz de confusión y métricas
3. **Hipótesis**: ¿Por qué el modelo falla?
4. **Intervención**: Cambio específico (datos, arquitectura, hiperparámetros)
5. **Medir**: Evaluar con mismas métricas
6. **Comparar**: ¿Mejoró la métrica objetivo?
7. **Iterar**: Repetir proceso

### Ejemplo de Iteración

```
Baseline:
  Precision: 70%, Recall: 50%, F1: 58%
  
Problema identificado: Recall bajo
Hipótesis: Umbral de clasificación muy alto
Intervención: Reducir umbral de 0.5 a 0.3

Resultado:
  Precision: 65%, Recall: 75%, F1: 70%
  
Decisión: ✓ Mejora aceptable en F1
```

## Checklist de Evaluación

### Antes de Evaluar
- [ ] Datos de test completamente separados (nunca vistos)
- [ ] Test set representativo del problema real
- [ ] Mismo preprocesamiento que en entrenamiento
- [ ] Clases balanceadas en test (o estratificadas)

### Durante la Evaluación
- [ ] Calcular matriz de confusión
- [ ] Calcular múltiples métricas (no solo accuracy)
- [ ] Analizar errores por clase
- [ ] Visualizar predicciones incorrectas
- [ ] Considerar contexto del problema

### Después de Evaluar
- [ ] Interpretar métricas en contexto del negocio
- [ ] Identificar patrones de error
- [ ] Documentar rendimiento
- [ ] Comparar con baseline/modelos anteriores
- [ ] Decidir si el modelo es aceptable para producción

## Errores Comunes

### 1. Usar solo Accuracy
❌ **Error**: "Mi modelo tiene 98% accuracy, es excelente"
✓ **Correcto**: Verificar si el dataset está balanceado y revisar otras métricas

### 2. Evaluar en datos de entrenamiento
❌ **Error**: Medir rendimiento en datos usados para entrenar
✓ **Correcto**: Siempre evaluar en test set separado

### 3. Data leakage
❌ **Error**: Información del test set filtra al entrenamiento
✓ **Correcto**: Separar datos ANTES de cualquier preprocesamiento

### 4. Ignorar el contexto
❌ **Error**: "F1=0.9 es bueno" (sin considerar el problema)
✓ **Correcto**: Evaluar si las métricas son adecuadas para el caso de uso

### 5. No validar regularmente
❌ **Error**: Evaluar solo al final
✓ **Correcto**: Monitorear métricas durante todo el entrenamiento

## Resumen

### Conceptos Clave

1. **Matriz de Confusión**: Herramienta fundamental para entender rendimiento
2. **Accuracy**: Útil solo en datasets balanceados
3. **Precision**: Importante cuando FP son costosos
4. **Recall**: Importante cuando FN son costosos
5. **F1-Score**: Balance entre Precision y Recall
6. **Validación**: K-Fold para estimación robusta
7. **Contexto**: Las mejores métricas dependen del problema

### Flujo de Trabajo

```
1. Entrenar modelo
2. Generar predicciones en test set
3. Calcular matriz de confusión
4. Calcular métricas relevantes
5. Analizar errores
6. Iterar y mejorar
7. Validar con K-Fold
8. Decisión: ¿Es adecuado para producción?
```

## Próximos Pasos

En la práctica implementaremos:
- Cálculo de matriz de confusión
- Todas las métricas discutidas
- Visualizaciones de métricas
- Validación cruzada
- Análisis de errores
- Comparación de modelos

¡Ahora es momento de poner esto en práctica! 🎯
