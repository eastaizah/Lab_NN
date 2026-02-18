# Lab 07: Métricas de Evaluación y Matriz de Confusión

## Objetivos

1. Entender la matriz de confusión y sus componentes
2. Calcular e interpretar métricas de clasificación (Precision, Recall, F1-Score, Accuracy)
3. Utilizar métricas apropiadas según el problema
4. Implementar validación cruzada (K-Fold)
5. Analizar errores del modelo para mejorarlo
6. Trabajar con datasets balanceados y desbalanceados

## Estructura

```
Lab07_Metricas_Evaluacion/
├── README.md
├── teoria.md
├── practica.ipynb
└── codigo/
    └── metricas.py
```

## Conceptos Clave

### Matriz de Confusión

```
                Predicción
             Positivo  Negativo
          ┌──────────┬──────────┐
Real   P  │    TP    │    FN    │
       N  │    FP    │    TN    │
          └──────────┴──────────┘
```

- **TP (True Positives)**: Predicciones positivas correctas
- **TN (True Negatives)**: Predicciones negativas correctas
- **FP (False Positives)**: Falsos positivos (Error Tipo I)
- **FN (False Negatives)**: Falsos negativos (Error Tipo II)

### Métricas Principales

**Accuracy (Exactitud)**:
```
Accuracy = (TP + TN) / Total
```
→ Proporción de predicciones correctas

**Precision (Precisión)**:
```
Precision = TP / (TP + FP)
```
→ De las predicciones positivas, ¿cuántas fueron correctas?

**Recall (Sensibilidad)**:
```
Recall = TP / (TP + FN)
```
→ De los casos positivos reales, ¿cuántos detectamos?

**F1-Score**:
```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```
→ Media armónica entre Precision y Recall

### Cuándo Usar Cada Métrica

| Situación | Métrica Principal | Razón |
|-----------|------------------|-------|
| Dataset balanceado | Accuracy | Clases representadas equitativamente |
| Dataset desbalanceado | F1-Score | Evita sesgo hacia clase mayoritaria |
| FP muy costosos | Precision | Ej: diagnóstico que requiere cirugía |
| FN muy costosos | Recall | Ej: detección de fraude o enfermedades |
| Balance P y R | F1-Score | Mejor métrica única general |

### Validación Cruzada

**K-Fold Cross-Validation**: Divide datos en K partes, entrena K veces

```
Fold 1: [TEST][TRAIN][TRAIN][TRAIN][TRAIN]
Fold 2: [TRAIN][TEST][TRAIN][TRAIN][TRAIN]
...
Fold K: [TRAIN][TRAIN][TRAIN][TRAIN][TEST]

Métrica final = promedio de K evaluaciones
```

**Ventajas**:
- Estimación más robusta
- Mejor uso de datos limitados
- Reduce varianza de la evaluación

## Práctica

### Ejecutar código:
```bash
cd codigo/
python metricas.py
```

### Notebook:
```bash
jupyter notebook practica.ipynb
```

## Ejercicios

1. **Básico**: Calcular métricas manualmente desde matriz de confusión
2. **Intermedio**: Implementar K-Fold cross-validation
3. **Avanzado**: Comparar modelos en dataset desbalanceado
4. **Desafío**: Optimizar umbral de clasificación para maximizar F1

## Casos de Uso Reales

### Ejemplo 1: Detección de Spam
```
FP: Email importante marcado como spam (muy malo)
FN: Spam que llega a inbox (tolerable)
→ Optimizar: PRECISION (minimizar FP)
```

### Ejemplo 2: Detección de Fraude
```
FP: Transacción legítima bloqueada (tolerable)
FN: Fraude no detectado (muy malo)
→ Optimizar: RECALL (minimizar FN)
```

### Ejemplo 3: Clasificación General
```
Ambos errores igualmente importantes
→ Optimizar: F1-SCORE (balance)
```

## Debugging

**Accuracy alto pero modelo malo**:
- Verificar si dataset está desbalanceado
- Revisar otras métricas (Precision, Recall, F1)

**Precision muy alta pero Recall baja**:
- Modelo demasiado conservador
- Reducir umbral de clasificación

**Recall muy alto pero Precision baja**:
- Modelo demasiado liberal
- Aumentar umbral de clasificación

**Todas las métricas bajas**:
- Problema con los datos o el modelo
- Revisar preprocesamiento
- Considerar modelo más complejo

## Checklist de Aprendizaje

- [ ] Entiendo la matriz de confusión y sus componentes
- [ ] Puedo calcular Accuracy, Precision, Recall y F1
- [ ] Sé cuándo usar cada métrica
- [ ] Puedo implementar validación cruzada
- [ ] Sé interpretar resultados en contexto del problema
- [ ] Puedo identificar y solucionar problemas comunes

## Relación con Otros Labs

**De Lab 06 (Entrenamiento)**:
- Usamos modelos entrenados
- Aplicamos división train/val/test
- Monitoreamos métricas durante entrenamiento

**Hacia Lab 08 (Frameworks)**:
- PyTorch y TensorFlow tienen métricas built-in
- Mismo principio, implementación más fácil
- Automatización de validación cruzada

## Recursos Adicionales

- Scikit-learn: `sklearn.metrics` (confusion_matrix, classification_report)
- Visualización: seaborn heatmap para matriz de confusión
- ROC curves y AUC para evaluación avanzada

## Verificación

Al finalizar este lab deberías poder:
1. ✓ Generar y interpretar matriz de confusión
2. ✓ Calcular métricas de clasificación desde cero
3. ✓ Elegir métricas apropiadas para tu problema
4. ✓ Implementar K-Fold cross-validation
5. ✓ Analizar y mejorar modelos basándote en métricas

## Próximo Lab

**Lab 08**: Frameworks de Deep Learning (PyTorch, TensorFlow)
- Implementación eficiente de todo lo aprendido
- Métricas y validación automatizadas

---

**¡La evaluación correcta es tan importante como el entrenamiento! 📊**
