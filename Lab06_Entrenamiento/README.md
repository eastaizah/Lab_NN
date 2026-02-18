# Lab 06: Entrenamiento de Redes Neuronales

## Objetivos
1. Implementar loop de entrenamiento completo
2. Dividir datos en train/val/test
3. Implementar early stopping
4. Monitorear métricas durante entrenamiento
5. Detectar y prevenir overfitting

## Estructura
```
Lab06_Entrenamiento/
├── README.md
├── teoria.md
├── practica.ipynb
└── codigo/
    └── entrenamiento.py
```

## Conceptos Clave

### Época vs Iteración vs Batch
- **Época**: Pase completo por todos los datos
- **Batch**: Subconjunto de datos procesados juntos
- **Iteración**: Un paso de actualización (procesar un batch)

### División de Datos
```
Train (70%): Entrenar el modelo
Validation (15%): Ajustar hiperparámetros
Test (15%): Evaluación final
```

### Early Stopping
Detener cuando validación deja de mejorar:
```python
if val_loss no mejora en 10 épocas:
    detener entrenamiento
```

## Práctica

### Ejecutar:
```bash
cd codigo/
python entrenamiento.py
```

### Notebook:
```bash
jupyter notebook practica.ipynb
```

## Hiperparámetros Recomendados

Para empezar:
```
Learning rate: 0.001 - 0.01
Batch size: 32
Epochs: 100
Optimizer: Adam (o SGD)
Hidden layers: 2
```

## Ejercicios

1. Entrenar en MNIST
2. Implementar learning rate decay
3. Comparar diferentes batch sizes
4. Experimentar con arquitecturas

## Debugging

**Pérdida no baja**:
- Verificar learning rate
- Normalizar datos
- Verificar gradientes

**Overfitting**:
- Añadir regularización
- Early stopping
- Más datos

## Verificación
- [ ] Puedo entrenar una red completa
- [ ] Entiendo épocas, batches e iteraciones
- [ ] Sé detectar overfitting
- [ ] Puedo implementar early stopping

## Próximo Lab
**Lab 07**: Métricas de Evaluación y Matriz de Confusión

---
**¡El entrenamiento es donde todo cobra vida! 🚀**
