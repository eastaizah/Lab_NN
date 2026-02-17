# Lab 05: Backpropagation

## Objetivos
1. Comprender la regla de la cadena
2. Entender grafos computacionales
3. Implementar backpropagation desde cero
4. Verificar gradientes numéricamente
5. Entrenar una red neuronal completa

## Estructura
```
Lab05_Backpropagation/
├── README.md
├── teoria.md
├── practica.ipynb
└── codigo/
    └── backprop.py
```

## Conceptos Clave

### Regla de la Cadena
Fundamento matemático de backpropagation:
```
∂z/∂x = (∂z/∂y) * (∂y/∂x)
```

### Forward y Backward Pass
1. **Forward**: Calcular predicciones (guardar valores intermedios)
2. **Backward**: Calcular gradientes (usar valores guardados)

### Verificación de Gradientes
Siempre verifica tu implementación:
```python
difference = |analytical - numerical| / |analytical + numerical|
if difference < 1e-7: ✓ Correcto
```

## Práctica

### Ejecutar código:
```bash
cd codigo/
python backprop.py
```

### Notebook interactivo:
```bash
jupyter notebook practica.ipynb
```

## Ejercicios

1. Implementar backpropagation para 3 capas
2. Visualizar gradientes en cada capa
3. Comparar con gradientes numéricos
4. Entrenar en problema no lineal

## Verificación
- [ ] Entiendo la regla de la cadena
- [ ] Puedo implementar forward/backward pass
- [ ] Sé verificar gradientes numéricamente
- [ ] Puedo entrenar una red completa

## Próximo Lab
**Lab 06: Entrenamiento** - Loops de entrenamiento completos, validación, early stopping.

---
**¡Backpropagation es la magia del deep learning! ✨**
