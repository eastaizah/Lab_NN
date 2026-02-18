"""
Métricas de Evaluación y Matriz de Confusión
Implementación desde cero de métricas de clasificación
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional


class MatrizConfusion:
    """
    Clase para calcular y visualizar matriz de confusión
    """
    
    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray, labels: Optional[List[str]] = None):
        """
        Inicializar matriz de confusión
        
        Args:
            y_true: Etiquetas verdaderas
            y_pred: Etiquetas predichas
            labels: Nombres de las clases (opcional)
        """
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        
        # Obtener clases únicas
        self.classes = np.unique(np.concatenate([self.y_true, self.y_pred]))
        self.n_classes = len(self.classes)
        
        # Nombres de clases
        if labels is None:
            self.labels = [f"Clase {i}" for i in self.classes]
        else:
            self.labels = labels
        
        # Calcular matriz
        self.matriz = self._calcular_matriz()
    
    def _calcular_matriz(self) -> np.ndarray:
        """
        Calcular matriz de confusión desde cero
        
        Returns:
            Matriz de confusión de forma (n_classes, n_classes)
        """
        matriz = np.zeros((self.n_classes, self.n_classes), dtype=int)
        
        # Crear mapeo de clase a índice
        class_to_idx = {clase: idx for idx, clase in enumerate(self.classes)}
        
        # Llenar matriz
        for true_label, pred_label in zip(self.y_true, self.y_pred):
            true_idx = class_to_idx[true_label]
            pred_idx = class_to_idx[pred_label]
            matriz[true_idx, pred_idx] += 1
        
        return matriz
    
    def visualizar(self, normalizar: bool = False, figsize: Tuple[int, int] = (8, 6)):
        """
        Visualizar matriz de confusión
        
        Args:
            normalizar: Si True, muestra porcentajes en lugar de conteos
            figsize: Tamaño de la figura
        """
        plt.figure(figsize=figsize)
        
        if normalizar:
            # Normalizar por filas (total de cada clase verdadera)
            matriz_norm = self.matriz.astype('float') / self.matriz.sum(axis=1)[:, np.newaxis]
            sns.heatmap(matriz_norm, annot=True, fmt='.2%', cmap='Blues',
                       xticklabels=self.labels, yticklabels=self.labels)
            plt.title('Matriz de Confusión (Normalizada)')
        else:
            sns.heatmap(self.matriz, annot=True, fmt='d', cmap='Blues',
                       xticklabels=self.labels, yticklabels=self.labels)
            plt.title('Matriz de Confusión')
        
        plt.ylabel('Etiqueta Verdadera')
        plt.xlabel('Etiqueta Predicha')
        plt.tight_layout()
        plt.show()
    
    def obtener_tp_fp_fn_tn(self, clase_positiva: int = 1) -> Tuple[int, int, int, int]:
        """
        Obtener TP, FP, FN, TN para clasificación binaria
        
        Args:
            clase_positiva: Índice de la clase considerada positiva
        
        Returns:
            (TP, FP, FN, TN)
        """
        if self.n_classes != 2:
            raise ValueError("Esta función solo funciona para clasificación binaria")
        
        # Para binario, asumimos que clase_positiva es el índice 1
        TP = self.matriz[1, 1]
        FN = self.matriz[1, 0]
        FP = self.matriz[0, 1]
        TN = self.matriz[0, 0]
        
        return TP, FP, FN, TN


class MetricasClasificacion:
    """
    Clase para calcular métricas de clasificación
    """
    
    @staticmethod
    def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calcular Accuracy (Exactitud)
        
        Accuracy = (TP + TN) / Total
        """
        return np.mean(y_true == y_pred)
    
    @staticmethod
    def precision(tp: int, fp: int) -> float:
        """
        Calcular Precision (Precisión)
        
        Precision = TP / (TP + FP)
        """
        if tp + fp == 0:
            return 0.0
        return tp / (tp + fp)
    
    @staticmethod
    def recall(tp: int, fn: int) -> float:
        """
        Calcular Recall (Sensibilidad, True Positive Rate)
        
        Recall = TP / (TP + FN)
        """
        if tp + fn == 0:
            return 0.0
        return tp / (tp + fn)
    
    @staticmethod
    def f1_score(precision: float, recall: float) -> float:
        """
        Calcular F1-Score
        
        F1 = 2 * (Precision * Recall) / (Precision + Recall)
        """
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    
    @staticmethod
    def fbeta_score(precision: float, recall: float, beta: float = 1.0) -> float:
        """
        Calcular F-beta Score
        
        F_β = (1 + β²) * (Precision * Recall) / (β² * Precision + Recall)
        
        Args:
            precision: Precision
            recall: Recall
            beta: Peso de recall vs precision (beta > 1 favorece recall)
        """
        if precision + recall == 0:
            return 0.0
        return (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)
    
    @staticmethod
    def specificity(tn: int, fp: int) -> float:
        """
        Calcular Specificity (Especificidad)
        
        Specificity = TN / (TN + FP)
        """
        if tn + fp == 0:
            return 0.0
        return tn / (tn + fp)
    
    @staticmethod
    def mcc(tp: int, tn: int, fp: int, fn: int) -> float:
        """
        Calcular Matthews Correlation Coefficient
        
        MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
        
        Rango: -1 a +1, donde 1 es predicción perfecta
        """
        numerador = tp * tn - fp * fn
        denominador = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        
        if denominador == 0:
            return 0.0
        
        return numerador / denominador
    
    @staticmethod
    def reporte_clasificacion_binaria(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Generar reporte completo para clasificación binaria
        
        Returns:
            Diccionario con todas las métricas
        """
        matriz = MatrizConfusion(y_true, y_pred)
        tp, fp, fn, tn = matriz.obtener_tp_fp_fn_tn()
        
        precision = MetricasClasificacion.precision(tp, fp)
        recall = MetricasClasificacion.recall(tp, fn)
        
        return {
            'accuracy': MetricasClasificacion.accuracy(y_true, y_pred),
            'precision': precision,
            'recall': recall,
            'f1_score': MetricasClasificacion.f1_score(precision, recall),
            'specificity': MetricasClasificacion.specificity(tn, fp),
            'mcc': MetricasClasificacion.mcc(tp, tn, fp, fn),
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn
        }
    
    @staticmethod
    def metricas_multiclase(y_true: np.ndarray, y_pred: np.ndarray, 
                           average: str = 'macro') -> Dict:
        """
        Calcular métricas para clasificación multiclase
        
        Args:
            y_true: Etiquetas verdaderas
            y_pred: Etiquetas predichas
            average: 'macro', 'micro', o 'weighted'
        
        Returns:
            Diccionario con métricas
        """
        matriz_conf = MatrizConfusion(y_true, y_pred)
        classes = matriz_conf.classes
        n_classes = len(classes)
        
        # Calcular métricas por clase
        precisions = []
        recalls = []
        f1s = []
        supports = []  # Número de muestras por clase
        
        for i, clase in enumerate(classes):
            # TP: diagonal
            tp = matriz_conf.matriz[i, i]
            
            # FP: suma de columna i excepto diagonal
            fp = np.sum(matriz_conf.matriz[:, i]) - tp
            
            # FN: suma de fila i excepto diagonal
            fn = np.sum(matriz_conf.matriz[i, :]) - tp
            
            # Support: total de muestras de esta clase
            support = np.sum(y_true == clase)
            supports.append(support)
            
            # Calcular métricas
            prec = MetricasClasificacion.precision(tp, fp)
            rec = MetricasClasificacion.recall(tp, fn)
            f1 = MetricasClasificacion.f1_score(prec, rec)
            
            precisions.append(prec)
            recalls.append(rec)
            f1s.append(f1)
        
        # Calcular promedio según el tipo
        if average == 'macro':
            # Promedio simple
            precision_avg = np.mean(precisions)
            recall_avg = np.mean(recalls)
            f1_avg = np.mean(f1s)
        
        elif average == 'micro':
            # Calcular globalmente
            tp_total = np.sum([matriz_conf.matriz[i, i] for i in range(n_classes)])
            fp_total = np.sum(matriz_conf.matriz) - tp_total
            fn_total = fp_total  # Para micro, FP = FN
            
            precision_avg = MetricasClasificacion.precision(tp_total, fp_total)
            recall_avg = MetricasClasificacion.recall(tp_total, fn_total)
            f1_avg = MetricasClasificacion.f1_score(precision_avg, recall_avg)
        
        elif average == 'weighted':
            # Promedio ponderado por support
            total_support = sum(supports)
            precision_avg = sum(p * s for p, s in zip(precisions, supports)) / total_support
            recall_avg = sum(r * s for r, s in zip(recalls, supports)) / total_support
            f1_avg = sum(f * s for f, s in zip(f1s, supports)) / total_support
        
        else:
            raise ValueError("average debe ser 'macro', 'micro', o 'weighted'")
        
        return {
            'accuracy': MetricasClasificacion.accuracy(y_true, y_pred),
            f'precision_{average}': precision_avg,
            f'recall_{average}': recall_avg,
            f'f1_score_{average}': f1_avg,
            'precision_per_class': precisions,
            'recall_per_class': recalls,
            'f1_per_class': f1s,
            'support_per_class': supports
        }


class ValidacionCruzada:
    """
    Implementación de K-Fold Cross-Validation
    """
    
    @staticmethod
    def k_fold_split(X: np.ndarray, y: np.ndarray, k: int = 5, 
                     shuffle: bool = True, random_state: Optional[int] = None):
        """
        Dividir datos en K folds
        
        Args:
            X: Features
            y: Labels
            k: Número de folds
            shuffle: Si True, mezcla los datos antes de dividir
            random_state: Semilla para reproducibilidad
        
        Yields:
            (X_train, X_val, y_train, y_val) para cada fold
        """
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        if shuffle:
            if random_state is not None:
                np.random.seed(random_state)
            np.random.shuffle(indices)
        
        fold_sizes = np.full(k, n_samples // k, dtype=int)
        fold_sizes[:n_samples % k] += 1
        
        current = 0
        for fold_size in fold_sizes:
            # Índices del fold de validación
            val_indices = indices[current:current + fold_size]
            
            # Índices del fold de entrenamiento
            train_indices = np.concatenate([
                indices[:current],
                indices[current + fold_size:]
            ])
            
            # Dividir datos
            X_train, X_val = X[train_indices], X[val_indices]
            y_train, y_val = y[train_indices], y[val_indices]
            
            yield X_train, X_val, y_train, y_val
            
            current += fold_size
    
    @staticmethod
    def evaluar_con_k_fold(modelo, X: np.ndarray, y: np.ndarray, 
                          k: int = 5, metrica_fn = None) -> Dict:
        """
        Evaluar modelo usando K-Fold Cross-Validation
        
        Args:
            modelo: Modelo con métodos fit() y predict()
            X: Features
            y: Labels
            k: Número de folds
            metrica_fn: Función para calcular métrica (default: accuracy)
        
        Returns:
            Diccionario con métricas por fold y promedio
        """
        if metrica_fn is None:
            metrica_fn = MetricasClasificacion.accuracy
        
        scores = []
        
        for fold_idx, (X_train, X_val, y_train, y_val) in enumerate(
            ValidacionCruzada.k_fold_split(X, y, k=k, shuffle=True, random_state=42)
        ):
            # Entrenar modelo
            modelo.fit(X_train, y_train)
            
            # Predecir
            y_pred = modelo.predict(X_val)
            
            # Calcular métrica
            score = metrica_fn(y_val, y_pred)
            scores.append(score)
            
            print(f"Fold {fold_idx + 1}/{k}: Score = {score:.4f}")
        
        return {
            'scores': scores,
            'mean': np.mean(scores),
            'std': np.std(scores),
            'min': np.min(scores),
            'max': np.max(scores)
        }


def ejemplo_basico():
    """
    Ejemplo básico de uso de métricas
    """
    print("=" * 60)
    print("EJEMPLO: Clasificación Binaria - Detector de Spam")
    print("=" * 60)
    
    # Datos de ejemplo: 0 = No Spam, 1 = Spam
    y_true = np.array([1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 
                       0, 1, 1, 0, 0, 0, 1, 0, 1, 0])
    y_pred = np.array([1, 1, 0, 1, 0, 0, 0, 0, 0, 1,
                       1, 1, 1, 0, 0, 0, 1, 0, 0, 0])
    
    # Calcular matriz de confusión
    matriz = MatrizConfusion(y_true, y_pred, labels=['No Spam', 'Spam'])
    
    print("\nMatriz de Confusión:")
    print(matriz.matriz)
    
    # Obtener TP, FP, FN, TN
    tp, fp, fn, tn = matriz.obtener_tp_fp_fn_tn()
    print(f"\nTP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
    
    # Calcular métricas
    reporte = MetricasClasificacion.reporte_clasificacion_binaria(y_true, y_pred)
    
    print("\n" + "=" * 40)
    print("MÉTRICAS DE EVALUACIÓN")
    print("=" * 40)
    print(f"Accuracy:    {reporte['accuracy']:.3f}")
    print(f"Precision:   {reporte['precision']:.3f}")
    print(f"Recall:      {reporte['recall']:.3f}")
    print(f"F1-Score:    {reporte['f1_score']:.3f}")
    print(f"Specificity: {reporte['specificity']:.3f}")
    print(f"MCC:         {reporte['mcc']:.3f}")
    
    print("\n" + "=" * 40)
    print("INTERPRETACIÓN")
    print("=" * 40)
    print(f"• De {tp + fn} spam reales, detectamos {tp} ({reporte['recall']:.1%})")
    print(f"• De {tp + fp} predicciones de spam, {tp} fueron correctas ({reporte['precision']:.1%})")
    print(f"• {fp} emails normales fueron marcados como spam (Falsos Positivos)")
    print(f"• {fn} spam pasaron sin detectar (Falsos Negativos)")


def ejemplo_multiclase():
    """
    Ejemplo de clasificación multiclase
    """
    print("\n\n" + "=" * 60)
    print("EJEMPLO: Clasificación Multiclase - Reconocimiento de Animales")
    print("=" * 60)
    
    # 0 = Gato, 1 = Perro, 2 = Pájaro
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0,
                       1, 2, 0, 1, 2, 0, 1, 2, 0, 1])
    y_pred = np.array([0, 1, 2, 0, 1, 1, 0, 1, 2, 0,
                       0, 2, 0, 1, 2, 2, 1, 2, 0, 1])
    
    # Calcular matriz de confusión
    matriz = MatrizConfusion(y_true, y_pred, labels=['Gato', 'Perro', 'Pájaro'])
    
    print("\nMatriz de Confusión:")
    print(matriz.matriz)
    
    # Calcular métricas con diferentes promedios
    for avg_type in ['macro', 'micro', 'weighted']:
        metricas = MetricasClasificacion.metricas_multiclase(y_true, y_pred, average=avg_type)
        
        print(f"\n{avg_type.upper()} Average:")
        print(f"  Precision: {metricas[f'precision_{avg_type}']:.3f}")
        print(f"  Recall:    {metricas[f'recall_{avg_type}']:.3f}")
        print(f"  F1-Score:  {metricas[f'f1_score_{avg_type}']:.3f}")


if __name__ == "__main__":
    # Ejecutar ejemplos
    ejemplo_basico()
    ejemplo_multiclase()
    
    print("\n" + "=" * 60)
    print("Para más ejemplos, ver practica.ipynb")
    print("=" * 60)
