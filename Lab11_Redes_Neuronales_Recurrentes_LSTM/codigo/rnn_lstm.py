"""
Implementación de Redes Neuronales Recurrentes (RNN) y LSTM desde Cero
Laboratorio 10: Redes Neuronales Recurrentes y LSTM
=====================================================

Este módulo implementa:
- RNN vanilla desde cero con NumPy
- LSTM con todas las puertas (forget, input, output)
- GRU (Gated Recurrent Unit)
- Funciones auxiliares para procesamiento de secuencias
- Ejemplos con PyTorch para uso práctico
- Clasificación de texto y predicción de series temporales
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Dict
import warnings

# Importar PyTorch de manera opcional
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("Advertencia: PyTorch no está disponible. "
          "Las funciones de entrenamiento con PyTorch estarán deshabilitadas.")


# =============================================================================
# PARTE 1: RNN VANILLA DESDE CERO
# =============================================================================

class RNNCellNumPy:
    """
    Celda RNN vanilla implementada desde cero con NumPy.
    
    Ecuaciones:
        h_t = tanh(W_hh @ h_{t-1} + W_xh @ x_t + b_h)
    """
    
    def __init__(self, input_size: int, hidden_size: int, seed: int = 42):
        """
        Args:
            input_size: Dimensión de la entrada
            hidden_size: Dimensión del estado oculto
            seed: Semilla para reproducibilidad
        """
        np.random.seed(seed)
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Inicialización Xavier/Glorot para estabilidad
        limit_xh = np.sqrt(6 / (input_size + hidden_size))
        limit_hh = np.sqrt(6 / (hidden_size + hidden_size))
        
        # Pesos: input-to-hidden
        self.W_xh = np.random.uniform(-limit_xh, limit_xh, 
                                      (hidden_size, input_size))
        
        # Pesos: hidden-to-hidden (recurrentes)
        self.W_hh = np.random.uniform(-limit_hh, limit_hh,
                                      (hidden_size, hidden_size))
        
        # Sesgo
        self.b_h = np.zeros((hidden_size, 1))
        
        # Cache para backpropagation
        self.cache = {}
    
    def forward(self, x_t: np.ndarray, h_prev: np.ndarray) -> np.ndarray:
        """
        Forward pass de una celda RNN.
        
        Args:
            x_t: Entrada en tiempo t, shape (input_size, batch_size)
            h_prev: Estado oculto anterior, shape (hidden_size, batch_size)
        
        Returns:
            h_t: Nuevo estado oculto, shape (hidden_size, batch_size)
        """
        # Asegurar que x_t es 2D
        if x_t.ndim == 1:
            x_t = x_t.reshape(-1, 1)
        if h_prev.ndim == 1:
            h_prev = h_prev.reshape(-1, 1)
        
        # h_t = tanh(W_hh @ h_{t-1} + W_xh @ x_t + b_h)
        z = self.W_hh @ h_prev + self.W_xh @ x_t + self.b_h
        h_t = np.tanh(z)
        
        # Guardar para backward
        self.cache['x_t'] = x_t
        self.cache['h_prev'] = h_prev
        self.cache['h_t'] = h_t
        self.cache['z'] = z
        
        return h_t
    
    def backward(self, dh_t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Backward pass de una celda RNN.
        
        Args:
            dh_t: Gradiente del hidden state, shape (hidden_size, batch_size)
        
        Returns:
            dh_prev: Gradiente para h_{t-1}
            dx_t: Gradiente para x_t
        """
        x_t = self.cache['x_t']
        h_prev = self.cache['h_prev']
        z = self.cache['z']
        
        # Derivada de tanh: 1 - tanh^2(z)
        dtanh = 1 - np.tanh(z) ** 2
        
        # dz = dh_t * dtanh
        dz = dh_t * dtanh
        
        # Gradientes de los pesos
        self.dW_xh = dz @ x_t.T
        self.dW_hh = dz @ h_prev.T
        self.db_h = np.sum(dz, axis=1, keepdims=True)
        
        # Gradientes para la entrada y hidden state anterior
        dx_t = self.W_xh.T @ dz
        dh_prev = self.W_hh.T @ dz
        
        return dh_prev, dx_t


class RNNNumPy:
    """
    RNN completa que procesa secuencias completas.
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int, 
                 seed: int = 42):
        """
        Args:
            input_size: Dimensión de entrada
            hidden_size: Dimensión del estado oculto
            output_size: Dimensión de salida
            seed: Semilla para reproducibilidad
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Celda RNN
        self.cell = RNNCellNumPy(input_size, hidden_size, seed)
        
        # Capa de salida
        np.random.seed(seed)
        limit = np.sqrt(6 / (hidden_size + output_size))
        self.W_hy = np.random.uniform(-limit, limit, (output_size, hidden_size))
        self.b_y = np.zeros((output_size, 1))
        
        # Cache para secuencias
        self.hidden_states = []
    
    def forward(self, X: np.ndarray, h_0: Optional[np.ndarray] = None) -> Tuple[np.ndarray, List]:
        """
        Forward pass en una secuencia completa.
        
        Args:
            X: Secuencia de entrada, shape (seq_len, input_size, batch_size)
            h_0: Estado oculto inicial, shape (hidden_size, batch_size)
        
        Returns:
            outputs: Salidas en cada paso, shape (seq_len, output_size, batch_size)
            hidden_states: Lista de hidden states
        """
        seq_len = X.shape[0]
        batch_size = X.shape[2] if X.ndim == 3 else 1
        
        # Inicializar h_0 si no se proporciona
        if h_0 is None:
            h_0 = np.zeros((self.hidden_size, batch_size))
        
        h_t = h_0
        outputs = []
        self.hidden_states = [h_0]
        
        # Procesar secuencia
        for t in range(seq_len):
            x_t = X[t]
            if x_t.ndim == 1:
                x_t = x_t.reshape(-1, 1)
            
            # RNN step
            h_t = self.cell.forward(x_t, h_t)
            self.hidden_states.append(h_t)
            
            # Output
            y_t = self.W_hy @ h_t + self.b_y
            outputs.append(y_t)
        
        outputs = np.array(outputs)
        return outputs, self.hidden_states[1:]  # Excluir h_0
    
    def predict(self, X: np.ndarray, h_0: Optional[np.ndarray] = None) -> np.ndarray:
        """Predicción sin guardar cache."""
        outputs, _ = self.forward(X, h_0)
        return outputs


# =============================================================================
# PARTE 2: LSTM DESDE CERO
# =============================================================================

class LSTMCellNumPy:
    """
    Celda LSTM implementada desde cero con NumPy.
    
    Incluye las tres puertas: forget, input, output.
    
    Ecuaciones:
        f_t = sigmoid(W_f @ [h_{t-1}, x_t] + b_f)  # Forget gate
        i_t = sigmoid(W_i @ [h_{t-1}, x_t] + b_i)  # Input gate
        C_tilde = tanh(W_C @ [h_{t-1}, x_t] + b_C) # Candidato
        C_t = f_t * C_{t-1} + i_t * C_tilde        # Cell state
        o_t = sigmoid(W_o @ [h_{t-1}, x_t] + b_o)  # Output gate
        h_t = o_t * tanh(C_t)                      # Hidden state
    """
    
    def __init__(self, input_size: int, hidden_size: int, seed: int = 42):
        """
        Args:
            input_size: Dimensión de entrada
            hidden_size: Dimensión del estado oculto
            seed: Semilla para reproducibilidad
        """
        np.random.seed(seed)
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Tamaño combinado [h, x]
        combined_size = hidden_size + input_size
        
        # Inicialización Xavier
        limit = np.sqrt(6 / (combined_size + hidden_size))
        
        # Pesos para forget gate
        self.W_f = np.random.uniform(-limit, limit, (hidden_size, combined_size))
        self.b_f = np.zeros((hidden_size, 1))
        
        # Pesos para input gate
        self.W_i = np.random.uniform(-limit, limit, (hidden_size, combined_size))
        self.b_i = np.zeros((hidden_size, 1))
        
        # Pesos para candidato
        self.W_C = np.random.uniform(-limit, limit, (hidden_size, combined_size))
        self.b_C = np.zeros((hidden_size, 1))
        
        # Pesos para output gate
        self.W_o = np.random.uniform(-limit, limit, (hidden_size, combined_size))
        self.b_o = np.zeros((hidden_size, 1))
        
        # Cache
        self.cache = {}
    
    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        """Función sigmoide estable numéricamente."""
        return np.where(
            x >= 0,
            1 / (1 + np.exp(-x)),
            np.exp(x) / (1 + np.exp(x))
        )
    
    def forward(self, x_t: np.ndarray, h_prev: np.ndarray, 
                C_prev: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass de una celda LSTM.
        
        Args:
            x_t: Entrada en tiempo t, shape (input_size, batch_size)
            h_prev: Hidden state anterior, shape (hidden_size, batch_size)
            C_prev: Cell state anterior, shape (hidden_size, batch_size)
        
        Returns:
            h_t: Nuevo hidden state
            C_t: Nuevo cell state
        """
        # Asegurar dimensiones correctas
        if x_t.ndim == 1:
            x_t = x_t.reshape(-1, 1)
        if h_prev.ndim == 1:
            h_prev = h_prev.reshape(-1, 1)
        if C_prev.ndim == 1:
            C_prev = C_prev.reshape(-1, 1)
        
        # Concatenar [h_{t-1}, x_t]
        combined = np.vstack([h_prev, x_t])
        
        # Forget gate: f_t = σ(W_f @ [h, x] + b_f)
        f_t = self.sigmoid(self.W_f @ combined + self.b_f)
        
        # Input gate: i_t = σ(W_i @ [h, x] + b_i)
        i_t = self.sigmoid(self.W_i @ combined + self.b_i)
        
        # Candidato: C_tilde = tanh(W_C @ [h, x] + b_C)
        C_tilde = np.tanh(self.W_C @ combined + self.b_C)
        
        # Cell state: C_t = f_t ⊙ C_{t-1} + i_t ⊙ C_tilde
        C_t = f_t * C_prev + i_t * C_tilde
        
        # Output gate: o_t = σ(W_o @ [h, x] + b_o)
        o_t = self.sigmoid(self.W_o @ combined + self.b_o)
        
        # Hidden state: h_t = o_t ⊙ tanh(C_t)
        h_t = o_t * np.tanh(C_t)
        
        # Guardar para backward
        self.cache = {
            'x_t': x_t,
            'h_prev': h_prev,
            'C_prev': C_prev,
            'combined': combined,
            'f_t': f_t,
            'i_t': i_t,
            'C_tilde': C_tilde,
            'C_t': C_t,
            'o_t': o_t,
            'h_t': h_t
        }
        
        return h_t, C_t
    
    def backward(self, dh_t: np.ndarray, 
                 dC_t: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Backward pass de una celda LSTM.
        
        Args:
            dh_t: Gradiente de h_t
            dC_t: Gradiente de C_t
        
        Returns:
            dh_prev: Gradiente para h_{t-1}
            dC_prev: Gradiente para C_{t-1}
            dx_t: Gradiente para x_t
        """
        # Recuperar valores del cache
        x_t = self.cache['x_t']
        h_prev = self.cache['h_prev']
        C_prev = self.cache['C_prev']
        combined = self.cache['combined']
        f_t = self.cache['f_t']
        i_t = self.cache['i_t']
        C_tilde = self.cache['C_tilde']
        C_t = self.cache['C_t']
        o_t = self.cache['o_t']
        
        # Gradiente a través del output gate
        # h_t = o_t * tanh(C_t)
        dtanh_C_t = 1 - np.tanh(C_t) ** 2
        dC_t += dh_t * o_t * dtanh_C_t  # Acumular gradiente de C_t
        do_t = dh_t * np.tanh(C_t)
        
        # Output gate: o_t = sigmoid(...)
        dsigmoid_o = o_t * (1 - o_t)
        dW_o = (do_t * dsigmoid_o) @ combined.T
        db_o = np.sum(do_t * dsigmoid_o, axis=1, keepdims=True)
        dcombined_o = self.W_o.T @ (do_t * dsigmoid_o)
        
        # Cell state: C_t = f_t * C_{t-1} + i_t * C_tilde
        df_t = dC_t * C_prev
        dC_prev = dC_t * f_t
        di_t = dC_t * C_tilde
        dC_tilde = dC_t * i_t
        
        # Candidato: C_tilde = tanh(...)
        dtanh_C_tilde = 1 - C_tilde ** 2
        dW_C = (dC_tilde * dtanh_C_tilde) @ combined.T
        db_C = np.sum(dC_tilde * dtanh_C_tilde, axis=1, keepdims=True)
        dcombined_C = self.W_C.T @ (dC_tilde * dtanh_C_tilde)
        
        # Input gate: i_t = sigmoid(...)
        dsigmoid_i = i_t * (1 - i_t)
        dW_i = (di_t * dsigmoid_i) @ combined.T
        db_i = np.sum(di_t * dsigmoid_i, axis=1, keepdims=True)
        dcombined_i = self.W_i.T @ (di_t * dsigmoid_i)
        
        # Forget gate: f_t = sigmoid(...)
        dsigmoid_f = f_t * (1 - f_t)
        dW_f = (df_t * dsigmoid_f) @ combined.T
        db_f = np.sum(df_t * dsigmoid_f, axis=1, keepdims=True)
        dcombined_f = self.W_f.T @ (df_t * dsigmoid_f)
        
        # Sumar gradientes de combined
        dcombined = dcombined_o + dcombined_C + dcombined_i + dcombined_f
        
        # Separar en dh_prev y dx_t
        dh_prev = dcombined[:self.hidden_size, :]
        dx_t = dcombined[self.hidden_size:, :]
        
        # Guardar gradientes de pesos
        self.dW_f, self.db_f = dW_f, db_f
        self.dW_i, self.db_i = dW_i, db_i
        self.dW_C, self.db_C = dW_C, db_C
        self.dW_o, self.db_o = dW_o, db_o
        
        return dh_prev, dC_prev, dx_t


class LSTMNumPy:
    """
    LSTM completa que procesa secuencias.
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 seed: int = 42):
        """
        Args:
            input_size: Dimensión de entrada
            hidden_size: Dimensión del estado oculto
            output_size: Dimensión de salida
            seed: Semilla
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Celda LSTM
        self.cell = LSTMCellNumPy(input_size, hidden_size, seed)
        
        # Capa de salida
        np.random.seed(seed)
        limit = np.sqrt(6 / (hidden_size + output_size))
        self.W_hy = np.random.uniform(-limit, limit, (output_size, hidden_size))
        self.b_y = np.zeros((output_size, 1))
        
        # Cache
        self.hidden_states = []
        self.cell_states = []
    
    def forward(self, X: np.ndarray, h_0: Optional[np.ndarray] = None,
                C_0: Optional[np.ndarray] = None) -> Tuple[np.ndarray, List, List]:
        """
        Forward pass en una secuencia.
        
        Args:
            X: Secuencia, shape (seq_len, input_size, batch_size)
            h_0: Hidden state inicial
            C_0: Cell state inicial
        
        Returns:
            outputs: Salidas
            hidden_states: Lista de hidden states
            cell_states: Lista de cell states
        """
        seq_len = X.shape[0]
        batch_size = X.shape[2] if X.ndim == 3 else 1
        
        # Inicializar estados
        if h_0 is None:
            h_0 = np.zeros((self.hidden_size, batch_size))
        if C_0 is None:
            C_0 = np.zeros((self.hidden_size, batch_size))
        
        h_t, C_t = h_0, C_0
        outputs = []
        self.hidden_states = [h_0]
        self.cell_states = [C_0]
        
        # Procesar secuencia
        for t in range(seq_len):
            x_t = X[t]
            if x_t.ndim == 1:
                x_t = x_t.reshape(-1, 1)
            
            # LSTM step
            h_t, C_t = self.cell.forward(x_t, h_t, C_t)
            self.hidden_states.append(h_t)
            self.cell_states.append(C_t)
            
            # Output
            y_t = self.W_hy @ h_t + self.b_y
            outputs.append(y_t)
        
        outputs = np.array(outputs)
        return outputs, self.hidden_states[1:], self.cell_states[1:]
    
    def predict(self, X: np.ndarray, h_0: Optional[np.ndarray] = None,
                C_0: Optional[np.ndarray] = None) -> np.ndarray:
        """Predicción sin guardar cache."""
        outputs, _, _ = self.forward(X, h_0, C_0)
        return outputs


# =============================================================================
# PARTE 3: GRU DESDE CERO
# =============================================================================

class GRUCellNumPy:
    """
    Celda GRU (Gated Recurrent Unit) desde cero.
    
    Ecuaciones:
        r_t = sigmoid(W_r @ [h_{t-1}, x_t] + b_r)  # Reset gate
        z_t = sigmoid(W_z @ [h_{t-1}, x_t] + b_z)  # Update gate
        h_tilde = tanh(W_h @ [r_t * h_{t-1}, x_t] + b_h)  # Candidato
        h_t = (1 - z_t) * h_{t-1} + z_t * h_tilde  # Hidden state
    """
    
    def __init__(self, input_size: int, hidden_size: int, seed: int = 42):
        """
        Args:
            input_size: Dimensión de entrada
            hidden_size: Dimensión del estado oculto
            seed: Semilla
        """
        np.random.seed(seed)
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        combined_size = hidden_size + input_size
        limit = np.sqrt(6 / (combined_size + hidden_size))
        
        # Reset gate
        self.W_r = np.random.uniform(-limit, limit, (hidden_size, combined_size))
        self.b_r = np.zeros((hidden_size, 1))
        
        # Update gate
        self.W_z = np.random.uniform(-limit, limit, (hidden_size, combined_size))
        self.b_z = np.zeros((hidden_size, 1))
        
        # Candidato
        self.W_h = np.random.uniform(-limit, limit, (hidden_size, combined_size))
        self.b_h = np.zeros((hidden_size, 1))
        
        self.cache = {}
    
    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        """Sigmoid estable."""
        return np.where(
            x >= 0,
            1 / (1 + np.exp(-x)),
            np.exp(x) / (1 + np.exp(x))
        )
    
    def forward(self, x_t: np.ndarray, h_prev: np.ndarray) -> np.ndarray:
        """
        Forward pass de GRU.
        
        Args:
            x_t: Entrada
            h_prev: Hidden state anterior
        
        Returns:
            h_t: Nuevo hidden state
        """
        if x_t.ndim == 1:
            x_t = x_t.reshape(-1, 1)
        if h_prev.ndim == 1:
            h_prev = h_prev.reshape(-1, 1)
        
        # Concatenar [h_{t-1}, x_t]
        combined = np.vstack([h_prev, x_t])
        
        # Reset gate: r_t = σ(W_r @ [h, x] + b_r)
        r_t = self.sigmoid(self.W_r @ combined + self.b_r)
        
        # Update gate: z_t = σ(W_z @ [h, x] + b_z)
        z_t = self.sigmoid(self.W_z @ combined + self.b_z)
        
        # Candidato: h_tilde = tanh(W_h @ [r_t * h, x] + b_h)
        combined_reset = np.vstack([r_t * h_prev, x_t])
        h_tilde = np.tanh(self.W_h @ combined_reset + self.b_h)
        
        # Hidden state: h_t = (1 - z_t) * h_{t-1} + z_t * h_tilde
        h_t = (1 - z_t) * h_prev + z_t * h_tilde
        
        # Cache
        self.cache = {
            'x_t': x_t,
            'h_prev': h_prev,
            'combined': combined,
            'combined_reset': combined_reset,
            'r_t': r_t,
            'z_t': z_t,
            'h_tilde': h_tilde,
            'h_t': h_t
        }
        
        return h_t


class GRUNumPy:
    """
    GRU completa para procesar secuencias.
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 seed: int = 42):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.cell = GRUCellNumPy(input_size, hidden_size, seed)
        
        np.random.seed(seed)
        limit = np.sqrt(6 / (hidden_size + output_size))
        self.W_hy = np.random.uniform(-limit, limit, (output_size, hidden_size))
        self.b_y = np.zeros((output_size, 1))
        
        self.hidden_states = []
    
    def forward(self, X: np.ndarray, h_0: Optional[np.ndarray] = None):
        """Forward pass en secuencia."""
        seq_len = X.shape[0]
        batch_size = X.shape[2] if X.ndim == 3 else 1
        
        if h_0 is None:
            h_0 = np.zeros((self.hidden_size, batch_size))
        
        h_t = h_0
        outputs = []
        self.hidden_states = [h_0]
        
        for t in range(seq_len):
            x_t = X[t]
            if x_t.ndim == 1:
                x_t = x_t.reshape(-1, 1)
            
            h_t = self.cell.forward(x_t, h_t)
            self.hidden_states.append(h_t)
            
            y_t = self.W_hy @ h_t + self.b_y
            outputs.append(y_t)
        
        return np.array(outputs), self.hidden_states[1:]


# =============================================================================
# PARTE 4: PYTORCH IMPLEMENTATIONS
# =============================================================================

if PYTORCH_AVAILABLE:
    
    class SimpleLSTM(nn.Module):
        """
        LSTM simple con PyTorch para clasificación de secuencias.
        Útil para clasificación de texto o series temporales.
        """
        
        def __init__(self, input_size: int, hidden_size: int, 
                     output_size: int, num_layers: int = 1, 
                     bidirectional: bool = False, dropout: float = 0.0):
            """
            Args:
                input_size: Dimensión de entrada
                hidden_size: Dimensión del hidden state
                output_size: Dimensión de salida
                num_layers: Número de capas LSTM apiladas
                bidirectional: Si usar LSTM bidireccional
                dropout: Dropout entre capas (si num_layers > 1)
            """
            super().__init__()
            
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.output_size = output_size
            self.num_layers = num_layers
            self.bidirectional = bidirectional
            
            # LSTM layer
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=False,  # (seq_len, batch, features)
                bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0
            )
            
            # Capa de salida
            lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
            self.fc = nn.Linear(lstm_output_size, output_size)
        
        def forward(self, x, hidden=None):
            """
            Args:
                x: Input tensor (seq_len, batch, input_size)
                hidden: Tuple (h_0, c_0) opcional
            
            Returns:
                output: Predicción final (batch, output_size)
                hidden: (h_n, c_n)
            """
            # LSTM forward
            lstm_out, hidden = self.lstm(x, hidden)
            
            # Tomar último output (para clasificación many-to-one)
            # Si bidirectional, lstm_out[-1] contiene ambas direcciones
            last_output = lstm_out[-1]  # (batch, hidden_size * num_directions)
            
            # Fully connected
            output = self.fc(last_output)  # (batch, output_size)
            
            return output, hidden
    
    
    class Seq2SeqLSTM(nn.Module):
        """
        LSTM para predicción secuencia-a-secuencia.
        Útil para series temporales, generación de texto.
        """
        
        def __init__(self, input_size: int, hidden_size: int, 
                     output_size: int, num_layers: int = 1):
            """
            Args:
                input_size: Dimensión de entrada
                hidden_size: Dimensión del hidden state
                output_size: Dimensión de salida en cada paso
                num_layers: Número de capas LSTM
            """
            super().__init__()
            
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=False
            )
            
            self.fc = nn.Linear(hidden_size, output_size)
        
        def forward(self, x, hidden=None):
            """
            Args:
                x: (seq_len, batch, input_size)
                hidden: Tuple opcional (h_0, c_0)
            
            Returns:
                outputs: (seq_len, batch, output_size)
                hidden: (h_n, c_n)
            """
            lstm_out, hidden = self.lstm(x, hidden)
            
            # Aplicar FC a cada paso temporal
            seq_len, batch, _ = lstm_out.shape
            lstm_out_reshaped = lstm_out.view(-1, self.hidden_size)
            outputs = self.fc(lstm_out_reshaped)
            outputs = outputs.view(seq_len, batch, -1)
            
            return outputs, hidden
    
    
    class CharRNN(nn.Module):
        """
        RNN para generación de texto carácter por carácter.
        Ejemplo clásico de RNN generativa.
        """
        
        def __init__(self, vocab_size: int, embed_size: int, 
                     hidden_size: int, num_layers: int = 1, 
                     rnn_type: str = 'lstm'):
            """
            Args:
                vocab_size: Tamaño del vocabulario
                embed_size: Dimensión de embeddings
                hidden_size: Dimensión del hidden state
                num_layers: Número de capas
                rnn_type: 'lstm', 'gru', o 'rnn'
            """
            super().__init__()
            
            self.vocab_size = vocab_size
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.rnn_type = rnn_type.lower()
            
            # Embedding layer
            self.embedding = nn.Embedding(vocab_size, embed_size)
            
            # RNN layer
            if self.rnn_type == 'lstm':
                self.rnn = nn.LSTM(embed_size, hidden_size, num_layers, 
                                   batch_first=True)
            elif self.rnn_type == 'gru':
                self.rnn = nn.GRU(embed_size, hidden_size, num_layers,
                                  batch_first=True)
            else:
                self.rnn = nn.RNN(embed_size, hidden_size, num_layers,
                                  batch_first=True)
            
            # Output layer
            self.fc = nn.Linear(hidden_size, vocab_size)
        
        def forward(self, x, hidden=None):
            """
            Args:
                x: Input indices (batch, seq_len)
                hidden: Hidden state
            
            Returns:
                output: Logits (batch, seq_len, vocab_size)
                hidden: Nuevo hidden state
            """
            # Embedding
            embedded = self.embedding(x)  # (batch, seq_len, embed_size)
            
            # RNN
            rnn_out, hidden = self.rnn(embedded, hidden)
            
            # Output
            output = self.fc(rnn_out)  # (batch, seq_len, vocab_size)
            
            return output, hidden
        
        def init_hidden(self, batch_size, device='cpu'):
            """Inicializar hidden state."""
            if self.rnn_type == 'lstm':
                h0 = torch.zeros(self.num_layers, batch_size, 
                                self.hidden_size).to(device)
                c0 = torch.zeros(self.num_layers, batch_size,
                                self.hidden_size).to(device)
                return (h0, c0)
            else:
                return torch.zeros(self.num_layers, batch_size,
                                  self.hidden_size).to(device)


# =============================================================================
# PARTE 5: FUNCIONES AUXILIARES
# =============================================================================

def generate_sine_sequence(seq_len: int, num_sequences: int = 1,
                          noise: float = 0.0) -> np.ndarray:
    """
    Generar secuencias de seno para pruebas.
    
    Args:
        seq_len: Longitud de la secuencia
        num_sequences: Número de secuencias
        noise: Nivel de ruido a añadir
    
    Returns:
        Array de shape (num_sequences, seq_len)
    """
    t = np.linspace(0, 4 * np.pi, seq_len)
    sequences = []
    
    for _ in range(num_sequences):
        # Frecuencia y fase aleatorias
        freq = np.random.uniform(0.5, 2.0)
        phase = np.random.uniform(0, 2 * np.pi)
        
        seq = np.sin(freq * t + phase)
        if noise > 0:
            seq += np.random.normal(0, noise, seq_len)
        
        sequences.append(seq)
    
    return np.array(sequences)


def create_sequences(data: np.ndarray, seq_length: int, 
                     pred_length: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crear secuencias de entrada y salida para series temporales.
    
    Args:
        data: Array 1D de datos
        seq_length: Longitud de la secuencia de entrada
        pred_length: Longitud de la secuencia a predecir
    
    Returns:
        X: Secuencias de entrada (num_sequences, seq_length)
        y: Secuencias objetivo (num_sequences, pred_length)
    """
    X, y = [], []
    
    for i in range(len(data) - seq_length - pred_length + 1):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length:i + seq_length + pred_length])
    
    return np.array(X), np.array(y)


def gradient_clipping(gradients: List[np.ndarray], max_norm: float = 5.0):
    """
    Aplicar gradient clipping para prevenir explosión de gradientes.
    
    Args:
        gradients: Lista de arrays de gradientes
        max_norm: Norma máxima permitida
    
    Returns:
        gradients: Gradientes clipeados
    """
    # Calcular norma total
    total_norm = 0
    for grad in gradients:
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)
    
    # Clipear si es necesario
    if total_norm > max_norm:
        clip_coef = max_norm / (total_norm + 1e-6)
        gradients = [grad * clip_coef for grad in gradients]
    
    return gradients


def visualize_hidden_states(hidden_states: List[np.ndarray], 
                           title: str = "Hidden States Evolution"):
    """
    Visualizar evolución de hidden states a través del tiempo.
    
    Args:
        hidden_states: Lista de hidden states (seq_len, hidden_size, batch)
        title: Título del gráfico
    """
    # Convertir a array (seq_len, hidden_size)
    if isinstance(hidden_states, list):
        # Tomar primer ejemplo del batch si hay batch
        hidden_states = [h[:, 0] if h.ndim > 1 else h for h in hidden_states]
        hidden_array = np.array(hidden_states)
    else:
        hidden_array = hidden_states
    
    if hidden_array.ndim == 3:
        hidden_array = hidden_array[:, :, 0]
    
    plt.figure(figsize=(12, 6))
    plt.imshow(hidden_array.T, aspect='auto', cmap='viridis', 
               interpolation='nearest')
    plt.colorbar(label='Activation')
    plt.xlabel('Time Step')
    plt.ylabel('Hidden Unit')
    plt.title(title)
    plt.tight_layout()
    plt.show()


def visualize_gates(lstm_cell: LSTMCellNumPy, seq_length: int = 20):
    """
    Visualizar activación de puertas LSTM durante una secuencia.
    
    Args:
        lstm_cell: Celda LSTM
        seq_length: Longitud de la secuencia a procesar
    """
    # Generar secuencia simple
    X = np.random.randn(seq_length, lstm_cell.input_size, 1)
    
    # Procesar y guardar activaciones de puertas
    h_t = np.zeros((lstm_cell.hidden_size, 1))
    C_t = np.zeros((lstm_cell.hidden_size, 1))
    
    forget_gates = []
    input_gates = []
    output_gates = []
    
    for t in range(seq_length):
        h_t, C_t = lstm_cell.forward(X[t], h_t, C_t)
        forget_gates.append(lstm_cell.cache['f_t'][:, 0])
        input_gates.append(lstm_cell.cache['i_t'][:, 0])
        output_gates.append(lstm_cell.cache['o_t'][:, 0])
    
    # Convertir a arrays
    forget_gates = np.array(forget_gates)
    input_gates = np.array(input_gates)
    output_gates = np.array(output_gates)
    
    # Visualizar
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    
    axes[0].imshow(forget_gates.T, aspect='auto', cmap='coolwarm',
                   vmin=0, vmax=1)
    axes[0].set_title('Forget Gate Activations')
    axes[0].set_ylabel('Hidden Unit')
    
    axes[1].imshow(input_gates.T, aspect='auto', cmap='coolwarm',
                   vmin=0, vmax=1)
    axes[1].set_title('Input Gate Activations')
    axes[1].set_ylabel('Hidden Unit')
    
    axes[2].imshow(output_gates.T, aspect='auto', cmap='coolwarm',
                   vmin=0, vmax=1)
    axes[2].set_title('Output Gate Activations')
    axes[2].set_xlabel('Time Step')
    axes[2].set_ylabel('Hidden Unit')
    
    plt.tight_layout()
    plt.show()


# =============================================================================
# PARTE 6: EJEMPLOS DE ENTRENAMIENTO (PyTorch)
# =============================================================================

if PYTORCH_AVAILABLE:
    
    def train_simple_lstm(model: nn.Module, train_loader: DataLoader,
                         criterion, optimizer, num_epochs: int = 10,
                         device: str = 'cpu', verbose: bool = True):
        """
        Entrenar modelo LSTM simple.
        
        Args:
            model: Modelo PyTorch
            train_loader: DataLoader con datos de entrenamiento
            criterion: Función de pérdida
            optimizer: Optimizador
            num_epochs: Número de épocas
            device: 'cpu' o 'cuda'
            verbose: Mostrar progreso
        
        Returns:
            losses: Lista de pérdidas por época
        """
        model.to(device)
        model.train()
        losses = []
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            num_batches = 0
            
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                # Forward
                optimizer.zero_grad()
                output, _ = model(batch_x)
                loss = criterion(output, batch_y)
                
                # Backward
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            losses.append(avg_loss)
            
            if verbose and (epoch + 1) % max(1, num_epochs // 10) == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
        
        return losses
    
    
    def predict_sequence(model: nn.Module, initial_sequence: torch.Tensor,
                        num_steps: int, device: str = 'cpu') -> np.ndarray:
        """
        Predecir próximos pasos de una secuencia.
        
        Args:
            model: Modelo entrenado
            initial_sequence: Secuencia inicial (seq_len, batch, input_size)
            num_steps: Número de pasos a predecir
            device: 'cpu' o 'cuda'
        
        Returns:
            predictions: Array con predicciones
        """
        model.eval()
        model.to(device)
        
        with torch.no_grad():
            # Procesar secuencia inicial
            _, hidden = model(initial_sequence.to(device))
            
            # Último valor como entrada inicial
            current_input = initial_sequence[-1:, :, :]
            
            predictions = []
            
            for _ in range(num_steps):
                # Predecir siguiente paso
                output, hidden = model(current_input.to(device), hidden)
                predictions.append(output.cpu().numpy())
                
                # Usar predicción como próxima entrada
                current_input = output.unsqueeze(0)
        
        return np.concatenate(predictions, axis=0)


# =============================================================================
# FUNCIÓN DE DEMOSTRACIÓN
# =============================================================================

def demo_rnn_vs_lstm():
    """
    Demostración comparando RNN vanilla vs LSTM.
    """
    print("=" * 70)
    print("DEMOSTRACIÓN: RNN vs LSTM")
    print("=" * 70)
    
    # Parámetros
    input_size = 1
    hidden_size = 10
    output_size = 1
    seq_len = 20
    
    # Crear secuencia de ejemplo (seno)
    t = np.linspace(0, 4 * np.pi, seq_len)
    X = np.sin(t).reshape(seq_len, input_size, 1)
    
    print(f"\nSecuencia de entrada: {seq_len} pasos temporales")
    print(f"Input size: {input_size}, Hidden size: {hidden_size}")
    
    # RNN vanilla
    print("\n1. RNN Vanilla:")
    rnn = RNNNumPy(input_size, hidden_size, output_size, seed=42)
    rnn_out, rnn_hidden = rnn.forward(X)
    print(f"   - Output shape: {rnn_out.shape}")
    print(f"   - Número de hidden states: {len(rnn_hidden)}")
    print(f"   - Total parámetros: {hidden_size * (input_size + hidden_size + 1) + output_size * (hidden_size + 1)}")
    
    # LSTM
    print("\n2. LSTM:")
    lstm = LSTMNumPy(input_size, hidden_size, output_size, seed=42)
    lstm_out, lstm_hidden, lstm_cell = lstm.forward(X)
    print(f"   - Output shape: {lstm_out.shape}")
    print(f"   - Número de hidden states: {len(lstm_hidden)}")
    print(f"   - Número de cell states: {len(lstm_cell)}")
    print(f"   - Total parámetros: ~{4 * hidden_size * (input_size + hidden_size + 1) + output_size * (hidden_size + 1)}")
    
    # GRU
    print("\n3. GRU:")
    gru = GRUNumPy(input_size, hidden_size, output_size, seed=42)
    gru_out, gru_hidden = gru.forward(X)
    print(f"   - Output shape: {gru_out.shape}")
    print(f"   - Número de hidden states: {len(gru_hidden)}")
    print(f"   - Total parámetros: ~{3 * hidden_size * (input_size + hidden_size + 1) + output_size * (hidden_size + 1)}")
    
    print("\n" + "=" * 70)
    print("Comparación:")
    print(f"LSTM tiene ~4x más parámetros que RNN (puertas)")
    print(f"GRU tiene ~3x más parámetros que RNN (menos que LSTM)")
    print(f"LSTM y GRU resuelven el problema del gradiente desvaneciente")
    print("=" * 70)


if __name__ == "__main__":
    # Ejecutar demostración
    demo_rnn_vs_lstm()
    
    print("\n\n✓ Implementaciones de RNN, LSTM y GRU completadas!")
    print("  Ver practica.ipynb para ejemplos interactivos.")
