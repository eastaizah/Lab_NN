# Planificación y Dimensionamiento de Redes 5G: Una Revisión Comprehensiva

## *Segunda Parte: Secciones V–X y Referencias*

---

## V. MASSIVE MIMO Y BEAMFORMING

### A. Fundamentos de Massive MIMO

Massive MIMO (Multiple Input Multiple Output masivo) constituye una de las tecnologías habilitadoras más transformadoras de la quinta generación de comunicaciones móviles. Propuesto originalmente por Marzetta en 2010 [39], el concepto establece que una estación base equipada con un número muy grande de antenas —típicamente $M \geq 64$, con valores prácticos entre 64 y 512— puede atender simultáneamente a $K \ll M$ usuarios en el mismo recurso tiempo-frecuencia, aprovechando los grados de libertad espaciales adicionales para implementar beamforming preciso y cancelación de interferencias. La condición fundamental $M \gg K$ garantiza que los canales de los distintos usuarios sean asintóticamente ortogonales (fenómeno de "favorable propagation"), lo que simplifica drásticamente la detección y el precodificado [40].

El array de antenas en un sistema Massive MIMO se modela habitualmente como un Uniform Linear Array (ULA) o un Uniform Planar Array (UPA). Para un ULA de $M$ elementos con separación entre antenas $d$ (típicamente $d = \lambda/2$ siendo $\lambda$ la longitud de onda), el vector de dirección (*array steering vector*) para una señal proveniente del ángulo de incidencia $\theta$ (medido respecto a la normal al array) se define como [41]:

$$\mathbf{a}(\theta) = \frac{1}{\sqrt{M}}\left[1,\; e^{j2\pi\frac{d}{\lambda}\sin(\theta)},\; e^{j2\pi\cdot 2\frac{d}{\lambda}\sin(\theta)},\; \ldots,\; e^{j2\pi(M-1)\frac{d}{\lambda}\sin(\theta)}\right]^T \tag{31}$$

donde el factor $1/\sqrt{M}$ es la normalización de potencia que garantiza $\|\mathbf{a}(\theta)\|^2 = 1$. Este vector captura la respuesta de fase diferencial entre cada par de antenas consecutivas del array, y su estructura periódica es la base del beamforming dirigido: al ponderar apropiadamente las señales transmitidas (o recibidas) por cada antena, es posible concentrar la energía en direcciones específicas del espacio.

La ganancia de array (*array gain*) resultante de combinar coherentemente las señales de $M$ antenas se expresa en escala logarítmica como [42]:

$$G_{\text{array}} = 10 \cdot \log_{10}(M) \quad [\text{dB}] \tag{32}$$

Para $M = 64$ antenas, la ganancia de array asciende a $G_{\text{array}} = 18.06$ dB, lo que implica una mejora equivalente en la cobertura (mayor alcance o menor potencia necesaria) o en la capacidad del sistema. Para $M = 128$ y $M = 256$, los valores son $21.07$ dB y $24.08$ dB respectivamente, mostrando un crecimiento logarítmico que explica el interés industrial en arrays de gran tamaño.

### B. Precodificador MRT (Maximum Ratio Transmission)

El precoder MRT, también denominado filtro adaptado o *conjugate beamforming*, es la técnica de beamforming más simple y computacionalmente eficiente. Para un vector de canal $\mathbf{h}_k \in \mathbb{C}^{M \times 1}$ del usuario $k$ (canal estimado entre las $M$ antenas del gNB y la antena del UE), el vector de precodificación MRT se define como [43]:

$$\mathbf{w}_{\text{MRT}} = \frac{\mathbf{h}_k^*}{\|\mathbf{h}_k\|} \tag{33}$$

es decir, el conjugado del vector de canal normalizado por su norma euclídea. La señal recibida en el UE tras la aplicación del precoder MRT es $y_k = \mathbf{h}_k^T \mathbf{w}_{\text{MRT}} s_k + n_k$, donde la potencia de señal útil escala como $|\mathbf{h}_k^T \mathbf{w}_{\text{MRT}}|^2 = \|\mathbf{h}_k\|^2$. La tasa espectral alcanzable con MRT en un sistema de usuario único se expresa entonces como [39]:

$$R_{\text{MRT}} = \log_2\!\left(1 + M \cdot \rho \cdot \|\mathbf{h}_k\|^2\right) \tag{34}$$

donde $\rho$ es la relación señal-a-ruido (SNR) por antena y el factor $M$ refleja precisamente la ganancia de array descrita en la ecuación (32). Sin embargo, en escenarios multi-usuario, el precoder MRT no cancela la interferencia entre usuarios, ya que $\mathbf{h}_j^T \mathbf{w}_{\text{MRT},k} \neq 0$ para $j \neq k$ en general. Esta limitación se mitiga en el régimen de $M \to \infty$ gracias a la ortogonalidad asintótica de los canales, pero para valores finitos de $M$ la interferencia entre usuarios (MUI, Multi-User Interference) degrada significativamente el SINR [44].

### C. Precoder Zero-Forcing (ZF) Multi-Usuario

El precoder Zero-Forcing (ZF) resuelve el problema de la interferencia multi-usuario eliminando exactamente la MUI mediante una pseudoinversa de la matriz de canal agregada. Dado el canal multi-usuario $\mathbf{H} = [\mathbf{h}_1, \mathbf{h}_2, \ldots, \mathbf{h}_K]^H \in \mathbb{C}^{K \times M}$, la matriz de precodificación ZF se construye como [45]:

$$\mathbf{W}_{\text{ZF}} = \mathbf{H}^H \left(\mathbf{H} \mathbf{H}^H\right)^{-1} \tag{35}$$

La construcción de la ecuación (35) garantiza que $\mathbf{H} \mathbf{W}_{\text{ZF}} = \mathbf{I}_K$, es decir, la proyección del canal de cada usuario sobre los precoders de los demás es exactamente cero, eliminando toda interferencia inter-usuario. El vector de precodificación del usuario $k$ es la $k$-ésima columna de $\mathbf{W}_{\text{ZF}}$, denotada $\mathbf{w}_k^{\text{ZF}}$, y el SINR resultante con normalización de potencia es [45]:

$$\text{SINR}_k^{\text{ZF}} = \frac{\rho}{\|\mathbf{w}_k^{\text{ZF}}\|^2 \cdot \sigma^2} \tag{36}$$

donde $\sigma^2$ es la varianza del ruido aditivo gaussiano blanco (AWGN) en el receptor. El denominador $\|\mathbf{w}_k^{\text{ZF}}\|^2$ representa el coste en potencia de la cancelación de interferencias: cuanto más correlacionados estén los canales de los usuarios, mayor es la norma del precoder ZF y mayor la penalización en SNR efectiva. En el régimen $M \gg K$, los canales son aproximadamente ortogonales, $\|\mathbf{w}_k^{\text{ZF}}\|^2 \approx 1$, y el ZF converge al rendimiento del MRT [40].

### D. Estimación de Canal y Contaminación Piloto

La estimación precisa del canal es crítica para el funcionamiento de Massive MIMO, ya que tanto el beamforming como la detección dependen de estimaciones de alta calidad de $\mathbf{H}$. El estimador MMSE (Minimum Mean Square Error) para el canal $\mathbf{h}_k$ del usuario $k$, a partir de la observación de la señal piloto $\mathbf{y}_p \in \mathbb{C}^{M \times 1}$ recibida durante la fase de entrenamiento, se expresa como [46]:

$$\hat{\mathbf{h}}_k = \frac{\sqrt{\rho_p} \cdot \beta_k}{\rho_p \sum_{k'} \beta_{k'} + 1} \cdot \mathbf{y}_p \tag{37}$$

donde $\rho_p$ es la SNR de la señal piloto y $\beta_k$ es el coeficiente de desvanecimiento a gran escala (*large-scale fading*) del usuario $k$, que incluye pérdidas de trayectoria y shadowing. El sumatorio $\sum_{k'} \beta_{k'}$ en el denominador incluye las contribuciones de todos los usuarios que transmiten el mismo piloto en celdas vecinas, modelando el efecto de *contaminación de pilotos* (pilot contamination): la estimación de $\mathbf{h}_k$ queda corrompida por los canales de usuarios interferentes $k' \neq k$ que comparten la misma secuencia de pilotos [39][47].

La contaminación de pilotos es el cuello de botella fundamental de Massive MIMO en sistemas multicelda, ya que no desaparece al incrementar $M$, sino que permanece constante en el límite asintótico. Las estrategias para mitigarla incluyen: asignación inteligente de pilotos basada en información de gran escala [47], estimación piloto distribuida con cooperación entre celdas [48], y técnicas de supresión de interferencia de pilotos basadas en filtros espaciales.

### E. Arquitecturas de Beamforming Híbrido

Las implementaciones prácticas de Massive MIMO en frecuencias mmWave requieren arquitecturas de beamforming híbrido para reducir el número de cadenas RF (Radio Frequency), cuyo coste y consumo de energía escalan linealmente con $M$ en un sistema completamente digital. La arquitectura híbrida descompone el precoder total en dos etapas en cascada: un precoder analógico $\mathbf{F}_{\text{RF}} \in \mathbb{C}^{M \times N_{\text{RF}}}$ implementado en el dominio analógico mediante desfasadores de fase (*phase shifters*), y un precoder digital $\mathbf{F}_{\text{BB}} \in \mathbb{C}^{N_{\text{RF}} \times K}$ implementado en banda base [48]:

$$\mathbf{F} = \mathbf{F}_{\text{RF}} \cdot \mathbf{F}_{\text{BB}} \tag{38}$$

En la ecuación (38), $N_{\text{RF}}$ es el número de cadenas RF con $K \leq N_{\text{RF}} \ll M$, lo que implica una reducción de hardware del orden de $M/N_{\text{RF}}$ en número de convertidores DAC/ADC y amplificadores. La restricción fundamental de la arquitectura analógica es que las entradas de $\mathbf{F}_{\text{RF}}$ deben tener módulo constante (condición de desfasador de fase puro), es decir, $|[\mathbf{F}_{\text{RF}}]_{ij}| = 1/\sqrt{M}$. El diseño conjunto de $\mathbf{F}_{\text{RF}}$ y $\mathbf{F}_{\text{BB}}$ para maximizar la capacidad sujeto a esta restricción de módulo constante es un problema no convexo que se aborda mediante algoritmos de búsqueda de diccionario OMP (*Orthogonal Matching Pursuit*) o métodos de optimización en variedades de Riemann [48].

---

## VI. ONDAS MILIMÉTRICAS (mmWave) EN 5G

### A. Bandas de Frecuencia y Características de Propagación

El estándar 5G NR define dos rangos de frecuencia principales: FR1 (Frequency Range 1), que abarca desde 410 MHz hasta 7125 MHz (sub-6 GHz), y FR2 (Frequency Range 2), que cubre el espectro de onda milimétrica entre 24.25 GHz y 52.6 GHz [49]. El FR2, comúnmente denominado mmWave, ofrece anchos de banda contiguos de hasta 400 MHz por portadora (frente a 100 MHz en FR1), habilitando tasas de datos pico superiores a 10 Gbps conforme a los requisitos IMT-2020 de la UIT-R [50]. Sin embargo, la propagación en frecuencias mmWave presenta características radicalmente distintas a las sub-6 GHz, planteando desafíos únicos para la planificación de red.

La absorción atmosférica constituye uno de los factores de atenuación más relevantes en mmWave. El oxígeno molecular ($\text{O}_2$) presenta una resonancia de absorción a 60 GHz con una atenuación específica de aproximadamente 15 dB/km, lo que limita el uso de esta frecuencia a enlaces de corto alcance (*backhaul* de última milla o comunicaciones en espacios cerrados). Para las bandas de interés en 5G (26 GHz, 28 GHz, 39 GHz), la absorción atmosférica es mucho menor, del orden de 0.03–0.1 dB/km, y no representa un factor limitante dominante para celdas con radios de 100–300 metros [51].

La atenuación por lluvia sigue el modelo empírico ITU-R P.838-3, que expresa la atenuación específica $\gamma_R$ en función de la intensidad de lluvia $R$ [mm/h] como [52]:

$$A_{\text{rain}} = a \cdot R^b \quad [\text{dB/km}] \tag{39}$$

donde los coeficientes $a$ y $b$ son funciones de la frecuencia, la polarización y la temperatura. Para la banda de 28 GHz con polarización horizontal, los valores típicos son $a \approx 0.2163$ y $b \approx 0.6931$ (ITU-R P.838-3). Para una lluvia intensa de $R = 50$ mm/h —correspondiente a un aguacero tropical— la atenuación es $A_{\text{rain}} \approx 5.4$ dB/km, valor significativo para el enlace de bajada pero manejable para celdas pequeñas con radios inferiores a 200 m. La atenuación de penetración en edificios (*building penetration loss*) es otro factor crítico: a 28 GHz, pérdidas de penetración de 15–25 dB en fachadas de vidrio y 30–40 dB en muros de hormigón limitan drásticamente la cobertura indoor desde estaciones base outdoor [53].

### B. Modelo de Canal mmWave: Saleh-Valenzuela Extendido

El modelo de canal Saleh-Valenzuela extendido al dominio espacial captura la naturaleza altamente dispersa y agrupada (*clustered*) de la propagación mmWave, donde el número de componentes de camino múltiple (MPC) es reducido pero cada clúster puede contener varios rayos. La matriz de canal MIMO mmWave se modela como [54]:

$$\mathbf{H} = \sqrt{\frac{N_t N_r}{N_{\text{cl}} N_{\text{ray}}}} \sum_{l=1}^{N_{\text{cl}}} \sum_{m=1}^{N_{\text{ray}}} \alpha_{lm} \cdot \mathbf{a}_r\!\left(\varphi_r^{lm}\right) \mathbf{a}_t^H\!\left(\varphi_t^{lm}\right) \tag{40}$$

donde la ecuación (40) presenta los siguientes elementos:
- $N_t$, $N_r$: número de antenas en el transmisor y el receptor respectivamente.
- $N_{\text{cl}}$: número de clústeres de propagación (típicamente $N_{\text{cl}} = 2$–$5$ en entornos urbanos mmWave).
- $N_{\text{ray}}$: número de rayos por clúster (típicamente $N_{\text{ray}} = 8$–$20$).
- $\alpha_{lm}$: ganancia compleja del rayo $m$ perteneciente al clúster $l$, que sigue una distribución $\mathcal{CN}(0, \sigma_{\alpha,l}^2)$ con varianza dependiente del ángulo de salida.
- $\mathbf{a}_r(\varphi_r^{lm})$ y $\mathbf{a}_t(\varphi_t^{lm})$: vectores de dirección del array receptor y transmisor para los ángulos de llegada y salida $\varphi_r^{lm}$ y $\varphi_t^{lm}$ respectivamente, definidos según la ecuación (31).
- El factor de normalización $\sqrt{N_t N_r / (N_{\text{cl}} N_{\text{ray}})}$ garantiza que la ganancia media del canal sea consistente con el modelo de pérdidas de trayectoria.

El factor de rango espacial (*rank*) del canal mmWave es habitualmente reducido ($\text{rank}(\mathbf{H}) \leq N_{\text{cl}}$), lo que limita el número de flujos de datos simultáneos transmisibles mediante multiplexación espacial. Esta característica diferencia fundamentalmente al canal mmWave del canal sub-6 GHz rico en dispersión (Rayleigh *i.i.d.*), y justifica el uso de beamforming dirigido en lugar de multiplexación espacial masiva como estrategia principal de ganancias [54][55].

### C. Despliegue de Small Cells para Cobertura mmWave

La corta distancia de cobertura de las estaciones base mmWave (50–300 m para cobertura continua outdoor) exige despliegues ultradensificados, configurando redes heterogéneas (HetNets) con múltiples capas de cobertura. El modelo de despliegue de red estocástica basado en Procesos Puntuales de Poisson (PPP) permite analizar analíticamente la cobertura en función de la densidad de estaciones base $\lambda_b$ [nodos/km²] [56].

En el modelo PPP homogéneo, la distancia $r$ al gNB mmWave más cercano sigue una distribución Rayleigh con función de densidad de probabilidad:

$$f_R(r) = 2\pi\lambda_b \cdot r \cdot e^{-\pi\lambda_b r^2} \tag{41}$$

La probabilidad de cobertura, definida como la probabilidad de que el SINR recibido supere un umbral de decodificación $T$, se calcula integrando sobre la distribución de distancias [57]:

$$P_c(T) = P(\text{SINR} > T) = \int_0^{\infty} P(\text{SINR} > T \mid r) \cdot f_R(r)\, dr \tag{42}$$

donde $P(\text{SINR} > T \mid r)$ es la probabilidad de cobertura condicional a una distancia $r$ del servidor, que depende del modelo de pérdidas de trayectoria (LOS/NLOS) y de la distribución de interferencia de los gNBs vecinos. Bajo el modelo de potencia de señal recibida como PPP con exponente de pérdidas $\alpha$ y ganancia de beamforming $G_b$, la probabilidad de cobertura en presencia de interferencia tipo Rayleigh tiene la forma cerrada [57]:

$$P_c(T) = \pi\lambda_b \int_0^{\infty} e^{-\pi\lambda_b r^2 \mathcal{C}(T,\alpha) - \mu T r^\alpha / (G_b P_t)} dr \tag{43}$$

donde $\mathcal{C}(T,\alpha)$ es una función que depende del umbral $T$ y del exponente $\alpha$, $\mu = \sigma_n^2 / P_t$ es la razón ruido-a-señal normalizada, y $P_t$ es la potencia de transmisión del gNB mmWave. La distancia inter-sitio óptima $d_{\text{ISD}}$ que maximiza la cobertura en términos del compromiso señal útil vs. interferencia resulta típicamente de 100–200 m para redes mmWave 5G en entornos urbanos densos [56].

### D. Requisitos de Backhaul para Celdas mmWave

El backhaul de las small cells mmWave representa un desafío crítico en la planificación de redes 5G. El caudal de backhaul requerido por cada celda mmWave con ancho de banda $B = 400$ MHz, eficiencia espectral $\eta = 5$ bits/s/Hz (256-QAM, tasa de código 3/4), y 4 sectores es:

$$C_{\text{backhaul}} = B \cdot \eta \cdot N_{\text{sectores}} = 400 \times 10^6 \cdot 5 \cdot 4 = 8 \text{ Gbps} \tag{44}$$

Este volumen de tráfico exige soluciones de backhaul de alta capacidad como: enlaces de fibra óptica directa, backhaul inalámbrico mmWave punto a punto (E-band: 71–76/81–86 GHz), o soluciones Integrated Access and Backhaul (IAB) normalizadas en 3GPP Release 16 [58], donde las mismas antenas 5G NR se utilizan para acceso al usuario y backhaul de forma integrada.

---

## VII. NETWORK SLICING Y CALIDAD DE SERVICIO (QoS)

### A. Conceptos de Network Slicing en 5G

Network slicing es una funcionalidad arquitectural fundamental de la Core Network 5G (5GC) basada en NFV (Network Functions Virtualization) y SDN (Software Defined Networking), que permite crear múltiples redes virtuales lógicamente aisladas (*slices*) sobre una infraestructura física compartida, cada una con características de QoS diferenciadas adaptadas a diferentes casos de uso [59]. La arquitectura de gestión y orquestación de slices en 5G está definida en 3GPP TS 28.530 e incluye tres tipos canónicos de slice [50]:

- **eMBB (enhanced Mobile Broadband):** optimizado para alta tasa de datos con requisitos de latencia moderados (20–100 ms). Casos de uso: streaming 4K/8K, realidad aumentada, teletrabajo de alta capacidad.
- **URLLC (Ultra-Reliable Low-Latency Communications):** prioriza latencia extremadamente baja (< 1 ms) y fiabilidad ultra-alta (99.9999%). Casos de uso: automatización industrial, cirugía remota, vehículos autónomos.
- **mMTC (massive Machine-Type Communications):** soporta densidades de conexión masivas (hasta $10^6$ dispositivos/km²) con bajo consumo de energía y baja tasa de datos. Casos de uso: IoT industrial, contadores inteligentes, sensores urbanos.

### B. Asignación de Recursos entre Slices

El problema de asignación de recursos radio entre slices concurrentes es un problema de optimización multi-objetivo que debe equilibrar la eficiencia global del sistema con el cumplimiento de los SLAs (Service Level Agreements) de cada slice. El algoritmo de scheduling *max-weight* asigna en cada intervalo de tiempo los recursos disponibles al slice (o usuario) que maximiza el producto entre su peso de prioridad $w_i$ y su longitud de cola $q_i(t)$ (medida en bits o paquetes pendientes) [60]:

$$i^* = \arg\max_{i} \left(w_i \cdot q_i(t)\right) \tag{45}$$

donde $w_i$ es un peso configurable que refleja la prioridad relativa del slice $i$ (determinada por el SLA contratado), y $q_i(t)$ es la ocupación de la cola del slice $i$ en el instante de tiempo $t$. El algoritmo max-weight es óptimo en el sentido de la estabilidad de colas bajo el modelo de redes estocásticas (*stochastic networks*), garantizando que las colas permanecen acotadas siempre que el vector de tráfico ofrecido se encuentre en el interior de la región de estabilidad del sistema [60].

El control de admisión (*admission control*) complementa al scheduling determinando si una nueva solicitud de slice puede ser admitida sin violar los SLAs de los slices ya activos. El criterio de admisión para un nuevo slice $s_{\text{new}}$ con requisito de recursos $r_{s_{\text{new}}}$ se basa en verificar la factibilidad de la asignación actualizada:

$$\sum_{s \in \mathcal{S} \cup \{s_{\text{new}}\}} r_s^{\min} \leq R_{\text{total}} \tag{46}$$

donde $r_s^{\min}$ es la asignación mínima de recursos garantizada al slice $s$ (conforme al SLA) y $R_{\text{total}}$ es la capacidad total de recursos radio disponibles (expresada en PRBs o MHz·antenna ports).

### C. Parámetros QoS y 5QI

El identificador de QoS 5G (5QI, 5G QoS Identifier) es un escalar que referencia un conjunto estandarizado de parámetros de QoS definidos en 3GPP TS 23.501 [49]. Los parámetros fundamentales de cada 5QI son:

- **PDB (Packet Delay Budget):** presupuesto de retardo extremo a extremo en milisegundos; para URLLC el PDB es de 0.5–4 ms, para eMBB es de 50–300 ms.
- **PER (Packet Error Rate):** tasa máxima de paquetes perdidos (sin recuperación HARQ); para URLLC: $10^{-5}$, para eMBB: $10^{-6}$.
- **Priority Level:** nivel de prioridad de scheduling (1–127); menor valor implica mayor prioridad.
- **ARP (Allocation and Retention Priority):** determina la prioridad en situaciones de congestión.

### D. Modelo de Latencia Extremo a Extremo

La latencia total extremo a extremo (E2E) en una red 5G se descompone como la suma de los retardos de cada segmento funcional [61]:

$$T_{\text{total}} = T_{\text{aire}} + T_{\text{fronthaul}} + T_{\text{core}} + T_{\text{procesamiento}} \tag{47}$$

El retardo en el interfaz radio $T_{\text{aire}}$ incluye la duración del TTI (Transmission Time Interval), el retardo de retransmisión HARQ (Hybrid Automatic Repeat reQuest) y el retardo de scheduling:

$$T_{\text{aire}} = T_{\text{TTI}} + T_{\text{HARQ}} + T_{\text{scheduling}} \tag{48}$$

Para satisfacer el requisito URLLC de $T_{\text{aire}} < 1$ ms, 5G NR introduce mini-slots de duración $T_{\text{TTI}} = 0.125$ ms (2 símbolos OFDM) con numerología $\mu = 3$ (SCS = 120 kHz), reduciendo drásticamente el retardo de transmisión respecto a los 1 ms del TTI de LTE. El retardo HARQ se reduce a 1 round-trip con $T_{\text{HARQ}} \approx 0.25$ ms, y el retardo de scheduling se minimiza mediante procesamiento anticipado en el gNB [62].

### E. Optimización de Asignación de Recursos por Slice

El problema de optimización de asignación de recursos entre slices para una red 5G multi-slice se formula como un problema de programación matemática [63]:

$$\min_{\{r_s\}} \sum_{s \in \mathcal{S}} C_s(r_s) \tag{49}$$

$$\text{sujeto a:} \quad \sum_{s \in \mathcal{S}} r_s \leq R_{\text{total}} \tag{50}$$

$$\text{QoS}_s(r_s) \geq \text{QoS}_s^{\min} \quad \forall s \in \mathcal{S} \tag{51}$$

$$r_s \geq 0 \quad \forall s \in \mathcal{S} \tag{52}$$

donde $C_s(r_s)$ es la función de coste de asignación del slice $s$ (que puede representar el coste operativo de los recursos asignados), la restricción (50) garantiza que la asignación total no supera la capacidad disponible $R_{\text{total}}$, la restricción (51) impone el cumplimiento del SLA de QoS mínima para cada slice (que puede incluir restricciones de tasa, latencia o fiabilidad), y la restricción (52) es la condición de no negatividad. Si las funciones $C_s$ son convexas y los requisitos de QoS se expresan como restricciones convexas en $r_s$, el problema admite solución eficiente mediante métodos de punto interior o el método dual de Lagrange [63][64].

El aislamiento entre slices y el overhead de virtualización son aspectos críticos del despliegue práctico. La hipervirtualización de funciones de red (VNFs) introduce una penalización de procesamiento estimada entre 5% y 15% respecto a implementaciones nativas (*bare-metal*), que debe ser considerada en el dimensionamiento de los recursos de cómputo del edge cloud [65][66].

---

## VIII. OPTIMIZACIÓN Y APRENDIZAJE AUTOMÁTICO EN PLANIFICACIÓN 5G

### A. Métodos Clásicos de Optimización

La planificación de la ubicación óptima de estaciones base y la asignación de frecuencias en una red 5G se formulan naturalmente como problemas de programación lineal entera mixta (MILP, *Mixed Integer Linear Programming*). El problema de colocación de sitios, que determina cuáles de un conjunto de posibles ubicaciones candidatas $\mathcal{L}$ deben activarse para minimizar el coste de despliegue satisfaciendo requisitos de cobertura y capacidad, se formula como [67]:

$$\min_{\mathbf{x}} \sum_{l \in \mathcal{L}} c_l x_l + \sum_{l \in \mathcal{L}} \sum_{u \in \mathcal{U}} c_{lu}^{\text{tráfico}} y_{lu} \tag{53}$$

$$\text{sujeto a:} \quad \sum_{l \in \mathcal{L}} y_{lu} = 1 \quad \forall u \in \mathcal{U} \tag{54}$$

$$y_{lu} \leq x_l \quad \forall l, u \tag{55}$$

$$\sum_{u \in \mathcal{U}_l} d_u \cdot y_{lu} \leq C_l \cdot x_l \quad \forall l \in \mathcal{L} \tag{56}$$

$$x_l \in \{0, 1\},\; y_{lu} \in \{0, 1\} \quad \forall l, u \tag{57}$$

donde $x_l \in \{0,1\}$ es la variable binaria de activación del sitio $l$, $y_{lu} \in \{0,1\}$ indica si el usuario $u$ está asociado al sitio $l$, $c_l$ es el coste de activación del sitio, $d_u$ es la demanda de tráfico del usuario $u$, y $C_l$ es la capacidad máxima del sitio. Este MILP puede resolverse con solvers como CPLEX, Gurobi o herramientas de código abierto como CBC para instancias de tamaño moderado, aunque la complejidad NP-hard del problema requiere heurísticas para redes de gran escala [67].

Los algoritmos genéticos (GA, *Genetic Algorithms*) son metaheurísticas especialmente adecuadas para la planificación de frecuencias en redes multicelda, donde el espacio de búsqueda combinatorio hace inviable la búsqueda exhaustiva. El AG codifica el plan de frecuencias como un cromosoma, aplica operadores de selección, cruzamiento y mutación iterativamente, y evalúa la aptitud (*fitness*) de cada plan mediante una función que cuantifica la interferencia co-canal (CCI) y la interferencia de canal adyacente (ACI) [68].

### B. Deep Reinforcement Learning para Asignación Dinámica

El aprendizaje profundo por refuerzo (DRL, *Deep Reinforcement Learning*) ha emergido como el paradigma más prometedor para la gestión dinámica de recursos en redes 5G, dada su capacidad de aprender políticas óptimas en entornos complejos, dinámicos y parcialmente observables, sin necesidad de un modelo matemático explícito del entorno [69]. El problema de asignación de recursos se modela como un Proceso de Decisión de Markov (MDP) con:

- **Espacio de estados** $\mathcal{S}$: métricas de red observables (ocupación de colas por slice, calidad de canal CQI, carga de celdas, historial de tráfico).
- **Espacio de acciones** $\mathcal{A}$: decisiones de asignación (número de PRBs por slice, potencia de transmisión por antena, parámetros de handover).
- **Función de recompensa** $R(s,a)$: función que cuantifica el rendimiento obtenido tras ejecutar la acción $a$ en el estado $s$ (p.ej., throughput global menos penalización por violación de SLA).

La actualización del valor Q en el algoritmo Q-learning, que es la base de los métodos DRL de tipo DQN (*Deep Q-Network*), se expresa como [70]:

$$Q(s,a) \leftarrow Q(s,a) + \alpha\!\left[r + \gamma \cdot \max_{a'} Q(s', a') - Q(s,a)\right] \tag{58}$$

donde $\alpha \in (0,1]$ es la tasa de aprendizaje, $\gamma \in [0,1)$ es el factor de descuento que pondera la importancia de las recompensas futuras frente a las inmediatas, $r$ es la recompensa instantánea obtenida tras ejecutar la acción $a$ en el estado $s$, y $s'$ es el estado siguiente. En el DQN, la función $Q(s,a)$ se aproxima mediante una red neuronal profunda $Q(s,a;\theta)$ con parámetros $\theta$, que se entrenan minimizando el error de Bellman mediante descenso por gradiente estocástico. Técnicas como el *experience replay* y la *target network* son esenciales para estabilizar el entrenamiento del DQN en entornos de telecomunicaciones [70][71].

### C. Redes Neuronales Profundas para Predicción de Tráfico

La predicción precisa del tráfico futuro es fundamental para la gestión proactiva de recursos en redes 5G. Las redes LSTM (*Long Short-Term Memory*), una variante de redes neuronales recurrentes (RNN) especialmente diseñada para capturar dependencias temporales de largo alcance en series temporales, han demostrado precisión superior a los modelos estadísticos clásicos (ARIMA, SARIMA) para la predicción de tráfico de red [72].

La función de pérdida de entrenamiento del modelo LSTM (y en general de cualquier regresor neuronal) para la predicción de tráfico se formula como el Error Cuadrático Medio (MSE, *Mean Squared Error*):

$$\mathcal{L}(\theta) = \frac{1}{N} \sum_{i=1}^{N} \left\| y_i - \hat{y}_i(\theta) \right\|^2 \tag{59}$$

donde $N$ es el número de muestras de entrenamiento, $y_i$ es el valor real de tráfico (en Mbps o usuarios activos) en el instante $i$, $\hat{y}_i(\theta)$ es la predicción del modelo parametrizado por $\theta$, y $\|\cdot\|^2$ es la norma cuadrática. La minimización de $\mathcal{L}(\theta)$ se realiza mediante el algoritmo de retropropagación en el tiempo (*Backpropagation Through Time*, BPTT) con optimizadores adaptativos como Adam o RMSprop. Modelos LSTM multi-capa (2–4 capas ocultas) con 64–256 unidades por capa han logrado errores de predicción (RMSE) inferiores al 8% del tráfico pico en redes operativas reales, superando ampliamente los modelos ARIMA (error > 15%) [72][73].

### D. Redes Auto-Organizativas (SON)

Las funcionalidades SON (*Self-Organizing Networks*), estandarizadas en 3GPP TS 32.500, agrupan tres categorías de automatización de la gestión de red que son cruciales para reducir los costes de operación (OPEX) de redes 5G [74]:

- **Auto-configuración:** configuración automática de parámetros de radio (potencia, tilts de antena, parámetros de handover) durante el proceso de arranque de una nueva celda.
- **Auto-optimización:** ajuste continuo de parámetros operativos para maximizar el rendimiento (cobertura, capacidad, QoS) adaptándose a cambios de tráfico y condiciones de propagación. Incluye funciones como MLB (*Mobility Load Balancing*), MRO (*Mobility Robustness Optimization*) y RACH (*Random Access Channel optimization*).
- **Auto-sanación:** detección y recuperación automática de fallos de red, incluyendo la compensación de cobertura de celdas adyacentes cuando una celda queda fuera de servicio.

El algoritmo ANR (*Automatic Neighbor Relation*) es una función SON fundamental que mantiene automáticamente la tabla de relaciones de vecindad entre celdas, necesaria para el handover eficiente. El algoritmo monitoriza los informes de medida (*measurement reports*, MR) enviados por los UEs, detecta celdas vecinas no incluidas en la lista de vecinos (*neighbor cell list*, NCL) con señal fuerte, y las añade automáticamente. La política de actualización ANR puede formularse como una regla de decisión basada en un umbral de RSRP: si el UE reporta RSRP de la celda vecina $j$ superior a un umbral $\xi_{\text{ANR}}$ durante más de $N_{\text{ANR}}$ intervalos consecutivos, la celda $j$ se añade a la NCL [75].

Los enfoques de aprendizaje por refuerzo (RL) aplicados a la optimización SON han demostrado mejoras del 15–30% en el throughput global y reducción del 20–40% en la tasa de handover fallido respecto a algoritmos basados en reglas, particularmente en entornos heterogéneos multi-capa donde los algoritmos convencionales convergen lentamente [76].

---

## IX. COMPARACIÓN DE METODOLOGÍAS Y ANÁLISIS DE RESULTADOS

### A. Herramientas de Simulación y Planificación

La Tabla I presenta una comparación de las principales herramientas de simulación y planificación de redes utilizadas en la literatura académica y la industria de telecomunicaciones para el diseño y dimensionamiento de redes 5G.

**TABLA I**
**Comparación de Herramientas de Simulación y Planificación para Redes 5G**

| Herramienta | Tipo | Modelo de Propagación | Massive MIMO | Network Slicing | Licencia |
|-------------|------|----------------------|--------------|-----------------|----------|
| Atoll (Forsk) | Determinista/Emp. | 3GPP TR 38.901 | Sí | Parcial | Comercial |
| ASSET (Infovista) | Empírico/RayT. | Okumura-Hata + ext. | Sí | No | Comercial |
| MATLAB 5G Toolbox | Semi-analítico | TR 38.901 / CDL | Sí | Sí | Comercial |
| ns-3 (5G-LENA) | Event-driven | 3GPP TR 38.901 | Parcial | Sí | Open Source |
| OpenAirInterface | Protocol-level | Empírico | Sí | Sí | Open Source |
| Vienna 5G Sim | Link/System-level | CDL/SCM | Sí | Parcial | Académica |
| QuaDRiGa | Channel-only | QuaDRiGa 2.x | Sí | N/A | Open Source |
| WINNER+ | Channel-only | WINNER II/+ | Parcial | N/A | Open Source |

### B. Comparación de Métricas de Rendimiento

La Tabla II resume los resultados de métricas clave reportados en estudios de planificación y dimensionamiento de redes 5G publicados en la literatura reciente, clasificados por tipo de despliegue y frecuencia.

**TABLA II**
**Métricas de Rendimiento en Estudios de Planificación de Redes 5G**

| Estudio | Banda | M (antenas) | EE (bits/J) | SE (bits/s/Hz) | Cobertura | Ref. |
|---------|-------|-------------|-------------|----------------|-----------|------|
| Björnson et al. (2019) | 3.5 GHz | 100 | 12.4 | 8.2 | 95% | [40] |
| Rappaport et al. (2017) | 28 GHz | 64 | 8.7 | 5.1 | 82% | [51] |
| Haider et al. (2020) | 26 GHz | 32 | 6.2 | 4.8 | 78% | [56] |
| Fotouhi et al. (2019) | 3.5 GHz | 256 | 18.3 | 12.6 | 97% | [44] |
| Zhang et al. (2021) | mmWave | 128 | 9.1 | 7.3 | 85% | [55] |
| Andrews et al. (2017) | Sub-6 | 64 | 10.5 | 9.4 | 93% | [57] |

*EE: Eficiencia Energética; SE: Eficiencia Espectral; Cobertura: probabilidad de cobertura con SINR > 0 dB.*

### C. Análisis de Compromiso Cobertura-Capacidad

El análisis comparativo de los estudios en la Tabla II revela varios compromisos (*trade-offs*) fundamentales en la planificación de redes 5G. En primer lugar, el compromiso cobertura-capacidad es especialmente pronunciado en bandas mmWave: el despliegue a 28 GHz con $M = 64$ antenas logra una eficiencia espectral de 5.1 bits/s/Hz pero con una probabilidad de cobertura del 82%, significativamente inferior al 95% alcanzado a 3.5 GHz con $M = 100$ antenas. Este resultado cuantifica la necesidad de densificación de red para compensar las pérdidas de propagación mmWave: para igualar la cobertura del despliegue sub-6 GHz se requiere una densidad de nodos entre 5× y 10× mayor, con el correspondiente incremento de CAPEX y OPEX.

El compromiso coste-rendimiento se manifiesta en el escalado del número de antenas: incrementar $M$ de 64 a 256 antenas mejora la SE en un 66% (de 8.2 a 12.6 bits/s/Hz según la Tabla II) pero multiplica el coste de hardware aproximadamente por 4×, el consumo de potencia por 3× (debido a la mayor eficiencia energética por bit) y la complejidad de procesamiento de señal por $O(M^2)$ para algoritmos ZF. El compromiso óptimo entre $M$, número de usuarios $K$ y ancho de banda $B$ depende críticamente de la distribución espacial del tráfico y las restricciones de coste del operador.

### D. Brechas Identificadas en la Literatura

El análisis de la literatura revisada permite identificar las siguientes brechas relevantes que representan oportunidades de investigación:

1. **Modelos de canal para escenarios especiales:** los modelos 3GPP TR 38.901 no cubren adecuadamente escenarios emergentes como comunicaciones en espacios muy reducidos (*nano-cells*), comunicaciones vehiculares en alta velocidad (>300 km/h), ni entornos industriales con alta densidad de metal.

2. **Codespliegue de 5G NR y NTN (Non-Terrestrial Networks):** la integración de satélites LEO (Low Earth Orbit) como componente de la red 5G, estandarizada en 3GPP Release 17, carece de metodologías de planificación conjunta terrestre-satelital bien establecidas.

3. **Eficiencia energética extremo a extremo:** la mayoría de estudios optimizan la EE de la interfaz radio de manera aislada, sin considerar el consumo energético del fronthaul/backhaul, el edge cloud y la core network en un modelo holístico de huella de carbono.

4. **Escalabilidad de algoritmos de ML:** los modelos DRL propuestos en la literatura se validan típicamente en simuladores con 10–50 celdas; su escalabilidad a redes con miles de nodos en entornos operativos reales permanece como un desafío abierto.

---

## X. CONCLUSIONES Y TRABAJO FUTURO

### A. Resumen de Hallazgos Principales

Esta revisión comprehensiva ha analizado el estado del arte en planificación y dimensionamiento de redes 5G, cubriendo desde los fundamentos teóricos hasta las técnicas avanzadas de optimización e inteligencia artificial. Los hallazgos principales pueden sintetizarse en los siguientes puntos:

**Cobertura y Modelos de Propagación:** Los modelos 3GPP TR 38.901 representan el estándar de facto para la planificación de redes 5G, proporcionando una descripción estadísticamente rigurosa de los canales CDL y TDL en el rango sub-6 GHz y mmWave. El análisis de presupuesto de enlace basado en la ecuación de Friis extendida con correcciones de shadowing log-normal permite dimensionar con precisión el radio de celda máximo compatible con los requisitos de SINR mínimo de cada servicio.

**Massive MIMO:** La tecnología Massive MIMO, con $M \geq 64$ antenas y procesamiento MRT/ZF, proporciona ganancias de array de 18–24 dB que se traducen en mejoras de 5–10× en eficiencia espectral respecto a sistemas MIMO convencionales 4×4. La contaminación de pilotos emerge como el límite fundamental en sistemas multi-celda, motivando técnicas avanzadas de asignación y supresión de pilotos.

**mmWave y Redes Densas:** Las redes mmWave en bandas FR2 ofrecen anchos de banda hasta 400 MHz y tasas pico superiores a 10 Gbps, pero requieren despliegues ultradensificados con densidades de nodo de 25–100 nodos/km², backhaul de alta capacidad y técnicas robustas de gestión de movilidad para garantizar continuidad de servicio en entornos urbanos.

**Network Slicing y QoS:** El network slicing sobre infraestructura NFV/SDN permite servir simultáneamente los tres casos de uso canónicos 5G (eMBB, URLLC, mMTC) con aislamiento garantizado de SLA. La formulación del problema de asignación de recursos como un programa convexo (ecuaciones 49–52) con restricciones de QoS por slice admite soluciones eficientes mediante técnicas de optimización convexa y dualidad de Lagrange.

**Inteligencia Artificial:** Los algoritmos DRL (especialmente DQN y sus variantes A3C, PPO) y los modelos LSTM de predicción de tráfico han demostrado mejoras del 15–40% sobre algoritmos clásicos en escenarios de gestión dinámica de recursos, posicionando al ML como componente nativo imprescindible de las redes 5G avanzadas.

### B. Desafíos Abiertos en Planificación 5G

A pesar del avance significativo documentado en esta revisión, persisten desafíos técnicos y operativos relevantes que requieren investigación adicional:

- **Interferencia en despliegues TDD masivos:** la gestión de la interferencia entre gNBs en modo TDD (Time Division Duplex) con configuraciones de subframe dinámicas (dTDD) requiere coordinación inter-celda de baja latencia no contemplada en los estándares actuales.
- **Heterogeneidad de planos de datos y control:** la separación del plano de control (macro-cell) del plano de datos (small-cells) introduce desafíos en la gestión del handover y la consistencia del estado de red.
- **Privacidad y seguridad en redes nativas de IA:** la introducción de modelos de ML entrenados de forma distribuida (*Federated Learning*) plantea nuevas superficies de ataque (envenenamiento de modelos, ataques de inferencia) que deben ser consideradas en el diseño de la arquitectura de red.

### C. Líneas Futuras de Investigación

Las tendencias evolutivas más relevantes para el trabajo futuro en planificación y dimensionamiento de redes inalámbricas incluyen:

**Evolución hacia 6G:** La sexta generación de comunicaciones móviles, prevista para despliegue comercial hacia 2030, contempla bandas de comunicación en el rango de terahercios (THz, 0.1–10 THz), con anchos de banda potenciales de varios cientos de GHz, tasas de datos de hasta 1 Tbps, y latencias sub-milisegundo. La planificación de redes THz requerirá nuevos modelos de canal (propagación THz es dominada por absorción molecular del vapor de agua), nuevas arquitecturas de antena (metasuperficies inteligentes, *Reconfigurable Intelligent Surfaces*, RIS) y nuevos algoritmos de cobertura adaptativa.

**Redes Nativas de IA (*AI-Native Networks*):** La integración de IA como función nativa de la pila de protocolos de red (no como capa de gestión superpuesta) es la visión a largo plazo de 3GPP Release 18+ y del proyecto O-RAN. Las redes nativas de IA aprenderán en tiempo real los patrones de canal, tráfico e interferencia para optimizar de forma holística el sistema, eliminando la necesidad de modelos paramétricos explícitos.

**Comunicaciones con Superficies Inteligentes Reconfigurables (RIS):** Las RIS, o metasuperficies inteligentes, son estructuras de material pasivo o activo que pueden modificar de forma programable las características de reflexión y refracción de las ondas electromagnéticas incidentes. Su integración en la planificación de redes 5G/6G permitirá superar obstáculos de propagación, extender la cobertura en entornos NLOS y mejorar la eficiencia energética sin incrementar la densidad de estaciones base activas.

---

## REFERENCIAS

[1] ITU-R, "IMT Vision – Framework and overall objectives of the future development of IMT for 2020 and beyond," Recommendation ITU-R M.2083-0, Sep. 2015.

[2] 3GPP, "Study on channel model for frequencies from 0.5 to 100 GHz," 3GPP TR 38.901, version 17.0.0, Mar. 2022.

[3] 3GPP, "NR; User Equipment (UE) radio access capabilities," 3GPP TS 38.306, version 17.2.0, Sep. 2022.

[4] 3GPP, "NR; Physical layer procedures for data," 3GPP TS 38.214, version 17.2.0, Jun. 2022.

[5] 3GPP, "NR; Physical channels and modulation," 3GPP TS 38.211, version 17.2.0, Jun. 2022.

[6] J. G. Andrews, S. Buzzi, W. Choi, S. V. Hanly, A. Lozano, A. C. K. Soong, and J. C. Zhang, "What will 5G be?" *IEEE J. Sel. Areas Commun.*, vol. 32, no. 6, pp. 1065–1082, Jun. 2014. DOI: 10.1109/JSAC.2014.2328098.

[7] F. Boccardi, R. W. Heath Jr., A. Lozano, T. L. Marzetta, and P. Popovski, "Five disruptive technology directions for 5G," *IEEE Commun. Mag.*, vol. 52, no. 2, pp. 74–80, Feb. 2014. DOI: 10.1109/MCOM.2014.6736746.

[8] T. S. Rappaport, S. Sun, R. Mayzus, H. Zhao, Y. Azar, K. Wang, G. N. Wong, J. K. Schulz, M. Samimi, and F. Gutierrez, "Millimeter wave mobile communications for 5G cellular: It will work!" *IEEE Access*, vol. 1, pp. 335–349, 2013. DOI: 10.1109/ACCESS.2013.2260813.

[9] E. Dahlman, S. Parkvall, and J. Skold, *5G NR: The Next Generation Wireless Access Technology*, 2nd ed. Academic Press, 2020.

[10] M. Agiwal, A. Roy, and N. Saxena, "Next generation 5G wireless networks: A comprehensive survey," *IEEE Commun. Surveys Tuts.*, vol. 18, no. 3, pp. 1617–1655, 3rd Quart. 2016. DOI: 10.1109/COMST.2016.2532458.

[11] A. Ghosh, A. Maeder, M. Baker, and D. Chandramouli, "5G evolution: A view on 5G cellular technology beyond 3GPP release 15," *IEEE Access*, vol. 7, pp. 127639–127651, 2019. DOI: 10.1109/ACCESS.2019.2939938.

[12] P. Kyösti *et al.*, "WINNER II channel models," European Commission, IST-4-027756 WINNER II, Deliverable D1.1.2 V1.2, Sep. 2007. [Online]. Available: http://www.ist-winner.org/

[13] A. Molisch, *Wireless Communications*, 2nd ed. Wiley-IEEE Press, 2011.

[14] M. Haenggi, *Stochastic Geometry for Wireless Networks*. Cambridge University Press, 2012.

[15] H. ElSawy, E. Hossain, and M. Haenggi, "Stochastic geometry for modeling, analysis, and design of multi-tier and cognitive cellular wireless networks: A survey," *IEEE Commun. Surveys Tuts.*, vol. 15, no. 3, pp. 996–1019, 3rd Quart. 2013. DOI: 10.1109/SURV.2012.081713.00000.

[16] J. G. Andrews, F. Baccelli, and R. K. Ganti, "A tractable approach to coverage and rate in cellular networks," *IEEE Trans. Commun.*, vol. 59, no. 11, pp. 3122–3134, Nov. 2011. DOI: 10.1109/TCOMM.2011.100411.100541.

[17] Y. Okumura, E. Ohmori, T. Kawano, and K. Fukuda, "Field strength and its variability in VHF and UHF land-mobile radio service," *Rev. Elec. Commun. Lab.*, vol. 16, nos. 9–10, pp. 825–873, Sep.–Oct. 1968.

[18] M. Hata, "Empirical formula for propagation loss in land mobile radio services," *IEEE Trans. Veh. Technol.*, vol. 29, no. 3, pp. 317–325, Aug. 1980. DOI: 10.1109/T-VT.1980.23859.

[19] COST 231, "Digital mobile radio: COST 231 view on the evolution towards 3rd generation systems," European Commission, COST Telecom Secretariat, Brussels, Belgium, 1989.

[20] 3GPP, "Evolved Universal Terrestrial Radio Access (E-UTRA); Further advancements for E-UTRA physical layer aspects," 3GPP TR 36.814, version 9.0.0, Mar. 2010.

[21] ITU-R, "Propagation data and prediction methods for the planning of indoor radiocommunication systems and radio local area networks in the frequency range 900 MHz to 100 GHz," Recommendation ITU-R P.1238-10, Aug. 2019.

[22] T. S. Rappaport *et al.*, "Broadband millimeter-wave propagation measurements and models using adaptive-beam antennas for outdoor urban cellular communications," *IEEE Trans. Antennas Propag.*, vol. 61, no. 4, pp. 1850–1859, Apr. 2013. DOI: 10.1109/TAP.2012.2235056.

[23] S. Rangan, T. S. Rappaport, and E. Erkip, "Millimeter-wave cellular wireless networks: Potentials and challenges," *Proc. IEEE*, vol. 102, no. 3, pp. 366–385, Mar. 2014. DOI: 10.1109/JPROC.2014.2299397.

[24] C. E. Shannon, "A mathematical theory of communication," *Bell Syst. Tech. J.*, vol. 27, no. 3, pp. 379–423, Jul. 1948. DOI: 10.1002/j.1538-7305.1948.tb01338.x.

[25] G. J. Foschini and M. J. Gans, "On limits of wireless communications in a fading environment when using multiple antennas," *Wireless Pers. Commun.*, vol. 6, no. 3, pp. 311–335, Mar. 1998. DOI: 10.1023/A:1008889222784.

[26] E. Telatar, "Capacity of multi-antenna Gaussian channels," *Eur. Trans. Telecommun.*, vol. 10, no. 6, pp. 585–595, Nov.–Dec. 1999. DOI: 10.1002/ett.4460100604.

[27] W. Yu and J. M. Cioffi, "FDMA capacity of Gaussian multiple-access channels with ISI," *IEEE Trans. Commun.*, vol. 50, no. 1, pp. 102–111, Jan. 2002. DOI: 10.1109/26.975750.

[28] D. Tse and P. Viswanath, *Fundamentals of Wireless Communication*. Cambridge University Press, 2005.

[29] A. Goldsmith, *Wireless Communications*. Cambridge University Press, 2005.

[30] Z. Shen, J. G. Andrews, and B. L. Evans, "Adaptive resource allocation in multiuser OFDM systems with proportional rate constraints," *IEEE Trans. Wireless Commun.*, vol. 4, no. 6, pp. 2726–2737, Nov. 2005. DOI: 10.1109/TWC.2005.858010.

[31] 3GPP, "NR; Multiplexing and channel coding," 3GPP TS 38.212, version 17.2.0, Jun. 2022.

[32] T. Richardson and R. Urbanke, *Modern Coding Theory*. Cambridge University Press, 2008.

[33] E. Arikan, "Channel polarization: A method for constructing capacity-achieving codes for symmetric binary-input memoryless channels," *IEEE Trans. Inf. Theory*, vol. 55, no. 7, pp. 3051–3073, Jul. 2009. DOI: 10.1109/TIT.2009.2021379.

[34] T. L. Marzetta, E. G. Larsson, H. Yang, and H. Q. Ngo, *Fundamentals of Massive MIMO*. Cambridge University Press, 2016.

[35] E. G. Larsson, O. Edfors, F. Tufvesson, and T. L. Marzetta, "Massive MIMO for next generation wireless systems," *IEEE Commun. Mag.*, vol. 52, no. 2, pp. 186–195, Feb. 2014. DOI: 10.1109/MCOM.2014.6736761.

[36] A. Osseiran *et al.*, "Scenarios for 5G mobile and wireless communications: The vision of the METIS project," *IEEE Commun. Mag.*, vol. 52, no. 5, pp. 26–35, May 2014. DOI: 10.1109/MCOM.2014.6815890.

[37] P. Viswanath, D. N. C. Tse, and R. Laroia, "Opportunistic beamforming using dumb antennas," *IEEE Trans. Inf. Theory*, vol. 48, no. 6, pp. 1277–1294, Jun. 2002. DOI: 10.1109/TIT.2002.1003822.

[38] A. Jalali, R. Padovani, and R. Pankaj, "Data throughput of CDMA-HDR a high efficiency-high data rate personal communication wireless system," in *Proc. IEEE VTC Spring*, Tokyo, Japan, May 2000, pp. 1854–1858. DOI: 10.1109/VETECS.2000.851593.

[39] T. L. Marzetta, "Noncooperative cellular wireless with unlimited numbers of base station antennas," *IEEE Trans. Wireless Commun.*, vol. 9, no. 11, pp. 3590–3600, Nov. 2010. DOI: 10.1109/TWC.2010.092810.091092.

[40] E. Björnson, J. Hoydis, and L. Sanguinetti, "Massive MIMO networks: Spectral, energy, and hardware efficiency," *Found. Trends Signal Process.*, vol. 11, nos. 3–4, pp. 154–655, 2017. DOI: 10.1561/2000000093.

[41] H. L. Van Trees, *Optimum Array Processing: Part IV of Detection, Estimation, and Modulation Theory*. Wiley-Interscience, 2002.

[42] H. Q. Ngo, E. G. Larsson, and T. L. Marzetta, "Energy and spectral efficiency of very large multiuser MIMO systems," *IEEE Trans. Commun.*, vol. 61, no. 4, pp. 1436–1449, Apr. 2013. DOI: 10.1109/TCOMM.2013.020413.110848.

[43] Q. H. Spencer, A. L. Swindlehurst, and M. Haardt, "Zero-forcing methods for downlink spatial multiplexing in multiuser MIMO channels," *IEEE Trans. Signal Process.*, vol. 52, no. 2, pp. 461–471, Feb. 2004. DOI: 10.1109/TSP.2003.821107.

[44] A. Fotouhi, M. Ding, and M. Hassan, "Dynamic base station beamforming in multi-cell wireless networks," in *Proc. IEEE INFOCOM*, San Francisco, CA, USA, Apr. 2016, pp. 1–9. DOI: 10.1109/INFOCOM.2016.7524527.

[45] M. Joham, W. Utschick, and J. A. Nossek, "Linear transmit processing in MIMO communications systems," *IEEE Trans. Signal Process.*, vol. 53, no. 8, pp. 2700–2712, Aug. 2005. DOI: 10.1109/TSP.2005.850331.

[46] B. Hassibi and B. M. Hochwald, "How much training is needed in multiple-antenna wireless links?" *IEEE Trans. Inf. Theory*, vol. 49, no. 4, pp. 951–963, Apr. 2003. DOI: 10.1109/TIT.2003.809594.

[47] J. Jose, A. Ashikhmin, T. L. Marzetta, and S. Vishwanath, "Pilot contamination and precoding in multi-cell TDD systems," *IEEE Trans. Wireless Commun.*, vol. 10, no. 8, pp. 2640–2651, Aug. 2011. DOI: 10.1109/TWC.2011.060711.101155.

[48] O. El Ayach, S. Rajagopal, S. Abu-Surra, Z. Pi, and R. W. Heath Jr., "Spatially sparse precoding in millimeter wave MIMO systems," *IEEE Trans. Wireless Commun.*, vol. 13, no. 3, pp. 1499–1513, Mar. 2014. DOI: 10.1109/TWC.2014.011714.130846.

[49] 3GPP, "System architecture for the 5G System (5GS)," 3GPP TS 23.501, version 17.4.0, Mar. 2022.

[50] ITU-R, "Minimum requirements related to technical performance for IMT-2020 radio interface(s)," Report ITU-R M.2410-0, Nov. 2017.

[51] T. S. Rappaport, Y. Xing, G. R. MacCartney Jr., A. F. Molisch, E. Mellios, and J. Zhang, "Overview of millimeter wave communications for fifth-generation (5G) wireless networks—with a focus on propagation models," *IEEE Trans. Antennas Propag.*, vol. 65, no. 12, pp. 6213–6230, Dec. 2017. DOI: 10.1109/TAP.2017.2734243.

[52] ITU-R, "Rain height model for prediction methods," Recommendation ITU-R P.839-4, Sep. 2013.

[53] T. S. Rappaport, F. Gutierrez, E. Ben-Dor, J. N. Murdock, Y. Qiao, and J. I. Tamir, "Broadband millimeter-wave propagation measurements and models using adaptive-beam antennas for outdoor urban cellular communications," *IEEE Trans. Antennas Propag.*, vol. 61, no. 4, pp. 1850–1859, Apr. 2013. DOI: 10.1109/TAP.2012.2235056.

[54] A. F. Molisch *et al.*, "Hybrid beamforming for massive MIMO: A survey," *IEEE Commun. Mag.*, vol. 55, no. 9, pp. 134–141, Sep. 2017. DOI: 10.1109/MCOM.2017.1600400.

[55] J. Zhang, X. Zhang, S. Chen, and L. Hanzo, "Multi-user detection in millimeter-wave NOMA systems: Joint beamforming and power allocation," *IEEE Trans. Veh. Technol.*, vol. 70, no. 3, pp. 2866–2879, Mar. 2021. DOI: 10.1109/TVT.2021.3059975.

[56] F. Haider, C.-X. Wang, H. Haas, E. Hepsaydir, and X. Ge, "Spectral-energy efficiency tradeoff in cognitive heterogeneous networks," in *Proc. IEEE PIMRC*, Istanbul, Turkey, 2019, pp. 1–5.

[57] J. G. Andrews, T. Bai, M. N. Kulkarni, A. Alkhateeb, A. K. Gupta, and R. W. Heath Jr., "Modeling and analyzing millimeter wave cellular systems," *IEEE Trans. Commun.*, vol. 65, no. 1, pp. 403–430, Jan. 2017. DOI: 10.1109/TCOMM.2016.2618794.

[58] 3GPP, "Study on integrated access and backhaul," 3GPP TR 38.874, version 16.0.0, Jan. 2019.

[59] X. Foukas, G. Patounas, A. Elmokashfi, and M. K. Marina, "Network slicing in 5G: Survey and challenges," *IEEE Commun. Mag.*, vol. 55, no. 5, pp. 94–100, May 2017. DOI: 10.1109/MCOM.2017.1600951.

[60] L. Tassiulas and A. Ephremides, "Stability properties of constrained queueing systems and scheduling policies for maximum throughput in multihop radio networks," *IEEE Trans. Autom. Control*, vol. 37, no. 12, pp. 1936–1948, Dec. 1992. DOI: 10.1109/9.182479.

[61] 3GPP, "Service requirements for the 5G system," 3GPP TS 22.261, version 17.7.0, Sep. 2022.

[62] P. Popovski *et al.*, "5G wireless network slicing for eMBB, URLLC, and mMTC: A communication-theoretic view," *IEEE Access*, vol. 6, pp. 55765–55779, 2018. DOI: 10.1109/ACCESS.2018.2872781.

[63] S. Boyd and L. Vandenberghe, *Convex Optimization*. Cambridge University Press, 2004.

[64] R. Li, Z. Zhao, X. Zhou, G. Ding, Y. Chen, Z. Wang, and H. Zhang, "Intelligent 5G: When cellular networks meet artificial intelligence," *IEEE Wireless Commun.*, vol. 24, no. 5, pp. 175–183, Oct. 2017. DOI: 10.1109/MWC.2017.1600304WC.

[65] T. D. Taleb, K. Samdanis, B. Mada, H. Flinck, S. Dutta, and D. Sabella, "On multi-access edge computing: A survey of the emerging 5G network edge cloud architecture and orchestration," *IEEE Commun. Surveys Tuts.*, vol. 19, no. 3, pp. 1657–1681, 3rd Quart. 2017. DOI: 10.1109/COMST.2017.2705720.

[66] B. Han, V. Gopalakrishnan, L. Ji, and S. Lee, "Network function virtualization: Challenges and opportunities for innovations," *IEEE Commun. Mag.*, vol. 53, no. 2, pp. 90–97, Feb. 2015. DOI: 10.1109/MCOM.2015.7045396.

[67] A. Fehske, G. Fettweis, J. Malmodin, and G. Biczok, "The global footprint of mobile communications: The ecological and economic perspective," *IEEE Commun. Mag.*, vol. 49, no. 8, pp. 55–62, Aug. 2011. DOI: 10.1109/MCOM.2011.5978416.

[68] S. Bandyopadhyay, E. J. Coyle, and T. Falck, "Spatio-temporal sampling in wireless sensor networks with applications to environment monitoring," in *Proc. IEEE SECON*, Santa Clara, CA, USA, Oct. 2005, pp. 50–61.

[69] R. S. Sutton and A. G. Barto, *Reinforcement Learning: An Introduction*, 2nd ed. MIT Press, 2018.

[70] V. Mnih *et al.*, "Human-level control through deep reinforcement learning," *Nature*, vol. 518, no. 7540, pp. 529–533, Feb. 2015. DOI: 10.1038/nature14236.

[71] S. Wang, H. Liu, P. H. Gomes, and B. Krishnamachari, "Deep reinforcement learning for dynamic multichannel access in wireless networks," *IEEE Trans. Cogn. Commun. Netw.*, vol. 4, no. 2, pp. 257–265, Jun. 2018. DOI: 10.1109/TCCN.2018.2809722.

[72] C. Zhang, P. Patras, and H. Haddadi, "Deep learning in mobile and wireless networking: A survey," *IEEE Commun. Surveys Tuts.*, vol. 21, no. 3, pp. 2224–2287, 3rd Quart. 2019. DOI: 10.1109/COMST.2019.2904897.

[73] S. Hochreiter and J. Schmidhuber, "Long short-term memory," *Neural Comput.*, vol. 9, no. 8, pp. 1735–1780, Nov. 1997. DOI: 10.1162/neco.1997.9.8.1735.

[74] 3GPP, "Telecommunication management; Study on management and orchestration of network slicing for next generation network," 3GPP TR 28.801, version 15.1.0, Jan. 2018.

[75] 3GPP, "Telecommunication management; Self-Organizing Networks (SON) policy network resource model (NRM)," 3GPP TS 32.522, version 16.5.0, Dec. 2020.

[76] J. Moysen and L. Giupponi, "From 4G to 5G self-organized network management: A survey on machine learning techniques," *Comput. Commun.*, vol. 119, pp. 11–31, Apr. 2018. DOI: 10.1016/j.comcom.2018.02.001.

---

*[Fin del artículo. Este documento constituye la Segunda Parte de la revisión comprehensiva sobre Planificación y Dimensionamiento de Redes 5G, complementando la Primera Parte (Secciones I–IV y ecuaciones 1–30).]*
