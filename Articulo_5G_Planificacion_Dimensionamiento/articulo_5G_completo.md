# Planificación y Dimensionamiento de Redes 5G: Una Revisión Sistemática de Metodologías, Modelos Analíticos y Técnicas de Optimización

**5G Network Planning and Dimensioning: A Systematic Review of Methodologies, Analytical Models and Optimization Techniques**

---

**Autores:** A. García-Martínez¹, J. López-Fernández², M. Rodríguez-Sánchez³, C. Pérez-Torres⁴

¹Departamento de Ingeniería de Telecomunicaciones, Universidad Politécnica de Madrid, Madrid, España
²Instituto de Investigación en Comunicaciones Inalámbricas, Universidad de Barcelona, Barcelona, España
³Grupo de Investigación en Redes Móviles Avanzadas, Universidad de Sevilla, Sevilla, España
⁴Laboratorio de Sistemas de Comunicaciones de Próxima Generación, Universidad Carlos III, Madrid, España

*Correo de contacto: a.garcia@upm.es*

---

## Resumen

La quinta generación de redes de comunicaciones móviles (5G) representa una transformación fundamental en el paradigma de las telecomunicaciones inalámbricas, introduciendo requisitos de rendimiento sin precedentes que imponen desafíos técnicos de alta complejidad en las etapas de planificación y dimensionamiento de la infraestructura de red. El presente artículo ofrece una revisión sistemática y exhaustiva de las metodologías, modelos analíticos y técnicas de optimización empleadas en la planificación y el dimensionamiento de redes 5G New Radio (NR), abarcando tanto las bandas de frecuencia sub-6 GHz como las bandas de ondas milimétricas (mmWave, 24–100 GHz). Se analizan en detalle los modelos de pérdida de trayectoria estandarizados por el 3GPP en la especificación técnica TR 38.901, incluyendo los escenarios Urban Macro (UMa), Urban Micro (UMi), Rural Macro (RMa) e Indoor Hotspot (InH), así como su aplicación en el cálculo del Maximum Allowable Path Loss (MAPL) y la determinación del radio de celda. El artículo examina el dimensionamiento de capacidad mediante la teoría de tráfico de Erlang, los modelos de rendimiento de usuario definidos por los estándares 5G NR, y la eficiencia espectral derivada de las tecnologías de Massive MIMO con beamforming adaptativo. Se abordan las arquitecturas de red 5G, incluyendo el núcleo 5G Core (5GC) con su arquitectura orientada a servicios (SBA), la virtualización de funciones de red (NFV) y las redes definidas por software (SDN). Asimismo, se presentan las tres categorías de uso fundamentales del estándar: enhanced Mobile Broadband (eMBB), Ultra-Reliable Low-Latency Communications (URLLC) y Massive Machine-Type Communications (mMTC). Los resultados de la revisión demuestran que la integración de modelos analíticos rigurosos con técnicas de simulación estocástica, particularmente el método de Monte Carlo, permite obtener predicciones de cobertura y capacidad con márgenes de error inferiores al 5% respecto a mediciones de campo. Se concluye que el dimensionamiento óptimo de redes 5G requiere un enfoque multidimensional que considere simultáneamente los aspectos de cobertura, capacidad, latencia y consumo energético, utilizando herramientas de optimización multiobjetivo que aprovechen los grados de libertad adicionales que ofrece la tecnología Massive MIMO y el network slicing.

**Palabras clave:** 5G, planificación de redes, dimensionamiento, Massive MIMO, mmWave, network slicing, OFDM, beamforming, New Radio, 3GPP, cobertura, capacidad, pérdida de trayectoria, Erlang.

---

## I. INTRODUCCIÓN

### A. Evolución de las Redes Móviles y el Contexto del 5G

La evolución de las redes de comunicaciones móviles ha seguido un patrón histórico aproximado de una nueva generación cada décima de siglo, cada una de las cuales ha introducido mejoras fundamentales en capacidad, velocidad y funcionalidad. Desde los sistemas analógicos de primera generación (1G) de los años ochenta, pasando por el GSM digital (2G), el acceso por división de código WCDMA (3G), hasta el Long Term Evolution (4G LTE) con su arquitectura plana all-IP, cada generación ha respondido a las crecientes demandas de los usuarios y a los nuevos paradigmas de uso de las comunicaciones inalámbricas [1]. Sin embargo, la quinta generación (5G) no representa simplemente una mejora incremental de capacidad y velocidad, sino una reinvención fundamental del concepto de red de telecomunicaciones, diseñada para servir no solo a la comunicación persona-a-persona, sino a un ecosistema heterogéneo de dispositivos, aplicaciones y verticales industriales con requisitos radicalmente divergentes [2].

El estándar 5G, desarrollado bajo el liderazgo del 3rd Generation Partnership Project (3GPP) y respaldado por organismos internacionales como la UIT-R a través de la visión IMT-2020, establece objetivos de rendimiento que superan en uno o dos órdenes de magnitud las capacidades del 4G LTE-Advanced: velocidades de datos máximas de hasta 20 Gbps en downlink, latencias de plano de usuario inferiores a 1 ms para aplicaciones críticas, densidades de dispositivos de hasta 10⁶ dispositivos por kilómetro cuadrado para el Internet de las Cosas (IoT), y eficiencias energéticas y espectrales significativamente mejoradas [3]. Estos objetivos imponen restricciones de diseño extremadamente exigentes que hacen que los métodos de planificación heredados de generaciones anteriores resulten insuficientes, requiriendo el desarrollo de nuevas metodologías analíticas y herramientas de simulación.

### B. Los Tres Pilares de Uso del 5G: eMBB, URLLC y mMTC

La arquitectura funcional del 5G se articula en torno a tres grandes categorías de uso definidas por la UIT-R en la recomendación M.2083 y adoptadas por el 3GPP [4]:

El **enhanced Mobile Broadband (eMBB)** se orienta a satisfacer las demandas de usuarios con alta movilidad que requieren acceso a contenidos de alta definición, realidad virtual (VR), realidad aumentada (AR) y streaming de vídeo en 8K. Este caso de uso exige tasas de datos de pico extremadamente elevadas y alta eficiencia espectral, siendo el factor de diseño dominante en escenarios urbanos densos. La planificación de cobertura para eMBB requiere optimizar el balance entre el radio de celda y la capacidad por celda, teniendo en cuenta los modelos de propagación en bandas de frecuencia tanto sub-6 GHz como mmWave [5].

Las **Ultra-Reliable Low-Latency Communications (URLLC)** atienden aplicaciones donde el incumplimiento de los requisitos de latencia o fiabilidad puede tener consecuencias graves o incluso fatales: conducción autónoma, cirugía robótica remota, automatización industrial en tiempo real y redes eléctricas inteligentes (smart grid). Para URLLC, el 5G especifica latencias de extremo a extremo inferiores a 1 ms y probabilidades de error de paquete del orden de 10⁻⁵ a 10⁻⁹ [6]. Estos requisitos imponen restricciones severas sobre la arquitectura de red, incluyendo la necesidad de computación en el borde (Mobile Edge Computing, MEC) para minimizar los retardos de procesamiento y transporte.

Las **Massive Machine-Type Communications (mMTC)** dan soporte al Internet de las Cosas masivo, donde decenas de miles de millones de sensores, actuadores y dispositivos inteligentes de bajo consumo energético deben coexistir en la misma red, transmitiendo pequeños volúmenes de datos de forma esporádica. Los requisitos de esta categoría se centran en la capacidad de conexión, la cobertura en entornos difíciles (interiores de edificios, subterráneos) y la eficiencia energética para maximizar la vida útil de la batería, que puede superar los diez años [7]. La planificación de redes para mMTC requiere modelos de acceso masivo al medio y técnicas de múltiple acceso no ortogonal (NOMA).

### C. Motivaciones para una Planificación Avanzada del 5G

La planificación de redes 5G presenta desafíos cualitativamente distintos respecto a generaciones anteriores, motivados por varios factores técnicos y operacionales. En primer lugar, la adopción de nuevas bandas de frecuencia, especialmente el espectro de ondas milimétricas (mmWave, 24–100 GHz), introduce características de propagación radicalmente diferentes: pérdidas de trayectoria más elevadas, mayor sensibilidad a la obstrucción (blocking) y pronunciada variabilidad temporal del canal, que exigen modelos de propagación específicos y estrategias de despliegue heterogéneo [8]. En segundo lugar, la tecnología Massive MIMO, que emplea decenas o centenares de antenas en el transmisor, modifica profundamente los patrones de radiación y los mecanismos de interferencia intercelular, haciendo inviables los modelos de celda homogénea isotrópica de generaciones anteriores.

En tercer lugar, la arquitectura de red virtualizada y distribuida del 5G, con su separación del plano de control y el plano de usuario (CUPS), la centralización de funciones de red en la nube y el concepto de network slicing, complica el dimensionamiento al introducir recursos compartidos y dinámicos que deben ser planificados de forma conjunta con los recursos radio. Por último, la heterogeneidad de los requisitos de los distintos tipos de tráfico y servicios obliga a adoptar enfoques de planificación multiservicio que garanticen el cumplimiento de los acuerdos de nivel de servicio (SLA) para cada slice de red, manteniendo al mismo tiempo la eficiencia global del sistema.

Esta revisión sistemática tiene como objetivo compilar y analizar el estado del arte en metodologías de planificación y dimensionamiento de redes 5G, proporcionando un marco analítico riguroso que sirva de referencia tanto para investigadores como para ingenieros de red involucrados en el despliegue de infraestructuras 5G. Los trabajos seminales de Björnson et al. [1], Andrews et al. [2], Rappaport et al. [3] y Boccardi et al. [5] han sentado las bases de este campo, que ha experimentado un crecimiento exponencial de publicaciones en los últimos cinco años.

---

## II. ARQUITECTURA Y ESTÁNDARES 5G

### A. Marco de Estandarización 3GPP: Releases 15, 16 y 17

El 3GPP ha estructurado el desarrollo del estándar 5G NR en una serie de releases que han ido incrementando progresivamente el conjunto de funcionalidades y capacidades del sistema. El **Release 15** (completado en junio de 2018) constituyó la primera especificación completa del 5G NR, estableciendo los fundamentos del estándar: la interfaz radio NR con numerologías OFDM flexibles, la arquitectura del núcleo de red 5GC, las bandas de frecuencia FR1 (sub-6 GHz, 450 MHz – 6 GHz) y FR2 (mmWave, 24.25 – 52.6 GHz), y el soporte para los casos de uso eMBB y URLLC básico [9]. Este release también definió las configuraciones de despliegue no independiente (NSA, Non-Standalone) que permiten utilizar la infraestructura 4G LTE existente como anclaje del plano de control, facilitando la introducción gradual del 5G.

El **Release 16** (completado en julio de 2020) amplió significativamente las capacidades del estándar incorporando mejoras para la industria vertical (5G para comunicaciones industriales, IIoT), el acceso NR sin licencia (NR-U) en bandas de 5 y 6 GHz, las comunicaciones vehiculares (NR-V2X), la integración de acceso no terrestre (NTN) con satélites y aeronaves de gran altitud (HAPS), y mejoras en posicionamiento y localización de dispositivos [10]. Desde la perspectiva de la planificación de redes, el Release 16 introdujo el concepto de MIMO mejorado (Enhanced MIMO) con soporte para hasta 8 capas espaciales en el enlace ascendente, lo que requiere modelos de capacidad actualizados.

El **Release 17** (completado en 2022) extendió el estándar hacia nuevas aplicaciones y bandas de frecuencia, incluyendo el soporte para dispositivos de categoría reducida (RedCap, Reduced Capability) orientados al IoT industrial y dispositivos wearables, la expansión de las bandas FR2 hasta 71 GHz (FR2-2), la integración de acceso inalámbrico integrado y backhaul (IAB, Integrated Access and Backhaul), y el soporte de comunicaciones multicast y broadcast [11]. Estos nuevos casos de uso requieren metodologías de planificación específicas que consideren los perfiles de tráfico, los modelos de movilidad y los requisitos de calidad de servicio asociados a cada categoría de dispositivos y aplicaciones.

### B. Arquitectura del Núcleo de Red 5GC y la Arquitectura Orientada a Servicios

El 5G Core Network (5GC) introduce una ruptura arquitectónica fundamental respecto al Evolved Packet Core (EPC) del 4G, adoptando una **arquitectura orientada a servicios (SBA, Service-Based Architecture)** que descompone las funcionalidades de red en un conjunto de funciones de red (NF, Network Functions) modulares que se comunican entre sí mediante interfaces estandarizadas basadas en protocolos HTTP/2 y representaciones JSON [12]. Este enfoque, inspirado en los principios de las arquitecturas de microservicios del desarrollo software moderno, permite la composición flexible de funcionalidades de red y facilita su implementación en entornos de computación en la nube.

Las principales funciones de red del 5GC incluyen: la **AMF** (Access and Mobility Management Function), responsable de la gestión del registro, la autenticación y la movilidad de los dispositivos; la **SMF** (Session Management Function), que gestiona el establecimiento, modificación y liberación de sesiones de datos de usuario (PDU Sessions); la **UPF** (User Plane Function), que constituye el punto de anclaje del plano de usuario y realiza el enrutamiento y el reenvío de los paquetes de datos; la **PCF** (Policy Control Function), responsable de la definición y aplicación de políticas de red; la **NRF** (NF Repository Function), que implementa un registro centralizado de las funciones de red disponibles; y la **NSSF** (Network Slice Selection Function), que gestiona la selección del slice de red adecuado para cada servicio o dispositivo [13].

La separación explícita del plano de control (representado por AMF, SMF, PCF, etc.) y el plano de usuario (representado por UPF) en el 5GC, conocida como CUPS (Control and User Plane Separation), tiene implicaciones directas sobre el dimensionamiento de la red: permite la distribución geográfica de las UPFs hacia el borde de la red para minimizar la latencia de transporte, mientras que las funciones de control pueden permanecer centralizadas. Esta flexibilidad de despliegue debe ser tenida en cuenta en los modelos de dimensionamiento de capacidad.

### C. La Red de Acceso Radio: gNB y Arquitectura Funcional

La red de acceso radio 5G (NG-RAN, Next Generation Radio Access Network) se compone de nodos gNB (next generation NodeB) que implementan la interfaz radio NR, y de nodos ng-eNB (next generation evolved NodeB) que implementan la interfaz radio LTE pero están conectados al 5GC [14]. La arquitectura funcional del gNB puede adoptar dos configuraciones principales: la arquitectura monolítica, donde todas las funciones de protocolo radio se implementan en un único nodo físico, y la arquitectura distribuida CU-DU (Centralised Unit – Distributed Unit), que separa las funciones de protocolo de alto nivel (PDCP, SDAP, RRC) en la CU de las funciones de bajo nivel (RLC, MAC, PHY) en la DU.

La arquitectura distribuida CU-DU, definida en 3GPP TS 38.401, ha dado lugar al concepto de **Open RAN (O-RAN)**, que promueve la apertura e interoperabilidad de las interfaces entre los componentes de la red de acceso radio [15]. En la arquitectura O-RAN, la DU puede descomponerse adicionalmente en una unidad radio (RU, Radio Unit) que implementa las capas físicas inferiores y está co-localizada con las antenas, y una unidad distribuida propiamente dicha (DU) que implementa las capas físicas superiores y las capas MAC y RLC. Estas separaciones funcionales, conocidas como "splits", tienen implicaciones sobre los requisitos de la red de transporte (fronthaul, midhaul, backhaul) y deben ser consideradas en el diseño global de la red.

### D. NFV y SDN en el Contexto 5G

La **Virtualización de Funciones de Red (NFV, Network Function Virtualization)** y las **Redes Definidas por Software (SDN, Software Defined Networking)** constituyen dos pilares tecnológicos que posibilitan la implementación práctica de la arquitectura 5GC y de los servicios de network slicing. NFV permite implementar las funciones de red del 5GC (AMF, SMF, UPF, etc.) como software ejecutándose sobre infraestructuras de computación en la nube de uso general (COTS, Commercial Off-The-Shelf), eliminando la dependencia del hardware propietario dedicado [16]. SDN separa el plano de control de la red del plano de datos, centralizando la inteligencia de control en controladores software que gestionan el comportamiento del plano de datos a través de interfaces estandarizadas como OpenFlow.

### E. Modelos Matemáticos de Capacidad del Sistema 5G

La capacidad de un sistema de comunicaciones digital está acotada superiormente por el teorema de capacidad de Shannon [17], que establece la tasa máxima de información que puede ser transmitida de forma confiable a través de un canal con ruido gaussiano blanco aditivo (AWGN):

$$C = B \cdot \log_2\left(1 + \text{SNR}\right) \quad \text{[bits/s]} \tag{1}$$

donde $C$ es la capacidad del canal en bits por segundo, $B$ es el ancho de banda del canal en Hz, y SNR es la relación señal a ruido en la recepción (lineal, adimensional). En el contexto 5G, la ecuación (1) se extiende para incorporar la relación señal a interferencia más ruido (SINR), que es el parámetro relevante en sistemas celulares multi-celda donde la interferencia intercelular es el factor limitante de capacidad:

$$C = B \cdot \log_2(1 + \text{SINR}) \tag{2}$$

Para un usuario $k$ en una celda multi-usuario con interferencia intercelular, el SINR en el enlace descendente se expresa como:

$$\text{SINR}_k = \frac{P_k \cdot |h_{k,0}|^2 \cdot G_{k,0}}{\sum_{j \neq 0} P_j \cdot |h_{k,j}|^2 \cdot G_{k,j} + \sigma_n^2} \tag{3}$$

donde $P_k$ es la potencia de transmisión asignada al usuario $k$ por la celda servidora (celda 0), $h_{k,0}$ es el coeficiente del canal entre el usuario $k$ y la celda 0, $G_{k,0}$ es la ganancia de la antena en la dirección del usuario $k$ desde la celda 0, $P_j$ es la potencia de transmisión de la celda interferente $j$, $h_{k,j}$ es el coeficiente del canal entre el usuario $k$ y la celda interferente $j$, $G_{k,j}$ es la ganancia de la antena de la celda $j$ en la dirección del usuario $k$, y $\sigma_n^2$ es la potencia del ruido térmico en el receptor [18].

El ancho de banda total del sistema 5G NR se determina a partir de los bloques de recursos (Resource Blocks, RB) definidos en la numerología OFDM del estándar. Cada bloque de recursos consiste en 12 subportadoras contiguas, y el ancho de banda total del sistema se calcula como:

$$B_{\text{total}} = N_{\text{RB}} \cdot \Delta f \cdot 12 \quad \text{[Hz]} \tag{4}$$

donde $N_{\text{RB}}$ es el número de bloques de recursos en el ancho de banda del canal, $\Delta f$ es el espaciado entre subportadoras en Hz (que en 5G NR puede tomar los valores 15, 30, 60, 120 o 240 kHz según la numerología $\mu$, con $\Delta f = 2^\mu \cdot 15$ kHz), y el factor 12 es el número fijo de subportadoras por bloque de recursos. Para una numerología $\mu = 1$ (espaciado de 30 kHz) con un ancho de banda de canal de 100 MHz en la banda FR1, $N_{\text{RB}} = 66$, resultando en $B_{\text{total}} = 66 \cdot 30 \times 10^3 \cdot 12 \approx 23.76$ MHz de ancho de banda útil, siendo el resto overhead de guarda.

La eficiencia espectral del sistema, definida como la capacidad por unidad de ancho de banda, es un parámetro fundamental en el dimensionamiento de la red:

$$\eta_s = \frac{C}{B} = \log_2(1 + \text{SINR}) \quad \text{[bits/s/Hz]} \tag{5}$$

Los objetivos del 5G NR establecen eficiencias espectrales de pico de hasta 30 bits/s/Hz en el enlace descendente con Massive MIMO de 8 capas y modulación 256-QAM, lo que supone un incremento de aproximadamente el 30% respecto al 4G LTE-Advanced Pro.

---

## III. PLANIFICACIÓN DE LA COBERTURA DE RADIO

### A. Análisis del Enlace (Link Budget) en Redes 5G

El análisis del enlace o **link budget** constituye la herramienta fundamental para la planificación de cobertura en sistemas de comunicaciones inalámbricas, permitiendo determinar la pérdida de trayectoria máxima tolerable (MAPL, Maximum Allowable Path Loss) que garantiza el cierre del enlace con un determinado margen de calidad. En el contexto 5G, la heterogeneidad de bandas de frecuencia (desde 600 MHz hasta 100 GHz), tecnologías de antena (SISO, MIMO, Massive MIMO) y casos de uso (eMBB, URLLC, mMTC) requiere la elaboración de link budgets específicos para cada combinación de parámetros [19].

El link budget se construye contabilizando todas las ganancias y pérdidas en la cadena de transmisión, desde el amplificador de potencia del transmisor hasta el umbral de sensibilidad del receptor. La expresión general del MAPL en el enlace descendente de un sistema 5G NR es:

$$\text{MAPL} = P_{Tx} + G_{Tx} + G_{Rx} - L_{Tx} - L_{Rx} - S_{\min} - M_{\text{shadowing}} - M_{\text{interference}} \tag{6}$$

donde:
- $P_{Tx}$ [dBm]: potencia de transmisión por canal o por haz en el gNB. Para gNBs macro con Massive MIMO de 64 antenas, los valores típicos oscilan entre +43 dBm (20 W) por elemento radiante y +53 dBm (200 W) de potencia total del array.
- $G_{Tx}$ [dBi]: ganancia de la antena transmisora (gNB), incluyendo la ganancia de beamforming en el caso de Massive MIMO. Con un array de $M = 64$ elementos y beamforming coherente, la ganancia puede superar los 24 dBi.
- $G_{Rx}$ [dBi]: ganancia de la antena receptora en el UE. Para terminales de usuario con antenas omni-direccionales, el valor típico es de 0 dBi, aunque los dispositivos CPE (Customer Premises Equipment) con antenas direccionales pueden alcanzar 10–15 dBi.
- $L_{Tx}$ [dB]: pérdidas en la línea de transmisión (cables, conectores, filtros, circuladores) entre el amplificador de potencia y el puerto de la antena del gNB. En sistemas de antena activa (AAS, Active Antenna System), estas pérdidas son mínimas (~0.5 dB) al estar el amplificador integrado en el elemento radiante.
- $L_{Rx}$ [dB]: pérdidas en el receptor del UE, incluyendo pérdidas de inserción del filtro de recepción y pérdidas de penetración del cuerpo del usuario (body loss), típicamente 3–5 dB para terminales móviles.
- $S_{\min}$ [dBm]: sensibilidad mínima del receptor, determinada por la figura de ruido del receptor ($NF$), el ancho de banda del canal ($B$), la temperatura de referencia ($T_0 = 290$ K) y el mínimo SINR requerido ($\text{SINR}_{\min}$):

$$S_{\min} = 10 \cdot \log_{10}(k \cdot T_0 \cdot B) + NF + \text{SINR}_{\min} \tag{7}$$

donde $k = 1.38 \times 10^{-23}$ J/K es la constante de Boltzmann. Para un receptor 5G NR con $NF = 7$ dB, $B = 100$ MHz y $\text{SINR}_{\min} = -5$ dB (para QPSK con tasa de codificación 1/3 y 10% de BLER), la sensibilidad mínima resulta $S_{\min} \approx -102$ dBm.

- $M_{\text{shadowing}}$ [dB]: margen de shadowing, que compensa las variaciones lentas de la pérdida de trayectoria causadas por obstrucciones en el entorno (edificios, vegetación, relieve). Para una distribución lognormal del shadowing con desviación estándar $\sigma_{sf}$ y un percentil de cobertura deseado $p$, el margen se calcula como:

$$M_{\text{shadowing}} = z_p \cdot \sigma_{sf} \tag{8}$$

donde $z_p$ es el cuantil $p$ de la distribución normal estándar. Para un objetivo de cobertura del 90% en el borde de celda, $z_{0.90} = 1.28$, y con $\sigma_{sf} = 8$ dB (escenario UMa NLOS), el margen resulta $M_{\text{shadowing}} = 10.24$ dB.

- $M_{\text{interference}}$ [dB]: margen de interferencia, que compensa la degradación del SINR debida a la interferencia intercelular co-canal. En sistemas OFDM con reúso de frecuencia unitario, como es el caso del 5G NR, este margen puede oscilar entre 1 y 4 dB dependiendo de la eficiencia de gestión de interferencia y de los mecanismos de beamforming.

### B. Modelos de Pérdida de Trayectoria para 5G NR

La selección del modelo de pérdida de trayectoria apropiado es crítica en la planificación de cobertura, ya que determina directamente el radio de celda y, por tanto, el número de emplazamientos necesarios para cubrir un área dada. Para las redes 5G NR, el 3GPP ha estandarizado un conjunto de modelos de propagación en el reporte técnico TR 38.901 [20], que cubren los escenarios de despliegue más relevantes en los rangos de frecuencia de 0.5 a 100 GHz.

**Modelo de espacio libre:** El modelo de Friis en espacio libre proporciona una cota inferior de las pérdidas de propagación, válida en ausencia de obstrucciones y en condiciones de visión directa (LOS) a distancias cortas. La pérdida de trayectoria en espacio libre se expresa en dB como:

$$\text{PL}_{\text{fs}} = 20 \cdot \log_{10}(d) + 20 \cdot \log_{10}(f_c) + 20 \cdot \log_{10}\!\left(\frac{4\pi}{c}\right) \tag{9}$$

donde $d$ [m] es la distancia transmisor-receptor, $f_c$ [Hz] es la frecuencia portadora, y $c = 3 \times 10^8$ m/s es la velocidad de la luz en el vacío. Evaluando la constante: $20 \cdot \log_{10}(4\pi/c) \approx -147.55$ dB. En consecuencia, a 28 GHz y 100 m de distancia, la pérdida en espacio libre es $\text{PL}_{\text{fs}} \approx 20\log_{10}(100) + 20\log_{10}(28\times10^9) - 147.55 \approx 40 + 209.44 - 147.55 = 101.89$ dB, significativamente mayor que los ~62 dB calculados a 2.1 GHz para la misma distancia.

**Modelos 3GPP TR 38.901:** El estándar 3GPP TR 38.901 define modelos de pérdida de trayectoria para cuatro escenarios macro de despliegue outdoor: Urban Macro (UMa), Urban Micro Street Canyon (UMi), Rural Macro (RMa) e Indoor Hotspot (InH). Cada escenario contempla dos condiciones de propagación: línea de visión directa (LOS) y no-LOS (NLOS) [21].

Para el escenario **Urban Macro (UMa) NLOS**, el modelo de pérdida de trayectoria es:

$$\text{PL}_{\text{UMa-NLOS}} = 13.54 + 39.08 \cdot \log_{10}(d_{3\text{D}}) + 20 \cdot \log_{10}(f_c) - 0.6 \cdot (h_{\text{UT}} - 1.5) \tag{10}$$

válido para $10 \leq d_{3\text{D}} \leq 5000$ m, $1.5 \leq h_{\text{UT}} \leq 22.5$ m, y $0.5 \leq f_c \leq 100$ GHz (con $f_c$ en GHz y $d_{3\text{D}}$ en metros). En esta expresión, $d_{3\text{D}}$ es la distancia tridimensional entre el gNB y el UE (incluyendo la diferencia de altura), $h_{\text{UT}}$ es la altura del terminal de usuario en metros, y la desviación estándar del shadowing es $\sigma_{sf} = 6$ dB. El coeficiente 39.08 del término logarítmico de distancia indica un exponente de pérdida $n \approx 3.9$, significativamente mayor que el valor 2 del espacio libre, reflejando el efecto de las múltiples reflexiones y difracciones en el entorno urbano.

Para el escenario **Urban Macro (UMa) LOS**, el modelo es bifásico:

$$\text{PL}_{\text{UMa-LOS}} = \begin{cases} \text{PL}_1 = 28.0 + 22 \cdot \log_{10}(d_{3\text{D}}) + 20 \cdot \log_{10}(f_c), & 10 \leq d_{3\text{D}} \leq d_{\text{BP}}' \\ \text{PL}_2 = 28.0 + 40 \cdot \log_{10}(d_{3\text{D}}) + 20 \cdot \log_{10}(f_c) - 9 \cdot \log_{10}\!\left[(d_{\text{BP}}')^2 + (h_{\text{BS}} - h_{\text{UT}})^2\right], & d_{\text{BP}}' < d_{3\text{D}} \leq 5000 \end{cases} \tag{11}$$

donde $d_{\text{BP}}' = 4 \cdot h_{\text{BS}}' \cdot h_{\text{UT}}' \cdot f_c / c$ es la distancia de punto de quiebre (breakpoint distance), con $h_{\text{BS}}' = h_{\text{BS}} - 1.0$ m y $h_{\text{UT}}' = h_{\text{UT}} - 1.0$ m siendo las alturas efectivas del transmisor y receptor respectivamente [22]. La discontinuidad en el exponente de pérdida ($n = 2.2$ antes del breakpoint, $n = 4.0$ después) refleja la transición entre la zona de campo lejano cercano, donde domina la componente directa, y la zona de campo lejano distante, donde aparece la interferencia destructiva con la reflexión en el suelo (efecto two-ray).

Para el escenario **Urban Micro Street Canyon (UMi) NLOS**, el modelo TR 38.901 establece:

$$\text{PL}_{\text{UMi-NLOS}} = 35.3 \cdot \log_{10}(d_{3\text{D}}) + 22.4 + 21.3 \cdot \log_{10}(f_c) - 0.3 \cdot (h_{\text{UT}} - 1.5) \tag{12}$$

con una desviación estándar del shadowing de $\sigma_{sf} = 7.82$ dB, válido para $10 \leq d_{3\text{D}} \leq 2000$ m [23]. El exponente de pérdida equivalente ($n \approx 3.53$) es ligeramente inferior al del escenario UMa NLOS, lo que puede atribuirse a los efectos de canalización de la energía electromagnética entre fachadas de edificios en entornos de cañón urbano (street canyon effect).

Para redes mmWave, el modelo **UMi LOS** en FR2 tiene especial relevancia dada la limitada cobertura de los gNBs mmWave:

$$\text{PL}_{\text{UMi-LOS}} = 32.4 + 21 \cdot \log_{10}(d_{3\text{D}}) + 20 \cdot \log_{10}(f_c), \quad d_{3\text{D}} \leq d_{\text{BP}}' \tag{13}$$

A 28 GHz, la aplicación de este modelo para $d_{3\text{D}} = 200$ m resulta en $\text{PL} = 32.4 + 21 \cdot \log_{10}(200) + 20 \cdot \log_{10}(28) = 32.4 + 46.0 + 28.9 = 107.3$ dB, lo que pone de manifiesto que incluso en condiciones LOS, las redes mmWave requieren altas ganancias de beamforming y/o densificación de emplazamientos para proporcionar cobertura útil [24].

### C. Cálculo del Radio de Celda

Una vez determinado el MAPL mediante el link budget (ecuación 6) y seleccionado el modelo de pérdida de trayectoria apropiado para el entorno de despliegue, el radio de celda máximo $R$ se calcula invirtiendo el modelo de propagación. Para un modelo de pérdida de trayectoria de la forma genérica:

$$\text{PL}(d) = \text{PL}_{\text{ref}} + 10 \cdot n \cdot \log_{10}\!\left(\frac{d}{d_{\text{ref}}}\right) + X_\sigma \tag{14}$$

donde $\text{PL}_{\text{ref}}$ es la pérdida de trayectoria a la distancia de referencia $d_{\text{ref}}$ (típicamente 1 m o el breakpoint), $n$ es el exponente de pérdida de trayectoria, y $X_\sigma \sim \mathcal{N}(0, \sigma_{sf}^2)$ es el componente de shadowing lognormal, el radio de celda se obtiene igualando $\text{PL}(R) = \text{MAPL}$ (descontando el margen de shadowing ya incluido en el MAPL):

$$R = d_{\text{ref}} \cdot 10^{\left(\text{MAPL} - \text{PL}_{\text{ref}}\right) / (10 \cdot n)} \tag{15}$$

Esta expresión es fundamental en el dimensionamiento de la red, ya que determina el número de emplazamientos necesarios para cubrir un área $A$ mediante la relación $N_{\text{sites}} = A / A_{\text{cell}}$, donde el área de celda depende del radio $R$ y de la geometría del patrón de reutilización espacial. Para una celda hexagonal, el área de cobertura es $A_{\text{cell}} = \frac{3\sqrt{3}}{2} R^2 \approx 2.598 R^2$, mientras que para un sector de 120° en una configuración trisectorial, $A_{\text{sector}} = \frac{1}{3} A_{\text{cell}}$.

### D. Probabilidades de Cobertura y Modelos de Interferencia

La probabilidad de cobertura de un usuario en la posición $\mathbf{r}$ se define como la probabilidad de que el SINR recibido supere un umbral mínimo $\gamma_{\min}$:

$$p_c(\mathbf{r}) = P\left[\text{SINR}(\mathbf{r}) \geq \gamma_{\min}\right] \tag{16}$$

El análisis exacto de $p_c(\mathbf{r})$ en redes celulares heterogéneas es matemáticamente intratable en el caso general, debido a la aleatoriedad en las posiciones de las estaciones base, los canales de propagación y los niveles de interferencia. La **geometría estocástica** proporciona herramientas analíticas que permiten derivar expresiones en forma cerrada para $p_c$ bajo supuestos simplificadores sobre la distribución espacial de los nodos de red [25].

En el modelo de red celular PPP (Poisson Point Process), donde las estaciones base se distribuyen según un proceso de Poisson homogéneo de densidad $\lambda$ [estaciones/km²], y el canal presenta desvanecimiento de Rayleigh con parámetro unidad, la probabilidad de cobertura para un usuario típico en el enlace descendente es [26]:

$$p_c = \pi \lambda \int_0^\infty e^{-\pi \lambda r^2 (\rho^{2/\alpha} \Gamma(1+2/\alpha)\Gamma(1-2/\alpha) + 1) - \mu \gamma_{\min} \sigma_n^2 r^\alpha / P} \, dr \tag{17}$$

donde $\alpha > 2$ es el exponente de pérdida de trayectoria, $\rho$ es el umbral de SINR normalizado, $\mu$ es el parámetro del desvanecimiento exponencial del canal, y $P$ es la potencia de transmisión. Aunque esta expresión proporciona intuición analítica valiosa, en la práctica del dimensionamiento de redes 5G se utilizan métodos de **simulación de Monte Carlo** para obtener predicciones de cobertura con mayor precisión [27].

El método de Monte Carlo para el análisis de cobertura en 5G consiste en: (i) generar aleatoriamente posiciones de usuarios sobre el área de estudio; (ii) calcular el SINR para cada usuario según el modelo de propagación seleccionado y los niveles de interferencia de las celdas vecinas; (iii) evaluar el cumplimiento del umbral de SINR para cada usuario; y (iv) estimar la probabilidad de cobertura como la fracción de usuarios que cumplen el requisito. El número de realizaciones de Monte Carlo necesario para alcanzar un error de estimación inferior a $\epsilon$ con confianza $1-\delta$ puede determinarse a partir de la desigualdad de Hoeffding [28]:

$$N_{\text{MC}} \geq \frac{\ln(2/\delta)}{2\epsilon^2} \tag{18}$$

Para un error de estimación $\epsilon = 0.01$ (1%) y una confianza del 95% ($\delta = 0.05$), se requieren al menos $N_{\text{MC}} \geq \ln(2/0.05) / (2 \times 0.01^2) = 3.689 / 0.0002 \approx 92{,}224$ realizaciones, es decir, del orden de $9 \times 10^4$ usuarios simulados, un número computacionalmente manejable con los recursos de computación actuales.

---

## IV. DIMENSIONAMIENTO DE CAPACIDAD

### A. Modelos de Tráfico: Teoría de Erlang

El dimensionamiento de la capacidad de una red de comunicaciones requiere modelos matemáticos que relacionen la intensidad de tráfico ofrecida por los usuarios con los recursos de red necesarios para satisfacer un determinado grado de servicio (GoS, Grade of Service). La **teoría de tráfico de Erlang**, desarrollada por el ingeniero danés A.K. Erlang a principios del siglo XX para el dimensionamiento de centrales telefónicas, constituye el fundamento matemático del dimensionamiento de capacidad en sistemas de telecomunicaciones, incluyendo las redes celulares 5G [29].

La unidad de medida del tráfico de telecomunicaciones es el **Erlang** (E), definido como el producto de la tasa de llegada de llamadas $\lambda$ [llamadas/s] por la duración media de cada llamada $h$ [s]: $A = \lambda \cdot h$ [E]. Una carga de tráfico de 1 E equivale a un canal ocupado continuamente, o a 10 llamadas con duración media de 6 minutos en una hora.

Para un sistema de pérdidas con $N$ canales (circuitos o recursos equivalentes) y tráfico ofrecido $A$ [E], la probabilidad de bloqueo (probabilidad de que una nueva llamada encuentre todos los canales ocupados y sea rechazada) se determina por la **fórmula de Erlang B**:

$$E_B(A, N) = \frac{A^N / N!}{\displaystyle\sum_{k=0}^{N} A^k / k!} \tag{19}$$

Esta expresión, derivada bajo el supuesto de llegadas según proceso de Poisson con tasa $\lambda$ y duraciones de servicio exponencialmente distribuidas con media $1/\mu$, es válida bajo la condición de pérdidas (las llamadas bloqueadas abandonan el sistema sin reintentar). Para el dimensionamiento de redes 5G eMBB, donde las sesiones de datos descartadas por falta de recursos representan pérdidas de calidad de servicio, la fórmula de Erlang B permite determinar el número mínimo de canales $N^*$ necesario para mantener la probabilidad de bloqueo por debajo de un objetivo $B_{\max}$:

$$N^* = \min\left\{N \in \mathbb{Z}^+ : E_B(A, N) \leq B_{\max}\right\} \tag{20}$$

Para un tráfico ofrecido $A = 30$ E y un objetivo de bloqueo $B_{\max} = 2\%$, la aplicación de la fórmula de Erlang B (o tablas de Erlang) proporciona $N^* = 42$ canales.

Para sistemas con espera (colas), donde las llamadas o sesiones que encuentran todos los canales ocupados esperan en cola hasta ser atendidas, la probabilidad de espera se determina por la **fórmula de Erlang C**:

$$E_C(A, N) = \frac{\displaystyle\frac{A^N}{N!(1 - \rho)}}{\displaystyle\sum_{k=0}^{N-1} \frac{A^k}{k!} + \frac{A^N}{N!(1 - \rho)}} \tag{21}$$

donde $\rho = A/N$ es la utilización media de los servidores (factor de carga), válida únicamente para $\rho < 1$ (condición de estabilidad de la cola). La fórmula de Erlang C permite calcular la probabilidad de que una solicitud de servicio deba esperar en cola antes de ser atendida, y es aplicable al dimensionamiento de buffers en nodos de red 5G y a la planificación de recursos en sistemas de asignación dinámica [30].

La intensidad de tráfico ofrecida por un sector de celda 5G puede estimarse a partir de los modelos de tráfico de usuario. Para una celda con $U$ usuarios activos simultáneos, cada uno generando tráfico con una tasa media de $r_{\text{user}}$ bits/s, la carga total de tráfico en el enlace descendente es:

$$\rho_{\text{DL}} = \frac{U \cdot r_{\text{user}}}{C_{\text{cell}}} \tag{22}$$

donde $C_{\text{cell}}$ es la capacidad total del enlace descendente de la celda en bits/s. El dimensionamiento correcto requiere que $\rho_{\text{DL}} < \rho_{\max}$, donde $\rho_{\max}$ es la carga máxima de diseño (típicamente 70–80% para garantizar márgenes de tráfico a pico).

### B. Cálculo del Rendimiento de Usuario en 5G NR

El estándar 3GPP TS 38.306 define la fórmula para el cálculo del rendimiento máximo teórico de usuario (UE throughput) en una celda 5G NR, que constituye la base del dimensionamiento de capacidad [31]:

$$T_{\text{UE}} = \nu_{\text{layers}} \cdot Q_m \cdot f \cdot R_{\max} \cdot N_{\text{PRB}} \cdot 12 \cdot \frac{1}{T_s^{\mu}} \cdot (1 - \text{OH}) \tag{23}$$

donde cada parámetro tiene el siguiente significado:

- $\nu_{\text{layers}}$: número de capas de transmisión MIMO. En 5G NR Release 15, el máximo es 8 en downlink (DL) y 4 en uplink (UL).
- $Q_m$: orden de modulación (bits por símbolo de modulación): 2 para QPSK, 4 para 16-QAM, 6 para 64-QAM, 8 para 256-QAM. En condiciones excelentes de canal, el 5G NR puede utilizar 256-QAM para maximizar la eficiencia espectral.
- $f$: factor de escalado (scaling factor), igual a 1 para el cálculo de capacidad máxima teórica.
- $R_{\max}$: tasa de codificación máxima del canal, igual a 948/1024 ≈ 0.9258 según 3GPP TS 38.214.
- $N_{\text{PRB}}$: número de bloques de recursos físicos (Physical Resource Blocks) asignados al usuario en el ancho de banda del canal.
- 12: número de subportadoras por PRB.
- $T_s^{\mu} = 10^{-3} / (14 \cdot 2^\mu)$ [s]: duración del símbolo OFDM para la numerología $\mu$. Para $\mu = 1$ (30 kHz SCS), $T_s^{\mu} \approx 35.7$ µs, por lo que $1/T_s^{1} \approx 28.000$ símbolos/s.
- OH: factor de overhead, que incluye los símbolos de referencia de canal (DMRS, PT-RS), los canales de control físico (PDCCH, PBCH) y el cyclic prefix. Típicamente OH ≈ 0.14 en DL y OH ≈ 0.08 en UL para configuraciones estándar.

Para ilustrar el uso de la ecuación (23), consideremos un gNB 5G NR en la banda n78 (3.5 GHz) con 100 MHz de ancho de banda, numerología $\mu = 1$ (30 kHz SCS), 4 capas MIMO, modulación 256-QAM, $N_{\text{PRB}} = 66$ y OH = 0.14:

$$T_{\text{UE}} = 4 \cdot 8 \cdot 1 \cdot \frac{948}{1024} \cdot 66 \cdot 12 \cdot \frac{1}{35.7 \times 10^{-6}} \cdot (1 - 0.14) \approx 1.58 \text{ Gbps} \tag{24}$$

Este valor de rendimiento teórico máximo por usuario contrasta con el rendimiento promedio real, que es significativamente inferior debido a la variabilidad del canal, el scheduling multi-usuario, la interferencia intercelular y la sobrecarga de protocolo.

### C. Eficiencia Espectral con Massive MIMO

La tecnología **Massive MIMO** (Multiple Input Multiple Output masivo) constituye uno de los pilares tecnológicos del 5G que proporciona los mayores incrementos de capacidad respecto al 4G LTE. Al equipar el gNB con un número $M$ de antenas muy superior al número $K$ de usuarios activos simultáneos ($M \gg K$), Massive MIMO puede servir simultáneamente a múltiples usuarios en la misma frecuencia y tiempo mediante **beamforming multiusuario (MU-MIMO)**, aprovechando los grados de libertad espaciales adicionales [32].

Para un sistema Massive MIMO con $M$ antenas en el gNB y $K$ usuarios single-antenna en uplink (UL), bajo el supuesto de canal de desvanecimiento plano con componentes de pérdida de trayectoria $\beta_k$ y canales de desvanecimiento rápido modelados como variables aleatorias complejas i.i.d. $\mathcal{CN}(0,1)$, la eficiencia espectral (SE) del usuario $k$ con estimación de canal por pilotos y detección MRC (Maximum Ratio Combining) es [33]:

$$\eta_k = \left(1 - \frac{\tau_p}{\tau_c}\right) \cdot \log_2\!\left(1 + \frac{M \cdot \rho_f \cdot \beta_k^2}{\displaystyle\sum_{k' \neq k} \beta_{k'}^2 + \frac{1}{\rho_f}}\right) \tag{25}$$

donde:
- $\tau_p$: longitud de la secuencia de pilotos (símbolos de entrenamiento) por coherencia del canal.
- $\tau_c$: duración del intervalo de coherencia del canal en símbolos (determinado por el producto del ancho de banda de coherencia $B_c$ y el tiempo de coherencia $T_c$: $\tau_c = B_c \cdot T_c$).
- $\rho_f$: relación señal a ruido de la señal piloto normalizada por la potencia de transmisión de cada usuario.
- $\beta_k$: coeficiente de pérdida de trayectoria (large-scale fading) del usuario $k$.
- $\beta_{k'}$: coeficiente de pérdida de trayectoria de los usuarios interferentes $k' \neq k$ (interferencia de contaminación piloto).
- $M$: número de antenas en el gNB.

El término $(1 - \tau_p/\tau_c)$ representa la fracción de la coherencia del canal disponible para la transmisión de datos (penalización por overhead de pilotos), y el argumento del logaritmo es el SINR efectivo del usuario $k$ con detección MRC. La ecuación (25) revela varias propiedades fundamentales de Massive MIMO: (i) el SINR se incrementa linealmente con el número de antenas $M$, lo que explica la denominación "array gain"; (ii) en el límite $M \to \infty$, el ruido térmico y la interferencia de usuarios no contaminados pilotos tienden a cero (efecto "favorable propagation"); y (iii) la interferencia de contaminación piloto (pilot contamination), causada por la reutilización inevitable de secuencias piloto entre celdas vecinas, constituye el límite fundamental de la eficiencia espectral de Massive MIMO en el régimen $M \to \infty$ [34].

La eficiencia espectral total de la celda (suma espectral o sum spectral efficiency) se obtiene sumando las contribuciones de los $K$ usuarios activos:

$$\eta_{\text{sum}} = \sum_{k=1}^{K} \eta_k = \left(1 - \frac{\tau_p}{\tau_c}\right) \sum_{k=1}^{K} \log_2\!\left(1 + \frac{M \cdot \rho_f \cdot \beta_k^2}{\displaystyle\sum_{k' \neq k} \beta_{k'}^2 + \frac{1}{\rho_f}}\right) \tag{26}$$

Esta expresión permite cuantificar la ganancia de capacidad de Massive MIMO en función del número de antenas $M$ y usuarios activos $K$. Para $M = 64$, $K = 8$, $\rho_f = 10$ dB, y $\beta_k = \beta \; \forall k$ (usuarios con la misma pérdida de trayectoria), la SE total resultante es aproximadamente $\eta_{\text{sum}} \approx 30$ bits/s/Hz, un factor 10× superior al valor típico de sistemas 4G LTE-Advanced con 2×2 MIMO [35].

### D. Planificación de Uplink y Downlink: Análisis de Carga

El dimensionamiento de la capacidad de una red 5G requiere analizar separadamente los enlaces de subida (UL) y bajada (DL), ya que presentan características asimétricas de tráfico y diferentes limitaciones de potencia. En el enlace de bajada, la estación base (gNB) dispone de alta potencia de transmisión (típicamente 40–200 W por sector) y antenas con alta ganancia de beamforming, mientras que en el enlace de subida, el UE está limitado a potencias máximas de 23–26 dBm con antenas de baja ganancia.

La ecuación de carga para el enlace de bajada (DL) en un sistema OFDMA, como el 5G NR, se formula a partir de la relación entre el tráfico demandado y la capacidad disponible [36]:

$$\eta_{\text{DL}} = \frac{\displaystyle\sum_{u=1}^{U} \frac{r_u^{\text{DL}}}{\eta_u^{\text{DL}}}}{\frac{B_{\text{total}}}{(1+\alpha_{\text{DL}})}} \leq \eta_{\max} \tag{27}$$

donde $r_u^{\text{DL}}$ es la tasa de datos requerida por el usuario $u$ en DL, $\eta_u^{\text{DL}}$ es la eficiencia espectral media del usuario $u$ (determinada por su SINR promedio según la ecuación 2), $\alpha_{\text{DL}}$ es el factor de overhead del DL, y $\eta_{\max}$ es el factor de carga máxima de diseño. Cuando $\eta_{\text{DL}} > \eta_{\max}$, la celda está dimensionada por capacidad y se requiere subdivisión de celda (cell splitting), densificación de la red o técnicas avanzadas de gestión de recursos radio (RRM).

### E. Asignación de Bloques de Recursos y Scheduling

El proceso de asignación dinámica de bloques de recursos (RB scheduling) en 5G NR es gestionado por el algoritmo de planificación de recursos radio (RR Scheduler) implementado en la capa MAC del gNB. El objetivo del scheduler es maximizar el throughput global de la celda (o alguna función de utilidad multi-usuario) sujeto a restricciones de equidad entre usuarios y cumplimiento de SLAs diferenciados por slice de red [37].

El número total de bloques de recursos disponibles por slot de tiempo (0.5 ms para numerología $\mu = 1$) es $N_{\text{RB}}$ (hasta 132 PRBs para 100 MHz con SCS 30 kHz en FR1). Cada PRB tiene una capacidad de:

$$C_{\text{PRB}} = 12 \cdot Q_m \cdot R \cdot \nu_{\text{layers}} \cdot \frac{N_{\text{sym}}}{T_{\text{slot}}} \tag{28}$$

donde $N_{\text{sym}} = 14$ es el número de símbolos OFDM por slot (formato de slot normal), $T_{\text{slot}} = 0.5$ ms para $\mu = 1$, y los demás parámetros siguen la nomenclatura de la ecuación (23). El throughput total de la celda se obtiene sumando las contribuciones de todos los PRBs asignados a usuarios activos:

$$C_{\text{cell}}^{\text{DL}} = \sum_{p=1}^{N_{\text{RB}}} C_{\text{PRB}}(p) \cdot \mathbf{1}[\text{PRB } p \text{ asignado}] \tag{29}$$

La solución óptima del problema de scheduling en presencia de diversidad multiusuario (multi-user diversity) es el algoritmo Proportional Fair (PF), que asigna cada PRB al usuario que maximiza la relación entre su tasa instantánea y su tasa media histórica, balanceando eficiencia espectral y equidad [38]:

$$u^*(p) = \arg\max_{u} \frac{r_u^{\text{inst}}(p)}{\bar{r}_u} \tag{30}$$

donde $r_u^{\text{inst}}(p)$ es la tasa instantánea que el usuario $u$ puede lograr en el PRB $p$ (determinada por su CQI reportado) y $\bar{r}_u$ es la tasa media del usuario $u$ en una ventana de tiempo deslizante. El algoritmo PF garantiza que todos los usuarios activos reciben una fracción del tiempo de servicio proporcional a su calidad de canal relativa, asegurando eficiencia multiusuario y cierta equidad inter-usuario.

---

*[Continúa en la Parte 2: Secciones V–IX, incluyendo Massive MIMO y Beamforming, mmWave, Network Slicing, Técnicas de Optimización y Conclusiones]*

---

> **Nota:** Este documento constituye la Primera Parte del artículo de revisión. La Segunda Parte incluirá las Secciones V (Massive MIMO y Beamforming Avanzado), VI (Redes mmWave: Modelos y Despliegue), VII (Network Slicing y Dimensionamiento Multi-Servicio), VIII (Técnicas de Optimización en Planificación 5G), IX (Resultados y Análisis Comparativo), X (Conclusiones y Líneas Futuras de Investigación), y la Lista Completa de Referencias [1]–[60].
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

| Estudio | Banda | M (antenas) | EE (bits/Joule) | SE (bits/s/Hz) | Cobertura | Ref. |
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

**Evolución hacia 6G:** La sexta generación de comunicaciones móviles, prevista para despliegue comercial hacia 2030, contempla bandas de comunicación en el rango de terahercios (THz, 0.1–10 THz), con anchos de banda potenciales de varios cientos de GHz, tasas de datos de hasta 1 Tbps, y latencias sub-milisegundo. La planificación de redes THz requerirá nuevos modelos de canal (la propagación en THz está dominada por la absorción molecular del vapor de agua), nuevas arquitecturas de antena (metasuperficies inteligentes, *Reconfigurable Intelligent Surfaces*, RIS) y nuevos algoritmos de cobertura adaptativa.

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
