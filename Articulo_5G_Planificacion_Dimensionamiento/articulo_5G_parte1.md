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
