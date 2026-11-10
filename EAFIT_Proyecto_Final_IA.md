<div align="center">

**Universidad EAFIT**
Escuela de Ciencias Aplicadas e Ingeniería

# Proyecto Final

## Soluciones de IA con datos reales de Colombia: agentes LLM con RAG y dashboard interactivo

**SI3003 - Inteligencia Artificial**

Periodo académico 2026-1

**Fecha máxima de entrega:** Viernes 13 de noviembre de 2026, 6:00 p. m.

Elaborado por: **Jorge Iván Padilla Buriticá**

Universidad EAFIT - 2026

</div>

---

## Índice

1. [Contexto y objetivo](#1-contexto-y-objetivo)
2. [Requisitos obligatorios](#2-requisitos-obligatorios)
3. [Ideas de proyecto](#3-ideas-de-proyecto)
   - 3.1. [Agente de calidad del aire del Valle de Aburrá](#31-agente-de-calidad-del-aire-del-valle-de-aburrá)
   - 3.2. [Copiloto de movilidad: rutas con A\* y asistente RAG](#32-copiloto-de-movilidad-rutas-con-a-y-asistente-rag)
   - 3.3. [Despacho de energía con RL y asistente sobre regulación](#33-despacho-de-energía-con-rl-y-asistente-sobre-regulación)
   - 3.4. [Vigilancia de dengue en Antioquia con redes bayesianas](#34-vigilancia-de-dengue-en-antioquia-con-redes-bayesianas)
   - 3.5. [Asistente agroclimático para café](#35-asistente-agroclimático-para-café)
4. [Cronograma](#4-cronograma)
5. [Entregables](#5-entregables)
6. [Rúbrica de evaluación](#6-rúbrica-de-evaluación)
7. [Recomendaciones](#7-recomendaciones)

---

## 1. Contexto y objetivo

Este proyecto integra lo trabajado en las dos partes del curso. El estudiante debe desarrollar una solución a un problema real usando datos reales colombianos (SIATA, IDEAM, Área Metropolitana del Valle de Aburrá, XM, INS/SIVIGILA, datos.gov.co, Kaggle u otras fuentes abiertas verificables). La solución combina:

- **Parte 1 (fundamentos clásicos):** al menos una técnica como búsqueda (A\*), CSP, MDP/RL, redes bayesianas, HMM o ML clásico, implementada por el equipo.
- **Parte 2 (IA moderna):** un agente basado en LLM con RAG y al menos dos herramientas (tool calling), siguiendo el loop agéntico y, opcionalmente, LangGraph (Clase 12).
- **Dashboard funcional** con visualización e interacción del usuario.

El pacto pedagógico exige que el proyecto combine de forma no trivial al menos una técnica de la Parte 1 con al menos una de la Parte 2.

> **Plazo corto.** Se dispone de aproximadamente nueve días. El alcance esperado es un **MVP funcional**, no un sistema completo. Es preferible un alcance pequeño que funcione completamente a uno ambicioso que no corra.

## 2. Requisitos obligatorios

1. **Datos reales.** Fuente identificada, documentada y citada. Se verifica el acceso en las primeras 48 horas; si una API falla, se usan históricos descargados (plan B).
2. **Modelo de la Parte 1** implementado por el equipo, no solo invocado desde una librería que ya lo resuelve, con una **métrica cuantitativa frente a una línea base**.
3. **Agente LLM con RAG** sobre un corpus relevante al problema y al menos **dos herramientas** (por ejemplo, consultar datos, predecir, planificar, simular).
4. **Visualización:** al menos dos visualizaciones significativas (mapas, series de tiempo, comparaciones, etc.).
5. **Interacción del usuario:** el usuario debe poder, como mínimo, seleccionar parámetros (estación, municipio, fecha, escenario) y conversar con el agente desde la interfaz.
6. **Funcionalidad:** la aplicación corre de forma reproducible, sin intervención manual del equipo, siguiendo el README. Una demostración que falla durante la presentación se califica como no funcional.
7. **Repositorio en GitHub** con código (preferiblemente notebooks de Python y módulos), README, `requirements.txt` y datos o instrucciones para obtenerlos.
8. **Declaración de uso de IA generativa** en la entrega: herramientas usadas y propósito, conforme al pacto pedagógico.

**Sugerencia de stack**

- **Lenguaje y cómputo:** Python; Lightning AI Studio (CPU) por defecto, Kaggle con túnel SSH si se requiere GPU.
- **Agente:** loop manual o LangGraph; endpoint compatible con OpenAI de capa gratuita (NVIDIA NIM, Groq u Ollama) para controlar costos.
- **RAG:** embeddings más FAISS o Chroma.
- **Dashboard:** Streamlit o Gradio (alternativas: Plotly Dash, Panel).

## 3. Ideas de proyecto

Los equipos pueden elegir una de las siguientes ideas o proponer una propia, que debe ser validada previamente por el profesor.

### 3.1 Agente de calidad del aire del Valle de Aburrá

- **Problema.** Los ciudadanos no comprenden qué significa el $\mathrm{PM}_{2.5}$ del día ni qué hacer al respecto.
- **Datos.** SIATA (estaciones de calidad del aire y meteorología), datos abiertos del AMVA, datos.gov.co.
- **Parte 1 (técnica clásica).** HMM o filtrado para estimar el estado latente (normal, alerta, episodio crítico); red bayesiana que relacione lluvia, viento y $\mathrm{PM}_{2.5}$; clasificador (regresión logística o Naive Bayes) para predecir la categoría del día siguiente.
- **Parte 2 (agente LLM).** RAG sobre normativa, protocolos de episodios críticos y recomendaciones de salud. Herramientas: `get_pm25(estación)`, `predict_tomorrow()`, `search_docs()`.
- **Dashboard.** Mapa de estaciones, serie de tiempo, predicción y chat con el agente.
- **Riesgo y plan B.** Acceso a la API de SIATA. Plan B: históricos en CSV.

### 3.2 Copiloto de movilidad: rutas con A\* y asistente RAG

- **Problema.** Elegir ruta o modo (Metro, bus, bicicleta, carro) en el área metropolitana según tiempo, costo y emisiones.
- **Datos.** OpenStreetMap (`osmnx`) para la red vial; datos de movilidad del AMVA y del Metro disponibles como datos abiertos; calidad del aire de SIATA como costo ambiental opcional.
- **Parte 1 (técnica clásica).** Búsqueda A\*/UCS con costo multiobjetivo y heurística admisible; comparación contra BFS/Dijkstra en nodos expandidos.
- **Parte 2 (agente LLM).** Herramienta `plan_route()` y RAG sobre reglas de pico y placa y del sistema Metro; consultas que encadenan varias herramientas.
- **Dashboard.** Mapa con rutas alternativas y tabla comparativa de tiempo, costo y emisiones.
- **Riesgo y plan B.** Tráfico en tiempo real. Plan B: tiempos simulados con parámetros por hora.

### 3.3 Despacho de energía con RL y asistente sobre regulación

- **Problema.** Decidir cuándo cargar o descargar una batería, o desplazar demanda, según el precio de bolsa.
- **Datos.** XM (demanda, precio de bolsa, generación; librería `pydataxm`) e IDEAM para radiación solar.
- **Parte 1 (técnica clásica).** MDP con iteración de valor o Q-learning; comparación con una política ingenua y con una de referencia (Semana 5).
- **Parte 2 (agente LLM).** RAG sobre resoluciones CREG y documentos de XM; el agente explica las decisiones y simula escenarios con `simulate(policy, dates)`.
- **Dashboard.** Precio vs. decisiones, ahorro acumulado y chat; el usuario modifica parámetros de la batería.
- **Riesgo y plan B.** Definición del entorno MDP. Es la opción técnicamente más rigurosa.

### 3.4 Vigilancia de dengue en Antioquia con redes bayesianas

- **Problema.** Estimar el riesgo de brotes por municipio según clima, estacionalidad e historial de casos.
- **Datos.** Boletines y datos de SIVIGILA/INS, datos.gov.co, precipitación y temperatura de IDEAM y SIATA. Solo datos agregados, nunca individuales.
- **Parte 1 (técnica clásica).** Red bayesiana con inferencia por eliminación de variables o Naive Bayes, con validación temporal.
- **Parte 2 (agente LLM).** RAG sobre protocolos del INS y guías de vigilancia; el agente consulta el riesgo por municipio y cita los documentos.
- **Dashboard.** Mapa de riesgo por municipio, tendencia semanal y chat.
- **Riesgo y plan B.** Calidad y rezago de los datos. Es una herramienta analítica, no una guía clínica.

### 3.5 Asistente agroclimático para café

- **Problema.** Riesgo de roya y plagas, y planeación de labores (cosecha, fertilización) según el clima.
- **Datos.** IDEAM (precipitación, temperatura, humedad; datos.gov.co o DHIME) y publicaciones técnicas abiertas de Cenicafé y la Federación como corpus RAG.
- **Parte 1 (técnica clásica).** Red bayesiana o HMM para el riesgo de roya, y un CSP para programar labores con restricciones de clima y mano de obra.
- **Parte 2 (agente LLM).** RAG sobre literatura técnica; herramientas `risk(finca)` y `schedule(finca)`.
- **Dashboard.** Calendario de labores, alertas y chat.
- **Riesgo y plan B.** Pocos datos a nivel de finca. Plan B: usar la estación como proxy.

## 4. Cronograma

| Fecha | Hito |
|---|---|
| Mié 4 - Jue 5 nov<br>Vie 6 - Dom 8 nov | Conformación del equipo, idea elegida y datos descargados; primer EDA. |
| Lun 9 nov | Modelo de la Parte 1 funcionando, con métrica y línea base. |
| Lun 9 - Mié 11 nov | Checkpoint: agente mínimo con una herramienta y dashboard esqueleto. |
| Jue 12 nov | RAG, herramientas adicionales, visualizaciones e interacción. |
| Vie 13 nov, 6:00 p. m. | Congelar código; README, evaluación y declaración de uso de IA. Entrega final en GitHub y presentación/demostración. |

## 5. Entregables

1. **Repositorio GitHub** con el código, la aplicación corriendo desde el README y las instrucciones para reproducir resultados.
2. **Informe breve** (máximo 4 páginas): problema, datos, método de la Parte 1, diseño del agente y RAG, resultados y limitaciones.
3. **Demostración funcional del dashboard** (visualización, interacción y chat con el agente). Opcional como plan B: video demostrativo de 3-4 minutos en caso de que haya problemas.
4. **Evaluación del agente:** entre 10 y 15 preguntas de prueba, indicando si llamó a la herramienta correcta y si citó la fuente.
5. **Declaración de uso de IA generativa.**

> La entrega se realiza a más tardar el viernes 13 de noviembre de 2026 a las 6:00 p. m. Las entregas posteriores no serán recibidas, salvo causa justificada validada por el profesor.

## 6. Rúbrica de evaluación

El proyecto final integrador equivale al **25 %** de la nota del curso. La calificación es grupal, con evaluación del aporte individual.

| Criterio | Peso |
|---|---:|
| Técnica de la Parte 1: implementación propia, corrección y métrica frente a línea base | 20 % |
| Agente LLM: loop, RAG, 2 herramientas, manejo de errores y citas | 20 % |
| Dashboard: visualización clara y significativa | 10 % |
| Interacción del usuario y funcionalidad (corre de punta a punta, sin fallas en la demo) | 15 % |
| Rigor experimental: evaluación del modelo y del agente, análisis de limitaciones | 15 % |
| Reproducibilidad: repositorio, README, dependencias, uso responsable de datos | 10 % |
| Presentación y comunicación del equipo | 5 % |
| Declaración y uso ético de IA generativa | 5 % |
| **Total** | **100 %** |

**Condiciones adicionales**

- Un componente que no corre en la demostración no recibe puntaje en los criterios de funcionalidad e interacción.
- Todos los miembros del equipo deben enviar el repositorio a mi correo jipadillab@eafit.edu.co.
- El uso de IA generativa está permitido, pero debe ser declarado; la dependencia total de la herramienta o la falta de comprensión del trabajo entregado se considera una falta a la integridad académica y puede afectar la evaluación total.
- No habrá sustentación, por lo tanto, el repositorio debe contener todo lo necesario para revisar el proyecto.
- El asunto del correo electrónico del repositorio debe ser: `si3003-AI-grupo-nombre`. Cada integrante del equipo debe enviar el correo.
- La nota aprobatoria es **3.0**.

## 7. Recomendaciones

- Verifique el acceso a los datos el primer día y tenga un plan B con archivos locales.
- Construya primero una versión mínima de punta a punta y mejórela después.
- Establezca $T_{\max}$ en el loop del agente y devuelva los errores de las herramientas como observación.
- Evalúe el agente con preguntas de prueba fijas, no solo de forma manual.
- Documente las limitaciones; ningún sistema de este proyecto debe presentarse como consejo médico, legal o de seguridad.

---

*Elaborado por Jorge Iván Padilla Buriticá - Universidad EAFIT - 2026*
