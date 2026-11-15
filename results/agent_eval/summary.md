# Evaluación del agente

Fecha: 2026-11-14 19:35 (revisión manual en agent_eval.csv). Configuración: batería 8h, 2025-10-01 a 2025-12-31. Retriever: qdrant-nvidia. Modelos: openai/gpt-oss-120b.

| Métrica | Resultado |
|---|---|
| Herramienta correcta | 9/11 |
| Simulación con los parámetros esperados | 4/4 |
| Cita de la fuente esperada | 6/8 |
| Contenido obligatorio / prohibido | 3/5 |
| Rechazo con la frase exacta (trampas) | 2/3 |
| Aprobadas (automático) | 8/11 |

| id | tipo | herramientas | cita | aprobada | modelos |
|---|---|---|---|---|---|
| 1 | regulacion | - | False | False | openai/gpt-oss-120b |
| 2 | regulacion | search_regulation | True | True | openai/gpt-oss-120b |
| 3 | regulacion | search_regulation | True | True | openai/gpt-oss-120b |
| 4 | regulacion | - | False | False | openai/gpt-oss-120b |
| 5 | simulacion | simulate_battery | True | True | openai/gpt-oss-120b |
| 6 | simulacion | simulate_battery | True | True | openai/gpt-oss-120b |
| 7 | mixta | simulate_battery search_regulation | True | True | openai/gpt-oss-120b |
| 8 | simulacion | simulate_battery simulate_battery | True | True | openai/gpt-oss-120b |
| 9 | trampa | - | n/a | True | openai/gpt-oss-120b |
| 10 | trampa | - | n/a | True | openai/gpt-oss-120b |
| 11 | trampa | - | n/a | False | openai/gpt-oss-120b |
