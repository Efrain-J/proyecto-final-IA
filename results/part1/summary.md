# Parte 1: resultados de la política Q-learning

## Resultados principales (entrenado y probado en el mismo año, operación mixta)

| Batería | Año de prueba | Ahorro Q-learning (millones COP, mín-máx) | Mejor regla | Ahorro mejor regla (millones COP) | Q ÷ mejor regla | Q ÷ previsión perfecta |
|---|---|---|---|---|---|---|
| 4h | 2024 | 7.19 (6.26-8.47) | shift_schedule | 7.94 | 0.91 | 0.58 |
| 4h | 2025 | 2.67 (1.73-3.19) | shift_schedule | 2.70 | 0.99 | 0.74 |
| 8h | 2024 | 10.68 (9.64-12.26) | price_threshold (2025) | 9.67 | 1.10 | 0.57 |
| 8h | 2025 | 3.13 (2.09-3.75) | price_threshold (2025) | 2.97 | 1.05 | 0.69 |
| 12h | 2024 | 11.28 (10.62-12.57) | price_threshold (2025) | 11.29 | 1.00 | 0.51 |
| 12h | 2025 | 3.52 (2.33-4.27) | price_threshold (2025) | 3.07 | 1.15 | 0.71 |

## Verificación entre años (Q ÷ mejor regla, operación mixta)

| Batería | Entrenado en | Probado en | Q ÷ mejor regla |
|---|---|---|---|
| 4h | 2024 | 2024 | 0.91 |
| 4h | 2024 | 2025 | 0.94 |
| 4h | 2025 | 2024 | 0.82 |
| 4h | 2025 | 2025 | 0.99 |
| 8h | 2024 | 2024 | 1.10 |
| 8h | 2024 | 2025 | 1.06 |
| 8h | 2025 | 2024 | 1.02 |
| 8h | 2025 | 2025 | 1.05 |
| 12h | 2024 | 2024 | 1.00 |
| 12h | 2024 | 2025 | 1.07 |
| 12h | 2025 | 2024 | 1.15 |
| 12h | 2025 | 2025 | 1.15 |

## Tipos de operación (Q ÷ mejor regla, entrenado y probado en el mismo año)

| Batería | Año de prueba | Operación | Q ÷ mejor regla |
|---|---|---|---|
| 4h | 2024 | Mixto | 0.91 |
| 4h | 2024 | Principalmente turnos dobles | 0.92 |
| 4h | 2024 | Principalmente un turno | 0.91 |
| 4h | 2025 | Mixto | 0.99 |
| 4h | 2025 | Principalmente turnos dobles | 0.99 |
| 4h | 2025 | Principalmente un turno | 0.98 |
| 8h | 2024 | Mixto | 1.10 |
| 8h | 2024 | Principalmente turnos dobles | 1.16 |
| 8h | 2024 | Principalmente un turno | 1.02 |
| 8h | 2025 | Mixto | 1.05 |
| 8h | 2025 | Principalmente turnos dobles | 1.10 |
| 8h | 2025 | Principalmente un turno | 0.97 |
| 12h | 2024 | Mixto | 1.00 |
| 12h | 2024 | Principalmente turnos dobles | 1.06 |
| 12h | 2024 | Principalmente un turno | 0.95 |
| 12h | 2025 | Mixto | 1.15 |
| 12h | 2025 | Principalmente turnos dobles | 1.12 |
| 12h | 2025 | Principalmente un turno | 1.12 |

## Método

- Promedio sobre 5 calendarios de turnos de prueba; periodos de prueba octubre-diciembre.
- Los ahorros no incluyen el costo ni el desgaste de la batería (A3).
- La batería conoce el precio de la hora actual (A4).
- Los turnos de la fábrica son sintéticos (A5).
- La mejor regla es la de mayor ahorro medio entre shift_schedule, price_threshold (con cortes de cada año de entrenamiento) y night_day.
