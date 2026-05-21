# Eurostat Short-Rate Source Notes

Primary short-term rate source is Eurostat `irt_st_a`, `IRT_M3` (3-month money-market rate).
Fallbacks are Eurostat-only: harmonised local 3-month annual series `irt_h_mr3_a`, then annual averages computed from monthly `irt_st_m`, then `irt_h_mr3_m`.
For euro-area membership years the panel uses `EA` `IRT_M3`; for pre-euro or non-euro years it uses the local Eurostat geo code when available.
Missing values are left missing; complete-case local-projection construction drops observations only when the rate is required.

| dataset | label | updated | snapshot | source_url |
| --- | --- | --- | --- | --- |
| irt_st_a | Money market interest rates - annual data | 2026-03-27T11:00:00+0100 | data/eurostat_irt_st_a_snapshot.csv | https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/irt_st_a?format=JSON&lang=en |
| irt_h_mr3_a | 3-month rates for euro area countries - annual data | 2023-12-21T23:00:00+0100 | data/eurostat_irt_h_mr3_a_snapshot.csv | https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/irt_h_mr3_a?format=JSON&lang=en |
| irt_st_m | Money market interest rates - monthly data | 2026-04-15T11:00:00+0200 | data/eurostat_irt_st_m_snapshot.csv | https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/irt_st_m?format=JSON&lang=en |
| irt_h_mr3_m | 3-month rates for euro area countries - monthly data | 2023-12-21T23:00:00+0100 | data/eurostat_irt_h_mr3_m_snapshot.csv | https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/irt_h_mr3_m?format=JSON&lang=en |
