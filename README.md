# ML Hackathon 1 – Fahrradverkehr Vorhersage

**Vorlesung Maschinelles Lernen** | Philipps-Universität Marburg | SoSe 2025  
**Team:** Das Team  
**Plattform:** [Kaggle – UMR-ML-2025 Hackathon 1](https://www.kaggle.com/)

---

## Aufgabenstellung

Ziel dieses Hackathons war die tägliche Vorhersage von Fahrradverkehrszahlen an einer deutschen Zählstation. Der Testzeitraum umfasst **1. Januar 2023 bis Ende Januar 2024**. Als Bewertungsmetrik wurde der **MAE (Mean Absolute Error)** verwendet. Der zu unterbietende Benchmark (Lineare Regression) lag bei einem **MAE von 620,3**.

---

## Datensatz

| Kategorie | Beschreibung |
|---|---|
| Trainingsdaten | 3.901 Tage (Frühjahr 2012 – Winter 2024) |
| Testdaten | 390 Tage (Jan 2023 – Jan 2024) |
| Wetterdaten | Temperatur, Niederschlag, Wind, Sonnenscheindauer (DWD) |
| Feiertagsinformationen | Gesetzliche Feiertage & Schulferien |

---

## Unser Ansatz

### 1. Outlier-Entfernung

6 von 3.901 Datenpunkten (0,2 %) wurden als unrealistische Ausreißer identifiziert und aus dem Training entfernt.

### 2. Feature Engineering (78 Features)

Wir haben ein umfangreiches Feature Engineering durchgeführt, das folgende Kategorien umfasst:

- **Zeitfeatures:** Jahr, Monat, Wochentag, Quartal, Kalenderwoche, Tag im Jahr
- **Zyklische Encodierung:** Sinus/Kosinus-Transformation für Monat, Wochentag, Tag im Jahr und Woche – damit das Modell die Periodizität korrekt erlernt (z.B. dass Dezember und Januar nah beieinander liegen)
- **Saisonale Features:** Jahreszeit (Winter/Frühling/Sommer/Herbst), Wochenende/Werktag, Montag- und Freitagsindikator
- **Wetterfeatures:** Temperaturstufen (sehr kalt / kalt / mild / warm / heiß), Niederschlagsstufen, Interaktionsfeatures wie `perfect_weather` oder `bad_weather`, logarithmierter Niederschlag
- **Feiertagsfeatures:** Kombination aus Schul- und Feiertagen, Überschneidungen mit Wochenenden
- **Trendfeatures:** Jahre seit Beginn des Datensatzes (linear & quadratisch)
- **COVID-Periode:** Indikator für die Lockdown-Jahre 2020–2021

### 3. Modelle & Ensemble

Wir haben vier Modelle trainiert und mittels **Weighted Ensemble** kombiniert. Die Gewichtung basiert auf der inversen Cross-Validation-MAE (TimeSeriesSplit), sodass besser performende Modelle stärker gewichtet werden.

| Modell | CV MAE | Gewicht |
|---|---|---|
| LightGBM | 610,77 | 0,254 |
| XGBoost | 614,12 | 0,252 |
| Gradient Boosting | 615,96 | 0,252 |
| Extra Trees | 639,86 | 0,242 |

**Geschätzter Ensemble CV MAE: ~620,17**

Das finale Ensemble übertrifft den LR-Benchmark (620,3), wobei LightGBM als bestes Einzelmodell mit einem MAE von 610,77 hervorsticht.

---

## Ergebnisse

| Metrik | Wert |
|---|---|
| Benchmark MAE (Lineare Regression) | 620,3 |
| Bestes Einzelmodell (LightGBM CV MAE) | 610,77 |
| Vorhersage-Bereich | 398 – 7.229 |
| Durchschnittliche Vorhersage | 4.349 |

---

## Projektstruktur

```
├── Das_Team-Hackathon_1.ipynb   # Haupt-Notebook mit gesamtem Code
├── submission_advanced_weightedensemble.csv  # Finale Submission
└── README.md
```

---

## Abhängigkeiten

```
numpy
pandas
matplotlib
seaborn
scikit-learn
lightgbm
xgboost
```

---

