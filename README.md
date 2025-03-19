# Data Science Project

Dieses Projekt untersucht einen Datensatz von [Kaggle](https://www.kaggle.com/datasets/robikscube/flight-delay-dataset-20182022?select=readme.md).

In dem Datensatz sind Flugdaten von 2018 bis 2022 enthalten, mit Uhrzeit, Start, Ziel, Verspätung und noch mehr.

## Forschungsfrage

Ist es möglich, anhand der Eckdaten eines Fluges die Gesamtverspätung am Zielflughafen vorherzusagen?
Eckdaten sind hierbei Daten, die schon vor dem Flug bekannt sind, z.B: Datum, Airline, Start und Ziel.

## Hypothesen

1) Flüge im Winter haben häufiger Verspätungen (wegen schlechterem Wetter).
2) Langstreckenflüge haben weniger Verspätungen als Kurzstreckenflüge,
    da Langstreckenflüge von Flughäfen priorisiert werden.
3) Flüge mit Zielflughäfen, die viel Verkehr haben (z. B. Frankfurt), haben – falls sie Verspätungen haben – höhere Verspätungen als an Flughäfen mit weniger Verkehr.
    Das liegt daran, dass bei Verspätungen ein Ausweich-Slot an weniger beflogenen Flughäfen einfacher zu finden ist.

## Vorraussetzungen

Um das Projekt auszuführen, muss `python` mit `pip` auf dem System installiert sein.
Das kann überprüft werden mit folgenden Befehlen:

`python --version` und `python -m pip --version`.

Sind python und pip installiert, können die benötigten Python Module installiert werden:
`pip install -r requirements.txt`.

## Durchführung

1. Daten herunterladen. Ein Script, um die Daten herunterzuladen befindet sich in `./src/data/1_download.py`. Dieses kann ausgeführt werden mit `python 1_download.py`. Das Script wird den kompletten Datensatz runterladen und die benötigten Dateien extrahieren.
