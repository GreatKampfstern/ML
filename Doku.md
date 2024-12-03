
# Vergleich der Umsetzungen in den Notebooks

In diesem Bericht werden die Unterschiede in den Herangehensweisen, der Wahl der Parameter und den Methoden zwischen den beiden Notebooks analysiert. 

## Datenvorverarbeitung

### Deep Learning
Die Daten wurden mit standardisierten Methoden verarbeitet, einschließlich Datenaugmentation und Transformationen, um die Modelle robuster zu machen. Es wurde auf eine ausgewogene Verteilung von Trainings- und Testdaten geachtet.

### Klassisches Machine Learning
Die Datenverarbeitung fokussiert sich auf eine einfache Normalisierung und eine schnelle Aufteilung in Trainings- und Testdaten, wobei weniger Fokus auf Datenaugmentation gelegt wurde.

**Vergleich:** 
Deep Learning verwendet aufwändigere Transformationsmethoden, während Klassisches Machine Learning auf eine schnellere, weniger komplexe Vorverarbeitung setzt.

## Modellarchitektur

### Deep Learning
Das Modell verwendet eine komplexere Architektur mit mehreren Convolutional- und Pooling-Schichten, um eine bessere Feature-Extraktion zu gewährleisten. Dropout wurde implementiert, um Überanpassung zu vermeiden.

### Klassisches Machine Learning
Das Modell ist einfacher aufgebaut, mit weniger Schichten und einer stärkeren Fokussierung auf Dense-Layer, was zu einer schnelleren Berechnung führt.

**Vergleich:** 
Deep Learning legt Wert auf Tiefenmerkmale durch komplexe Architekturen, während Klassisches Machine Learning eine vereinfachte Architektur für Effizienz priorisiert.

## Hyperparameter

### Deep Learning
Die Lernrate wurde mit einem kleinen Wert gewählt (z. B. 0.001), um das Modell stabil zu trainieren. Die Batchgröße lag bei 32, und Adam wurde als Optimierer verwendet.

### Klassisches Machine Learning
Klassisches Machine Learning verwendet eine größere Lernrate (z. B. 0.01), um die Trainingszeit zu verkürzen. Die Batchgröße ist größer (64), und SGD wurde als Optimierer gewählt.

**Vergleich:** 
Deep Learning optimiert für Genauigkeit und Stabilität, während Klassisches Machine Learning auf Effizienz und Geschwindigkeit abzielt.

## Evaluation

### Deep Learning
Die Bewertung berücksichtigt sowohl die Genauigkeit als auch den Verlust auf dem Testdatensatz.

### Klassisches Machine Learning
Die Bewertung konzentriert sich hauptsächlich auf die Gesamtgenauigkeit, ohne detailliertere Analysen.

**Vergleich:** 
Beide präsentieren zur Evaluation eine Konfusionsmatrix. Dabei ist zu sehen, dass das Deep-Learning-Modell eine höhere Genauigkeit bei der Vorhersage erreicht als der klassische Machine-Learning-Ansatz.

---
