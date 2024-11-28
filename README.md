
# Ausführung und Vergleich

## Installation
Für die Ausführung muss zuerst geprüft werden, ob alle benötigten Bibliotheken installiert sind. Dafür wird der folgende Befehl ausgeführt:

```bash
pip install -r requirements.txt
```

## Vergleich
Nach der Ausführung beider Umsetzungen für die Klassifizierung von gegebenen Hunderassen, sind einige Unterschiede klargeworden. Der größte davon ist die Merkmalsdefinition.
Während bei klassischen Machine Learning Ansätzen die Merkmalsdefinition selbst durchgeführt werden muss, kann beim Deep Learning mit Tensorflow auf die eigene Erkennungn zurückgegriffen werden.
Dadurch ist ein einzelnes Definieren der Merkmale nicht notwendig.
In unserem Anwendungsfall erleichtert dies die Durchführung enorm, da Ähnlichkeiten in Merkmalen nicht im Voraus festgelegt, sondern selbst erkannt werden.
Daher würden wir für die weitere Umsetzung eines solchen Klassifizierungsproblems auf die Deep Learning Technologie zurückgreifen.

Auch wenn größere Datensätze verwendet werden, findet die Deep Learning Technologie mit Tensorflow eigene Merkmale und kann diese Unterscheiden und eine präzise Prognose für ein Testinput liefern.
Leider ist das Trainieren eines solchen Deep Learning Systems sehr Rechenintensiv und schränkt damit die Nutzung trotzdem wieder ein.
