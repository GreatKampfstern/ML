
# Vergleich der Umsetzungen in den Notebooks

Arbeitsaufteilung:
- Deep Learning: Joel Staiger
- Klassisches Machine Learning: Jonas Reif
- Doku: Joel Staiger und Jonas Reif

In diesem Bericht werden die Unterschiede in den Herangehensweisen, der Wahl der Parameter und den Methoden zwischen den beiden Notebooks analysiert. 
Um genauere Beschreibungen für die verwendeten Funktionen und deren Funktionsweise zu erhalten, sind in den Notebooks einzelne Beschreibungen angehangen.

## Datenvorverarbeitung

Für beide Lösungsansätze werden die Bilder durch die vorhandenen Metadaten so vorverarbeitet, dass alle Randinformationen abgeschnitten werden und der Hund formatfüllend im Bild zu erkennen ist. Dadurch werden Muster in Hintergrund seltener als Merkmale des Hundes erkannt und die Erkennung wird zuverlässiger.

### Deep Learning
Für die Voreverarbeitung wurde der Ansatz der Datenaugmentation versucht. Da dieser aber bei der Evakluation schlechtere Ergebnisse erzielt hat als ohne die Verwendung dieser Vorverarbeitung, wurde auf den Einsatz verzichtet.


## Modellarchitektur

### Deep Learning
#### Schichten
CNNs erkennen lokale Merkmale wie Kanten und Texturen, die für die Differenzierung von Hunderassen essenziell sind. Die Faltungsschichten extrahieren schrittweise komplexere Merkmale, während Batch-Normalisierung das Training stabilisiert und MaxPooling die Merkmalskomplexität reduziert, wodurch das Modell robuster gegen Bildverschiebungen wird. Dropout verhindert Overfitting durch gezielte Neuronendeaktivierung, was die Generalisierung verbessert. Die Softmax-Ausgabe liefert präzise Wahrscheinlichkeiten, die eine klare Zuordnung zu den fünf Klassen ermöglichen.
#### Loss-Funktion und Optimizer
Die Wahl der CategoricalCrossentropy als Loss-Funktion ermöglicht eine präzise Bewertung der Modellleistung bei der Klassifikation von Hunderassen, indem sie die Abweichung zwischen den vorhergesagten und den tatsächlichen Klassenlabels minimiert. Diese Methode ist besonders geeignet für die Klassifikation, da sie die Wahrscheinlichkeitsverteilung der Klassen berücksichtigt und somit eine feingranulare Anpassung des Modells ermöglicht.
Der Adam-Optimizer sorgt durch seine adaptiven Lernraten für eine stabile und schnelle Konvergenz des Modells, was besonders bei tiefen neuronalen Netzen von Vorteil ist. Adam kombiniert die Vorteile von zwei anderen Extensions von Stochastic Gradient Descent, nämlich AdaGrad und RMSProp, und ist daher besonders effektiv bei der Arbeit mit großen und komplexen Datensätzen. Es wurde neben Adam noch ein weiterer Optimizer getestet, jedoch hat sich Adam als der effektivere herausgestellt.
Diese Kombination aus Loss-Funktion und Optimizer trägt maßgeblich zur hohen Genauigkeit und Effizienz des Deep-Learning-Modells bei, da sie sowohl die Lernrate dynamisch anpasst als auch die Fehler minimiert, was zu einer robusten und zuverlässigen Klassifikation führt.

 

### Klassisches Machine Learning
Das klassische Machine Learning Modell verwendet eine einfachere Struktur, bestehend aus einem Random Forest Classifier. Dieser basiert auf einer Vielzahl von Entscheidungsbäumen, die jeweils eine Vorhersage treffen und dann die Ergebnisse aggregieren. Die Entscheidungsbäume sind unabhängig voneinander und können parallel trainiert werden, was zu einer schnelleren Berechnung führt. Der Random Forest Classifier ist robust gegenüber Overfitting und Rauschen, da er die Ergebnisse der einzelnen Bäume kombiniert und dadurch eine stabilere Vorhersage ermöglicht. Die Genauigkeit des Modells hängt von der Anzahl der Bäume und der Tiefe der Bäume ab, wobei eine ausgewogene Kombination zu den besten Ergebnissen führt.
Im direkten Vergleich dazu wird auch ein kNN-Modell verwendet. Dieses basiert auf der Berechnung der Distanz zwischen den Merkmalen der Trainingsdaten und den Testdaten, um die Klassenzugehörigkeit zu bestimmen. kNN ist einfach zu implementieren und robust gegenüber Rauschen, da es keine Annahmen über die Verteilung der Daten trifft. Die Genauigkeit des Modells hängt von der Wahl des k-Werts ab, der die Anzahl der nächsten Nachbarn bestimmt, die zur Vorhersage herangezogen werden. Ein niedriger k-Wert führt zu einer flexibleren, aber rauschigeren Vorhersage, während ein hoher k-Wert zu einer glatteren, aber weniger flexiblen Vorhersage führt.

**Vergleich:** 
Die Modellarchitektur im Deep Learning ist durch die Verwendung von Convolutional Neural Networks (CNNs) komplexer und darauf ausgelegt, tiefere Merkmale zu extrahieren. Dies ermöglicht eine präzisere Erkennung von Hunderassen durch die Analyse von lokalen Merkmalen. Im Gegensatz dazu ist die Architektur im klassischen Machine Learning einfacher, was zu einer schnelleren Berechnung führt, jedoch möglicherweise weniger präzise ist. Während Deep Learning Modelle robuster und weniger anfällig für Overfitting sind, setzt klassisches Machine Learning auf eine einfachere Struktur, die schneller trainiert werden kann.

## Hyperparameter

### Deep Learning
#### Lernrate, Batch-Größe und Anzahl der Epochen
Die Lernrate wurde initial auf 0,001 gesetzt, was dem Standardwert für den Adam-Optimizer entspricht. Dieser Wert ermöglicht eine schnelle Konvergenz des Modells in den frühen Trainingsphasen. Während des Transfer-Learnings, bei dem ein vortrainiertes Modell weiter verfeinert wird, wurde die Lernrate auf 0,0001 reduziert. Diese Reduktion der Lernrate ist entscheidend, um das Modell feiner abzustimmen und eine präzisere Anpassung an die neuen Daten zu ermöglichen, ohne die bereits gelernten Merkmale zu stark zu verändern.

Die Batch-Größe und die Anzahl der Epochen wurden schrittweise erhöht, um die optimale Konfiguration für das Training zu finden. Die Batch-Größe wurde von anfangs 16 über 32 auf schließlich 64 erhöht. Eine kleinere Batch-Größe von 16 ermöglichte eine häufigere Aktualisierung der Gewichte, was zu einer schnelleren Anpassung des Modells führte. Jedoch kann eine zu kleine Batch-Größe zu instabilen Gradienten führen. Eine größere Batch-Größe von 64 hingegen stabilisierte die Gradienten, verlangsamte jedoch die Lernrate des Modells. Die Anzahl der Epochen, also die Anzahl der vollständigen Durchläufe durch den gesamten Trainingsdatensatz, wurde von 10 über 20 auf 30 gesteigert. Mehr Epochen erlauben dem Modell, die Daten gründlicher zu lernen, erhöhen jedoch auch die Gefahr des Overfittings, wenn das Modell zu stark an die Trainingsdaten angepasst wird.

Die Ergebnisse der Experimente zeigten, dass eine Batch-Größe von 16 und eine Anzahl von 30 Epochen zu den genauesten Messungen führten. Diese Konfiguration ermöglichte eine ausgewogene Balance zwischen Trainingszeit und Modellgenauigkeit. Die kleinere Batch-Größe sorgte für eine schnellere Anpassung des Modells, während die größere Anzahl an Epochen sicherstellte, dass das Modell ausreichend Zeit hatte, die Merkmale der Daten zu lernen. Insgesamt führte diese Kombination zu einer robusten und präzisen Modellleistung, die sowohl eine hohe Genauigkeit als auch eine gute Generalisierungsfähigkeit aufwies.

### Klassisches Machine Learning
#### Random Forest
Für den verwendeten Random-Forest-Klassifikator wurden unter anderem die Hyperparameter `n_estimators`, `max_depth` und `random_state` optimiert. Dabei gibt `n_estimators` an, wie viele Entscheidungsbäume im Wald erzeugt werden. Eine höhere Anzahl von Bäumen kann zu stabileren Vorhersagen führen, kostet jedoch mehr Rechenzeit. Der Parameter `max_depth` begrenzt die Tiefe der einzelnen Bäume und verhindert somit ein Überanpassen (Overfitting) an die Trainingsdaten. Schließlich sorgt `random_state` für Reproduzierbarkeit der Ergebnisse, indem die Zufallszahlen für den Trainingsprozess festgelegt werden. Die Wahl dieser Hyperparameter ist wichtig, da sie direkt die Balance zwischen Genauigkeit und Generalisierungsfähigkeit des Modells beeinflussen. Eine zu große Tiefe oder zu wenige Bäume können zu ungenaueren Vorhersagen führen, während zu viele Bäume oder zu tiefe Modelle zwar hochgenau, aber anfälliger für Overfitting sind.
Die Analyse zeigt, dass eine Kombination von `max_depth=20` und `n_estimators=200` zu den besten Ergebnissen führt, da sie eine ausgewogene Balance zwischen Genauigkeit und Stabilität des Modells bietet.

#### k-NN Nearest-Neighbor
Beim k-NN-Klassifikator (k-Nearest Neighbor) ist der zentrale Hyperparameter `n_neighbors`, also die Anzahl der für die Klassifikation herangezogenen nächsten Nachbarn. Ein geringer Wert für `n_neighbors` kann dazu führen, dass das Modell sehr sensibel für Ausreißer wird (Overfitting), während ein zu großer Wert die Grenzen zwischen den Klassen verwässert und somit die Genauigkeit senken kann. Des Weiteren beeinflussen Parameter wie die Art der Distanzmetrik (z. B. euklidisch) und Normierungen der Daten die Leistung. Die sorgfältige Wahl dieser Hyperparameter ist von Bedeutung, um ein ausgewogenes Verhältnis zwischen Sensitivität gegenüber lokalen Strukturen der Daten und einer robusten, generalisierenden Entscheidungsfindung zu erreichen.
Die Analyse zeigt, dass ein Wert von `n_neighbors=3` zu den besten Ergebnissen führt, da er eine ausgewogene Balance zwischen lokaler Sensitivität und globaler Generalisierung bietet.

---

# Evaluation und Fazit
Deep Learning und klassisches Machine Learning sind zwei unterschiedliche Ansätze zur Lösung von Klassifikationsproblemen. Während Deep Learning auf komplexen neuronalen Netzen basiert und tiefe Merkmale extrahiert, setzt klassisches Machine Learning auf einfachere Modelle wie Random Forests und k-NN-Klassifikatoren. Die Wahl des geeigneten Ansatzes hängt von der Komplexität des Problems, der Größe des Datensatzes und den verfügbaren Ressourcen ab.
Um eine finale Vorhersage der entsprechenden Hunderasse zu erhalten, muss beim Klassischen Machine Learning Ansatz erst eine eigene Merkmalsdefinition mit der damit verbundenen Merkmalsextraktion durchgeführt werden. Dieser Schritt entfällt beim Deep Learning Ansatz, da die Merkmalsextraktion automatisch durch die Convolutional Neural Networks erfolgt.
Daher wird für die Ausführung eines entsprechenden Deep Learning Ansatzes eine höhere Rechenleistung benötigt, da die Trainingszeit deutlich länger ist als beim Klassischen Machine Learning Ansatz. Wenn jedoch die Ressourcen vorhanden sind, kann Deep Learning zu präziseren und robusteren Modellen führen, die auch mit komplexen Datenstrukturen umgehen können.

Diese Annahmen werden gestützt, von den jeweiligen Ergebnissen aus den Notebooks. Da für diese Auswertung nur ein überschaubarer Datensatz verwendet werden sollte und durch die vorhandene Hardware die benötigte Rechenleistung keine entscheidende Rolle spielt, würden wir uns in dieser Ausarbeitung für den Deep Learning Ansatz entscheiden.
Auch für zukünftige Ausarbeitungen würden wir uns für den Deep Learning Ansatz entscheiden, da dieser durch die automatische Merkmalsextraktion und die dadurch entfallende manuelle Merkmalsextraktion, eine höhere Genauigkeit, Robustheit und eine einfachere Implementierung verspricht.