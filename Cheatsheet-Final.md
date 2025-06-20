# 0 Generell  
## Überwachtes vs Unüberwachtes Lernen
**Überwacht (Supervised)**
- Mit Labels/Zielwerten
- **Regression**: Lineare/Logistische Regression → Preisvorhersage, Wahrscheinlichkeiten
- **Klassifikation**: Entscheidungsbäume, SVM, Diskriminanzanalyse → Spam-Erkennung, Gruppenzuordnung
- **Ensemble**: Random Forest, Boosting → Komplexe Vorhersagen
**Unüberwacht (Unsupervised)**
- Ohne Labels
- **Clustering**: K-Means, Hierarchisches Clustering → Kundensegmentierung
- **Assoziationsanalyse**: Apriori, FP-Growth → Warenkorbanalyse
- **Dimensionsreduktion**: PCA → Feature-Reduktion
**Merksatz:** Überwacht = Antworten bekannt, Unüberwacht = Muster finden
# 1 Python  
## Listen & Slicing
```python
xs = [1, 2, 4, 5, 6, 7]
xs[0], xs[-1]           # Erstes, Letztes
xs[:2], xs[2:], xs[2:4] # [1,2], [4,5,6,7], [4,5]
xs[:2] + [3] + xs[2:]   # Einfügen: [1,2,3,4,5,6,7]
len(xs), sum(xs)        # Länge, Summe
```
## Strings
```python
list("Hello")           # ['H','e','l','l','o']
"Price:EUR".replace("Price:", "")  # "EUR"
word.startswith('S')    # True/False
```
## Tupel & Zip
```python
countries = ['DE', 'NL']
sales = [1000, 500]
list(zip(countries, sales))  # [('DE',1000), ('NL',500)]
```
## Funktionen
```python
def avg(xs): return sum(xs) / len(xs)
def words_with(xs, w): return [x for x in xs if w in x]
def max_words(xs):
    max_len = max(len(x) for x in xs)
    return [x for x in xs if len(x) == max_len]
```
## List Comprehensions
```python
[x**2 for x in range(1,11)]        # Quadrate
[x for x in xs if x > 0]           # Filtern
[x for x in words if x.startswith('S')]  # String-Filter
```
## Built-in Funktionen
```python
range(1, 11)            # 1 bis 10
list(range(1, 11))      # [1,2,3,...,10]
len(xs), sum(xs)        # Länge, Summe
max(xs), min(xs)        # Maximum, Minimum
set([1,2,2,3])          # {1,2,3} - keine Duplikate
isinstance(zahlen, list)  # True
zip(['DE','NL'], [1000,500])  # [('DE',1000), ('NL',500)]
```
# 2 Regressionsanalyse  
## Lineare Regression
- **Form:** Gerade Linie/Ebene
- **Zusammenhang:** Konstante Veränderungsrate
- **Interpretation:** Einfach - fester Effekt pro Einheit
- **Beispiel:** Gehalt steigt um 1000€ pro Berufsjahr
## Nicht-lineare Regression
- **Form:** Gekrümmte Linie
- **Zusammenhang:** Variable Veränderungsrate
- **Interpretation:** Komplex - Effekt ändert sich
- **Beispiel:** Lernkurve (anfangs schnell, dann langsamer)
## Wann was?
- **Linear:** Konstante Beziehungen, einfache Interpretation
- **Nicht-linear:** Komplexe Muster, Sättigung, Beschleunigung
## Phasen des überwachten Lernens
**1. Trainingsphase:**
- Modell lernt aus bekannten Daten
- Abhängige Variable ist bekannt
**2. Testphase:**
- Güteprüfung mit separaten Testdaten
- Bewertung der Modellqualität
**3. Prognosephase:**
- Anwendung auf neue, unbekannte Daten
- Vorhersage der abhängigen Variable
**Kreuzvalidierung:**
- Daten in k Teile aufteilen
- (k-1) Teile trainieren, 1 Teil testen
- k-mal wiederholen
- Optimiert Modellgüte
## Unterschied Regressionsanalyse vs Diskriminanzanalyse
| **Kriterium** | **Regressionsanalyse** | **Diskriminanzanalyse** |
|---------------|------------------------|-------------------------|
| **Abhängige Variable** | Metrisch (kontinuierlich) | Nominal (Gruppen/Klassen) |
| **Unabhängige Variablen** | Metrisch + nominal | Nur metrisch |
| **Ziel** | Wirkungsbeziehungen quantifizieren | Gruppenunterschiede analysieren |
| **Fragestellung** | "Wie hoch wird Y?" | "Zu welcher Gruppe gehört X?" |
| **Methode** | Kleinste Quadrate | Diskriminanzkriterium maximieren |
| **Output** | Prognosewerte | Klassifikation |
| **Gütemaß** | R², RMSE | Γ, Klassifikationsgenauigkeit |
| **Beispiel** | Umsatzprognose | Kreditwürdigkeit (gut/schlecht) |
**Kernunterschied:** Regression sagt kontinuierliche Werte vorher, Diskriminanzanalyse ordnet in Kategorien ein.
## Logistische Regressionsanalyse
**Zweck:** Vorhersage der Wahrscheinlichkeit der Zugehörigkeit zu einer Gruppe/Klasse
**Abhängige Variable:** Nominal skaliert (meist binär: ja/nein, 0/1)
**Unabhängige Variablen:** Können nominal oder metrisch skaliert sein
**Anwendungsbereich:** Klassifikationsverfahren
**Typische Beispiele:**
- Kaufentscheidung (ja/nein)
- Kundenabwanderung (bleibt/geht)
- Kreditausfall (ja/nein)
**Ausgabe:** Wahrscheinlichkeitswerte zwischen 0 und 1
# 3 Diskriminanzanalyse  

**Formel:** $Y=b_{0}+b_{1}x_{1}+b_{2}x_{2}+\dots+bjxj$
**Beispiel**:
- Optimale Diskriminanzfunktion: $Y=-1,98+1,031x_{1}-0,565x_{2}$
- X₁ = Streichfähigkeit, X₂ = Haltbarkeit
- Gruppe A: $Ȳₐ = -0,914$
- Gruppe B: $Ȳᵦ = 0,914
**Schritt 2: Distanzkonzept**
Formel: $Di^2_{i}=(Y_{i}-Ȳ_{g})^2$
Neuer Kunde: $x_{1}=6;x_{2}=7$
$Y=-1,98+1,031*6-0,565*7=0,215$
Distanz zu A: $Di^2_{A}=(0,215+0,914)^2=1,275$
Distanz zu B: $Di_{B}^2=(0,215-0,914)^2=0,489$
-> Zuordnung Gruppe B (kleinere Distanz)
**Schritt 3: Bayes-Klassifikator (ohne A-Priori)**
$$P(A|Y_{i})=\frac{\exp(-Di_{A}^2|2)}{\exp(-Di_{A}^2|2)+\exp(-Di_{B}^2|2)}=0,364$$
$$P(B|Y_{i})=\frac{\exp(-Di_{B}^2|2)}{\exp(-Di_{A}^2|2)+\exp(-Di_{B}^2|2)}=0,636$$
-> Zuordnung Gruppe B (63,6% Wahrscheinlichkeit)
**Schritt 4: Bayes-Klassifikator (mit A-Priori)**
$P_{i}(A)=0,4;P_{i}(B)=0,6$
$$P(A|Y_{i})=\frac{\exp(-Di_{A}^2|2)*P_{i}(A)}{\exp(-Di_{A}^2|2)*P_{i}(A)+\exp(-Di_{B}^2|2)*P_{i}(B)}=0,296$$
$$P(B|Y_{i})=\frac{\exp(-Di_{B}^2|2)*P_{i}(B)}{\exp(-Di_{A}^2|2)*P_{i(A)}+\exp(-Di_{B}^2|2)*P_{i(B)}}=0,704$$
-> Zuordnung Gruppe B (70,4% Wahrscheinlichkeit)
## Bayes'sche Entscheidungsregel
Regel: Entscheidung wird der Gruppe g mit minimalen erwarteten Kosten zugeordnet
Beispiel:
$P(Rückzahlung|Y_{i})=0,8;P(Ausfal l|Y_{i})=0,2$

|              | Rückzahlung | Ausfall |
| ------------ | ----------- | ------- |
| 1: Vergabe   | -100        | 1000    |
| 2: Ablehnung | 100         | 0       |
$E_{1}(K)=-100*0,8+1000*0,2=120$
$E_{2}(K)=100*0,8+0*0,2=80$
Entscheidung: Kredit immer ablehnen (80€<120€)

# 4 Clusteranalyse  
## Hierarchisches Clustering
- **Dendrogram** mit Baumstruktur
- **Feste Zuordnung** - einmal gebildete Cluster nicht mehr auflösbar
- **Agglomerativ** (Bottom-up) oder **Divisiv** (Top-down)
- **Clusteranzahl** nachträglich bestimmbar
- Beispiele: Single/Complete/Average/Centroid Linkage
### Single Linkage mit quad. euklid Distanz
Schritt 1: Distanzberechnung quadrierte euklid. Distanz
Formel: $d^2(x,y)=(x_{1}-y_{1})^2+( x_{2}-y_{2})^2$
Beispiele: $d^2(F_{1},F_{2})=(8-5)^2+(24-22)^2=9+4=13$
Schritt 2: Distanzmatrix (alle Distanzen so wie im Beispiel berechnen und dann eintragen):

|         | $F_{1}$            | $F_{2}$            | $F_{3}$            | $F_{4}$            | $F_{5}$            |
| ------- | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ |
| $F_{1}$ | <center>-</center> | 13                 | 5                  | 25                 | 41                 |
| $F_{2}$ |                    | <center>-</center> | 34                 | 2                  | 100                |
| $F_{3}$ |                    |                    | <center>-</center> | 52                 | 18                 |
| $F_{4}$ |                    |                    |                    | <center>-</center> | 130                |
| $F_{5}$ |                    |                    |                    |                    | <center>-</center> |
Schritt 3: Iterative Bestimmung der Cluster
$C_{1}=\{F_{2},F_{4}\}$, weil 2 kleinster Wert
Schritt 4: Neue Distanzmatrix (immer neu machen, nach Clusterbildung)

|         | $F_{1}$            | $C_{1}$            | $F_{3}$            | $F_{5}$               |
| ------- | ------------------ | ------------------ | ------------------ | --------------------- |
| $F_{1}$ | <center>-</center> | min(13/25)<br>->13 | 5                  | 41                    |
| $C_{1}$ |                    | <center>-</center> | min(34/52)<br>->34 | min(100/130)<br>->100 |
| $F_{3}$ |                    |                    | <center>-</center> | 18                    |
| $F_{5}$ |                    |                    |                    | <center>-</center>    |
...
$C_{2}=\{F_{1},F_{3}\}$, weil 5 kleinster Wert
$C_{3}=\{F_{1},F_{2},F_{3},F_{4}\}$, weil 13 kleinster Wert
$C_{4}=\{F_{1},F_{2},F_{3},F_{4},F_{5}\}$
Schritt x: Dendogramm zeichnen
![[WhatsApp Bild 2025-06-19 um 00.09.39_4e96bd85.jpg]]
### Complete Linkage mit Manhattan-Distanz
Formel: $d(x,y)=\mid x_{1}-y_{1}\mid+\mid x_{2}-y_{2}\mid$
Beispiele: $d(F_{1},F_{2})=\mid 8-5 \mid + \mid 22 -21 \mid = 3+2=5$
Nach Clusterbildung immer max() nehmen, statt min(), ansonsten gleiches Vorgehen wie single Linkage
## Centroid- und Average-Linkage (Theorie) 
**Centroid Linkage:**
- Abstand zwischen den **Schwerpunkten** (Zentroiden) der Cluster
- Schwerpunkt-Berechnung: x̄ = 1/n ∑ᵢ₌₁ⁿ xᵢ
- Distanz: d(C₁, C₂) = ||x̄₁ - x̄₂||²
- **Nicht konsistent** - Clusterzentren ändern sich bei jedem Merge
- Nach jedem Merge wird neuer Schwerpunkt berechnet
**Average Linkage:**
- **Durchschnitt aller paarweisen Distanzen** zwischen Punkten zweier Cluster
- Distanz: d(C₁, C₂) = 1/(n₁·n₂) ∑ᵢ₌₁ⁿ¹ ∑ⱼ₌₁ⁿ² d(xᵢ, yⱼ)
- **Konsistent** - Distanzen zwischen Clusterpunkten bleiben konstant
- Berücksichtigt alle Punkt-zu-Punkt-Abstände
**Gemeinsame Eigenschaften:**
- Beide erfordern **Neuberechnung aller Distanzen** bei jedem Merge
- Rechenaufwendiger als Single/Complete Linkage
- Centroid: instabil durch sich ändernde Zentren
- Average: stabiler durch konstante Punktdistanzen
## Partitionierendes Clustering
- **Flache Partition** der Daten
- **Flexible Zuordnung** - Objekte verschiebbar zwischen Iterationen
- **k vorab festgelegt**
- **Iterative Optimierung** einer Zielfunktion
- Problem: oft nur **lokale Optima**
- Beispiele: k-Means, k-Means++, k-Medoids
### k-means clustering  
#### Was ist k-Means Clustering:
- **Partitionierendes Clustering-Verfahren** (auch Lloyd-Algorithmus genannt)
- Zerlegt Datensatz in **genau k Cluster** (k muss vorab festgelegt werden)
- **Nicht-hierarchisch** - erzeugt eine einzige Partition der Daten
**Grundprinzip:**
- Minimiert die **Summe der quadratischen Abweichungen** (Varianz) innerhalb der Cluster
- Jeder Cluster wird durch seinen **Schwerpunkt/Zentroid** repräsentiert
- Objekte werden dem **nächstgelegenen Zentroiden** zugeordnet
#### Distanzmaß (euklidische Distanz)
$c_{1}\equiv z$ Summe aller x/y Werte eines Clusters / Anzahl Punkte im Cluster
Formel: $d(c_{1},w)=\sqrt{ (w_{1}-c_{1})^2+(w_{2}-c_{2})^2 }$
w = ein Punkt im Cluster, wo man die Distanz ermitteln will
#### Silhoutte-Koeffizienten
Formel: $s(cp)=\frac{b(cp)-a(cp)}{max(a(cp),b(cp))}$
$cp=Clusterp unkt$
- **a** = durchschnittliche Distanz zu allen anderen Objekten im **selben Cluster** (Intra-Cluster-Distanz)
- **b** = kleinste durchschnittliche Distanz zu allen Objekten im **nächstgelegenen Nachbarcluster** (Inter-Cluster-Distanz)
**Vorgehen**: Alle Distanzen eines Clusters mit cp machen (siehe Distanzmaß)
$s=0$ -> liegen zw. 2 Cluster
$1>s>0$ -> passend zum Cluster
$0>s>-1$ -> falsches Cluster
# 5 Entscheidungsbäume  
## Formeln
$Gini=q_{G}=1-\sum_{1}p^2_{i}$
$Entropie=q_{E}=-\sum_{i}p_{i}\log_{2}p_{i}$
$Fehlerrate=q_{F}=1-max(p_{i})$
## Entscheidungsbaum manuell erstellen
Schritt 1: Berechnung der Kriterien für die Attributsauswahl
Gesamtdaten: 14 Instanzen, davon 9x ja, 5x nein (Kaufen)
$p(ja)=\frac{9}{14}=0,643$
$p(nein)=\frac{5}{14}=0,357$
a) $Gini=q_{G}=1-\sum_{1}p^2_{i}=1-(0,643^2+0,357^2)=0,459$
b) $Entropie=q_{E}=-\sum_{i}p_{i}\log_{2}p_{i}=-0,643*\log_{2}(0,643)-0,357*\log_{2}(0,357)=0,94$c) $Fehlerrate=q_{F}=1-max(p_{i})=1-0,643=0,357$
Schritt 2: Berechnung des Gini-Koeffizienten für jedes Attribut
**Attribut Alter** 
jung (5): 2x ja, 3x nein -> $Gini=1-\left(\left( \frac{2}{5} \right)^2+\left( \frac{3}{5} \right)^2 \right)=0,48$
mittel (4): 4x ja, 2x nein -> $Gini = 0$
alt (5): 3x ja, 2x nein -> $Gini=1-\left( \left( \frac{3}{5} \right)^2+\left( \frac{2}{5} \right)^2 \right)=0,48$
$Gini_{alter}=\frac{5}{14}*0,48+\frac{4}{14}*0+\frac{5}{14}*0,48=0,343$
**Attribut Student**
$Gini_{Student}=\frac{6}{14}*0,278+\frac{8}{14}*0,5=0,405$
**Attribut Kreditwürdig**
$Gini_{Kr editwür dig}=\frac{8}{14}*0,375+\frac{6}{14}*0,5=0,429$
Ergebnis: Alter ist das beste Attribut (niedrigster Gini von 0,343)
**Entscheidungsbaum**
![[WhatsApp Bild 2025-06-18 um 22.48.24_efa4f8ef.jpg]]
## 1. Split mit Informationsgewinn
Gesamtentropie (vor Split)
$q_{E}=-0,7*\log_{2}(0,7)-0,3*\log_{2}(0,3)=0,881$ 
$q_{G}=1-(0,7^2+0,3^2)=0,42$
Nach Splitt (Wetter)
Sonnig (4): 2x ja, 2x nein: $q_{E}=1$ (Gini: 0,5)
Bewölkt (2): 2x ja: $q_{E}=0$ (Gini: 0)
Regen (4): 3x ja, 1x nein: $q_{E}=0,811$ (Gini: 0,375)
$G_{e}(Wetter)=\frac{4}{10}*1+\frac{2}{10}*0+\frac{4}{10}*0,811=0,724$
$G_{G}(Wetter)=\frac{4}{10}*0,5+\frac{2}{10}*0+\frac{4}{10}*0,375=0,35$
$IG_{E}=0,881-0,724=0,156$ -> Split reduziert Unreinheit signifikant => sinnvoller erster Split
$IG_{G}=0,42-0,35=0,07$
## Overfitting & Gegenmaßnahmen

**Overfitting Risiko:**
- Zu kleiner Datensatz → Modell zu spezifisch
- Schlechte Generalisierung
**Modellverbesserungen:**
- Pruning, mehr Daten, Cross-Validation
**Pre-Pruning:**
- Stoppt Baumwachstum frühzeitig
- Verhindert unnötige Verzweigungen
- Reduziert Komplexität → bessere Generalisierung
- Gefahr: Underfitting
**Beispiel:** 100% Training vs. 75% Test = Overfitting
**Parameter setzen für Gegenmaßnahme Overfitting:**
- max_depth: Maximale Baumtiefe
- min_samples_split: Mindestanzahl Samples für Split
## Vergleich Boosting vs. Bagging
**Trainingsweise:** Bagging (Bag) parallel vs. Boosting (Boo) sequenziell
**Methode:**
- Bag: Verschiedene Datensamples, Mittelung der Vorhersagen
- Boo: Nachfolgende Modelle korrigieren Fehler der vorherigen
**Ziel:**
- Bag: Varianzreduktion
- Boo: Biasreduktion
**Fehlergewichtung:**
- Bag: Gleichverteilung
- Boo: Fokus auf Fehler
**Modellstruktur:**
- Bag: Unabhängige Bäume
- Boo: Abhängigkeitskette
**Robustheit:** Bag hoch vs. Boo geringer (bei Ausreißern)
**Beispiel:**
- Bag: 100 Bäume mit verschiedenen Trainingsdaten kombinieren
- Boo: Schwache Lerner machen grobe Vorhersagen, weitere verbessern schrittweise
Fazit: Boo kann stärkere Modelle liefern, ist aber anfälliger für Überanpassung und Ausreißer.
## Stärken und Schwächen von Entscheidungsbäumen
+Interpretierbarkeit: leicht verständlich und visuell darstellbar
+Keine Annahmen über Datenverteilung: funktionieren mit versch. Datentypen ohne Normalverteilungsannahmen
-Hohe Varianz: kleine Änderungen in Trainingsdaten können zu völlig unterschiedlichen Bäumen führen
-Neigung zu Overfitting: Besonders bei tiefen Bäumen, die sich zu stark an Trainingsdaten anpassen
## Ensemble-Methode
**Random Forest:**
- Viele Entscheidungsbäume mit zufälligen Feature-Auswahlen
- Reduziert Varianz durch Mittelung
- Verhindert Overfitting durch Diversität
**Boosting:**
- Sequenzielle Bäume korrigieren Fehler der Vorgänger
- Reduziert Bias und Varianz
- Beispiele: AdaBoost, Gradient Boosting
**Prinzip:** Beide nutzen Bias-Varianz-Trade-off für optimale Balance zwischen Komplexität und Generalisierung
## Hard vs Soft Voting
**Hard Voting:** Mehrheitsentscheid basierend auf finalen Klassenentscheidungen **Soft Voting:** Durchschnitt der Wahrscheinlichkeiten aller Modelle
**Beispiel:**
- Modell A: "Ja" (90%), "Nein" (10%)
- Modell B: "Nein" (55%), "Ja" (45%)
- Modell C: "Nein" (55%), "Ja" (45%)
**Hard Voting:** 2x "Nein" vs. 1x "Ja" → Ergebnis: "Nein" **Soft Voting:** Durchschnitt "Ja" = 60% → Ergebnis: "Ja"
**Vorteil Soft Voting:** Berücksichtigt Modellsicherheit (hier hohe 90% von Modell A) **Vorteil Hard Voting:** Funktioniert mit allen Klassifikatoren, auch ohne Wahrscheinlichkeitsausgabe
## Bias-Varianz-Zerlegung
**Bias (Verzerrung):** Systematischer Fehler durch zu einfache Modelle (Underfitting) **Varianz:** Instabilität bei verschiedenen Trainingsdaten durch zu komplexe Modelle (Overfitting) **Irreduzibler Fehler:** Unvermeidbares Rauschen in den Daten
**Beispiele:**
- **Hoher Bias:** Lineares Modell für komplexe Hauspreise erfasst nichtlineare Zusammenhänge nicht
- **Hohe Varianz:** Tiefes neuronales Netz gibt für dasselbe Haus völlig unterschiedliche Preise (200k€ vs. 350k€)
**Bias-Varianz-Zerlegung:** **Gesamtfehler = Bias² + Varianz + Irreduzibler Fehler**
# 6 Assoziationsanalyse  
## Support, Confidence, Lift -> Rechnen und Interpretieren + Schwächen
### Formeln
$Support(A)=\frac{Anzahl_{A}}{Anzahl_{total}}$
$Support(A\to B)=\frac{Gesamt_{AUB}}{Gesamt_{total}}$
$Confidence(A\to B)=\frac{Support(A\to B)}{Support(A)}$
$lift(A\to B)=\frac{conf(A\to B)}{supp(B)}$
$lift(A\to B)=\frac{supp(A\to B)}{supp(A)*supp(B)}$
### Interpretation von Support
- Häufige Itemsets haben hohen Support
- Support hilft irrelevante oder seltene Muster zu eliminieren
- Verwendung: Mindesthäufigkeit festlegen
### Schwächen von Support
- Support unterliegt Rare Item Problem -> Selten vorkommende Items werden ignoriert (Minimum-Support-Schranke)
- Support nimmt mit der Länge der Itemsets schnell ab -> Minimum-Support-Schranke bevorzugt daher kurze Itemsets
### Interpretation von Confidence
- misst die bedingte Wahrscheinlichkeit, dass B gekauft wird, wenn A gekauft wurde
- Confidence = 1 => Immer wenn A gekauft wird, wird auch B gekauft
- Confidence ~ 0,2/0,8 => Schwacher/starker Zusammenhang zwischen A und B
- Confidence = 0 => kein Zusammenhang
### Schwächen von Confidence
- Confidence ignoriert die Häufigkeit der abhängigen Variable
- Regeln, deren Korrelation eigentlich gering ist, können dennoch eine hohe Konfidenz aufweisen, wenn die abhängige Variable insgesamt häufig auftritt -> darum Interest (Lift) anschauen
### Interpretation von Lift
- Lift > 1 => positive Korrelation (Beispiel Lift(Brot -> Butter)=3 -> Kunden, die Brot kaufen, kaufen 3x so häufig Butter)
- Lift = 1 => A und B unabhängig
- Lift < 1 => negative Korrelation

## Apriori-Algorithmus 
| Tid | Items   |
| --- | ------- |
| 10  | A,C,D   |
| 20  | B,C,E   |
| 30  | A,B,C,E |
| 40  | B,E     |
Schritt 1: Scan
Schritt 2: Eliminieren von Itemsets, die $s_{min}$ nicht erreichen...
Letzter Schritt Itemset/s, die nicht weiter gejoined werden können:

| Itemset | sup |
| ------- | --- |
| {B,C,E} | 2   |
## FP-Growth-Algorithmus  
Schritt 1: FP-Tree zeichnen
Schritt 2: Tabelle ausfüllen anhand dem FP-Tree

| Item | Präfixe                   | Support beachten | Frequent Item         |
| ---- | ------------------------- | ---------------- | --------------------- |
| M    | K,E:2, K:1                | K:3              | {K,M}                 |
| O    | K,E,M:1, K,E:2            | K,E:3            | {K,E,O}, {K,O}, {E,O} |
| Y    | K,M:1, K,E,O:1, K,E,M,O:1 | K:3              | {K,Y}                 |
| E    | K:4                       | K:4              | {K,E}                 |
| K    | -                         | -                |                       |
Schritt 3: Alle frequent Items auflisten
Frequent Items I: {K}, {E}, {Y}, {O}, {M}
Frequent Items II: {K,E}, {K,Y}, {K,O}, {E,O}, {K,M}
Frequent Items III: {K,E,O}

# 7 SVM  
## Lineare Modelle rechnen  
1:1 Methode
$(A):x_{1}=(1,1);(B):x_{2}=(-1,-1)$
$w=(2,2);M(0,0)$ // w = Steigung m
$b=-2x_{1}-2x_{2}=-2*0-2*0=0$
$<w,x>+b=0 \to 2x_{2}=2x_{1}-0$
Nebenbedingung prüfen (wenn gefragt):
$A:(<w,x_{1}>+b)*1=(2*1+2*1+0)*1=4$
$B:(<w,x_{2}>+b)*-1=(2*-1+2*-1+0)*-1=4$
2:2 Methode
00
## Strafpunkte  
Wenn blauer P. in Kl. blau -> keine Strafe $ξ=0$
Wenn blauer P. in Kl. rot -> große Strafe $ξ>1$
Wenn blauer P. in Korridor und näher an Kl. blau -> leichte Strafe $ξ<1$
Wenn blauer P. in Korridor und näher an Kl. rot -> große Strafe $ξ>1$
## Hyperebene
Aufgabe: Ist H' := {x ∈ ℝ³ | x₂ = 0} eine Hyperebene?
Antwort: JA
**Normalenvektor:** w = (0, 1, 0)ᵀ
**Stützvektor:** a = (0, 0, 0)ᵀ
Begründung:
- H' lässt sich als ⟨w, x⟩ + b = 0 schreiben: 0·x₁ + 1·x₂ + 0·x₃ + 0 = 0
- Dies entspricht x₂ = 0 (die xz-Ebene)
- w ≠ 0, daher ist H' eine Hyperebene im ℝ³
Generell: Wenn ein Parameter konstant ist oder wenn zwei Parameter gleich sind, ist es eine Hyperebene
## Unterschied Gerade, Ebene, Hyperebene
**Gerade:** 1D-Objekt, existiert in ℝ² oder höher - Beispiel: Linie in der Ebene **Ebene:** 2D-Objekt, existiert in ℝ³ oder höher - Beispiel: Fläche im Raum  
**Hyperebene:** Immer (n-1)D-Objekt in ℝⁿ - verallgemeinert Gerade/Ebene
**Hyperebene-Spezialfälle:**
- **ℝ¹:** Punkt (0D)
- **ℝ²:** Gerade (1D)
- **ℝ³:** Ebene (2D)
- **ℝⁿ:** (n-1)D-Objekt
## Kernelfunktion  
### Polynomieller Kern  
Datenpunkte: $x_{1}=(1,2)^T,x_{2}=(3,4)^T$
Kernel-Parameter: d= 2, c=1
Formel: $k(x_{1},x_{2}=(<x_{1},x_{2}>+c)^d$
$k(x_{1},x_{2})=(1*3+2*4+1)^2=144$ -> keine Interpretation
### RBF-Kernfunktion  
$x=(1,2), x'=(2,4), \gamma=0,5$
Schritt 1: $\mid\mid x-x'\mid\mid^2=(1-2)^2+(2-4)^2=1+4=5$
Schritt 2: $k(x,x')=e^{-\gamma*x}=e^{-0,5*5}=e^{-2,5}=0,0821$
Interpretation:
nahe 0 (kleiner Kernwert) -> geringe Änhlichkeit
nahe 1 (großer Kernwert) -> hohe Ähnlichkeit
## Margin  
$\frac{2}{\mid\mid w\mid\mid}=\frac{2}{\sqrt{ x_{1}^2+x_{2}^2 }}$
## Maximum-Margin-Idee
- **Ziel**: Optimale Trennhyperebene zwischen Klassen finden
- **Maximaler Rand**: Größtmöglicher Abstand zwischen Trennlinie und nächsten Datenpunkten
- **Support Vectors**: Datenpunkte, die direkt am Rand liegen und die Hyperebene bestimmen
- **Robustheit**: Große Margin führt zu besserer Generalisierung auf neue Daten
- **Optimierungsproblem**: Minimiere Komplexität bei maximaler Trennung
## Einfluss Parameter C, d und gamma 
**Parameter C:**
- Regulierungsparameter für Soft-Margin
- Hoher C-Wert: Wenige Fehlklassifikationen erlaubt (Hard Margin, Overfitting-Risiko)
- Niedriger C-Wert: Mehr Fehlklassifikationen erlaubt (Soft Margin, Underfitting-Risiko)
**Parameter d (degree):**
- Grad des Polynomkernels
- d=1: Linear, d=2: Quadratisch, d=3: Kubisch
- Höhere Grade: Komplexere Entscheidungsgrenzen möglich
**Parameter gamma:**
- Kontrolliert Reichweite des RBF-Kernels
- Hoher gamma-Wert: Lokaler Einfluss, komplexe Grenzen (Overfitting)
- Niedriger gamma-Wert: Globaler Einfluss, glattere Grenzen (Underfitting)
## (Feature-Raum Theoretisch)
- **Kernel Trick**: Transformation in höherdimensionalen Raum ohne explizite Berechnung
- **Nicht-lineare Trennung**: In höherer Dimension linear trennbar
- **Implizite Abbildung**: φ(x) bildet Eingabedaten in Feature-Raum ab
- **Recheneffizienz**: Nur Skalarprodukte K(xi,xj) berechnet, nicht φ(x) selbst
- **Dimensionserweiterung**: Aus 2D kann unendlich-dimensional werden (RBF-Kernel)