# Spam Filter Development and Evaluation

Dieses Projekt befasst sich mit der Entwicklung und Validierung eines Machine-Learning-basierten Spam-Filters unter extrem strengen regulatorischen Bedingungen.

## 🎯 Projektziel

Die Hauptaufgabe bestand darin, einen Spam-Filter auf Basis von 10.000 gelabelten E-Mails zu entwickeln, der eine maximale Erkennungsrate bietet, während eine **False Positive Rate (FPR) von ≤ 0,2 %** strikt eingehalten wird. Dies stellt sicher, dass nahezu keine legitimen E-Mails fälschlicherweise blockiert werden.

## 🛠️ Methodik & Optimierung

Um eine robuste Performance zu garantieren, wurden folgende Schritte unternommen:

* **Modellvergleich:** Evaluierung von **Random Forest (RF)** und **Support Vector Machine (SVM)**.
* **Daten-Partitionierung:** Aufteilung in 80 % Training/Validierung und 20 % Hold-out Testset für eine unvoreingenommene Endbewertung.
* **K-Fold Cross-Validation:** Einsatz einer 5-fachen Kreuzvalidierung zur Schwellenwert-Optimierung.
* **Konservative Strategie:** Zur Minimierung des Risikos wurde der **maximale Schwellenwert (pessimistischer Ansatz)** aus der Kreuzvalidierung für die finale Anwendung gewählt.

## 📊 Ergebnisse

Die finale Evaluierung auf dem unseen Testset bestätigte, dass beide Modelle die strikte FPR-Vorgabe von 0,2 % übertroffen haben:

| Modell | Schwellenwert | FPs | FPR | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Random Forest** | 0.9401 | 0 | **0,000 %** | ✅ MET |
| **SVM** | 0.9841 | 0 | **0,000 %** | ✅ MET |

### Prognose der Effektivität (Spam Detection Rate)

Die Effektivität wird durch die **True Positive Rate (TPR)** gemessen, welche angibt, wie viel Prozent des künftigen Spams erkannt werden.

Für das empfohlene Modell ergibt sich:
$$TPR = \frac{TP}{TP + FN} = \frac{1503}{1503 + 103} \approx 93,59\%$$

## 🏆 Fazit & Empfehlung

Obwohl beide Modelle eine perfekte Fehlerquote von 0,0 % bei legitimen Mails erreichten, ist der **Random Forest das überlegene Modell**. Er identifizierte im Test 167 Spam-E-Mails mehr als die SVM und bietet mit einer Erkennungsrate von **93,59 %** den effektivsten Schutz.

---
*Erstellt auf Basis der finalen Projektdaten.*
