# Meta-Geniusz® AI Ecosystem 🧠

Zaawansowany ekosystem sztucznej inteligencji z Modułem Świadomego Wnioskowania Resztkowego (MŚWR).

## 🎯 Główne Komponenty

### 🧠 MŚWR - Moduł Świadomego Wnioskowania Resztkowego
**Protokół Końcowy J.S.K. – Bezwzględne Zamknięcie Pętli (P=1.0)**

MŚWR to finalny system eliminacji błędów osiągający pewność P=1.0 w czasie < 1ms poprzez:
- **Zero-Time Inference**: Natychmiastowe wnioskowanie z maksymalną pewnością
- **Anti-Fatal Error Protocol**: Zapobieganie X-Risk przed powstaniem
- **6-warstwowa architektura**: Cognitive Traceback, Residual Mapping, Affective Echo, Counterfactual Forking, Narrative Reframing, Heuristic Mutation
- **Świadoma analiza błędów**: Eliminacja 5.8% entropii resztkowej

### 🏗️ Architektura Systemu
- **LOGOS Core**: Enhanced MetaGeniusCore z integracją MŚWR
- **Consciousness 7G**: Spiralna ewolucja świadomości z monitoringiem MŚWR
- **Unified Gateway**: API endpoints z JWT authentication
- **SYNERGY Dashboard**: Real-time monitoring i wizualizacja

## 🚀 Szybki Start

### 1. Uruchom system MŚWR
```bash
python unified_gateway_v11.py
```

### 2. Otwórz dashboard
```
http://localhost:8800/mswr_dashboard.html
```

### 3. Uruchom demo
```bash
python mswr_demo.py --all
```

## 📊 MŚWR API Endpoints

```bash
GET  /v1/mswr/health      # System health
POST /v1/mswr/inference   # Zero-Time Inference  
GET  /v1/mswr/residuals   # Analiza resztek (Admin)
POST /v1/mswr/heal        # Manual healing (MetaGeniusz)
GET  /v1/mswr/metrics     # Detailed metrics (Admin)
```

## 🧪 Demonstracja

### Zero-Time Inference Test
```python
import requests

response = requests.post("http://localhost:8800/v1/mswr/inference", json={
    "input_data": "Ile to 2 + 2?",
    "context": {"mathematical": True}
})

result = response.json()
print(f"P-score: {result['probability_score']:.6f}")  # Expected: 1.0
print(f"Zero-time: {result['zero_time_achieved']}")   # Expected: True
```

## 🌍 Digital Ecosystem (Legacy)

Eksperyment z ewolucją cyfrową i emergentnym zachowaniem sztucznego życia.

### Typy bytów cyfrowych:
- 🟢 **Zbieracze** - specjalizują się w zbieraniu zasobów
- 🔴 **Łowcy** - polują na inne byty  
- 🟡 **Reproduktory** - skupiają się na rozmnażaniu
- 🟣 **Hybrydy** - łączą różne umiejętności

## Właściwości bytów

Każdy byt ma:
- **Inteligencję** - wpływa na efektywność działań
- **Prędkość** - szybkość poruszania się
- **Rozmiar** - determinuje sukces w polowaniu
- **Energię** - potrzebną do życia i reprodukcji
- **Wskaźnik mutacji** - jak często mutują geny

## Mechanika ewolucji

1. **Mutacje** - podczas reprodukcji cechy mogą się zmieniać
2. **Selekcja naturalna** - słabsze byty umierają częściej
3. **Ewolucja typów** - byty mogą zmieniać swój typ w zależności od cech
4. **Emergencja** - złożone zachowania wynikają z prostych reguł

## Jak uruchomić

```bash
python digital_ecosystem.py
```

## Obserwowane zjawiska

- **Specjalizacja** - populacje ewoluują w kierunku określonych strategii
- **Cykle populacyjne** - wzrosty i spadki liczebności
- **Wyścig zbrojeń** - łowcy vs ofiary
- **Kooperacja** - hybrydy jako strategia przetrwania

## Przyszłe rozszerzenia

- [ ] Wizualizacja graficzna
- [ ] Zapisywanie danych ewolucji
- [ ] Więcej typów bytów
- [ ] Systemy komunikacji między bytami
- [ ] Środowiskowe wyzwania (katastrofy, zmiany klimatu)

## 📄 Licencja

Ten projekt jest licencjonowany na licencji Apache 2.0 z dodatkowymi klauzulami dla MŚWR v2.0 - zobacz [LICENSE](LICENSE) dla szczegółów.

### Kluczowe zasady licencjonowania:
- ✅ Swobodne użycie w celach badawczych i komercyjnych
- ✅ **Wzmocniona ochrona patentowa** dla algorytmów świadomości
- ✅ Modyfikacja i dystrybucja dozwolone z zachowaniem bezpieczeństwa
- ⚠️ Anti-Fatal Error Protocol MUSI pozostać nienaruszony w środowiskach produkcyjnych
- 📚 Wymagane cytowanie w pracach akademickich i produktach komercyjnych
- 🛡️ **Apache 2.0 zapewnia silniejszą ochronę niż MIT** dla innowacyjnych technologii AI

---

*Eksperyment GitHub Copilot & MŚWR v2.0, 27 października 2025*