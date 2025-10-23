# Meta-Genius Digital Empire
## Zunifikowany ekosystem wszystkich projektów

Witaj w **Meta-Genius Digital Empire** - centralnym hub'ie łączącym wszystkie twoje 22 repozytoria w jeden działający ekosystem.

## 🎯 GŁÓWNY CEL

Tworzymy **jeden portal** z wieloma modułami:
- **MTA Quest** - AI Life Optimizer (główny produkt)
- **God Interface** - panel operatora MetaGeniusz
- **Content Universe** - publikowanie treści
- **Drift Economy** - tokenizacja i monetyzacja
- **Portfolio Hub** - showcase wszystkich projektów

## 🏗️ ARCHITEKTURA

```
┌─────────────────────────────────────────┐
│           Unified Gateway               │
│         (unified_gateway.py)            │
│              Port 8800                  │
└─────────────┬───────────────────────────┘
              │
    ┌─────────┼─────────┐
    │         │         │
┌───▼───┐ ┌───▼───┐ ┌───▼───┐
│ CORE  │ │CONTENT│ │ECONOMY│
│       │ │       │ │       │
│ MIGI  │ │ MTA   │ │ DRIFT │
│ GOK   │ │ RFG   │ │ TOKEN │
│ GOD   │ │ HHU   │ │       │
└───────┘ └───────┘ └───────┘
```

## 📁 STRUKTURA PLIKÓW

```
nowy_folder/                     # Tu jesteś teraz
├── workspace.json               # ✅ Mapa wszystkich 22 repo
├── sync_repos.py               # ✅ Skrypt synchronizacji
├── unified_gateway.py          # ✅ Główny gateway
├── pdf_rag_ingest.py          # ✅ System RAG
├── mta_quest_api.py           # ✅ Główny produkt
├── meta_genius_unified_system.py # ✅ MGUS core
├── templates/index.html       # ✅ Landing page
└── repos/                     # Tu będą wszystkie repo
    ├── m-zg_Boga/
    ├── GOK-AI-MixTape/
    ├── rocket_fuell_girls/
    └── ... (20 innych repo)
```

## 🚀 JAK URUCHOMIĆ

### 1. Synchronizacja repozytoriów
```bash
python sync_repos.py
```

### 2. Uruchomienie głównego produktu (MTA Quest)
```bash
python mta_quest_api.py
# Otwórz: http://localhost:5000
```

### 3. Uruchomienie Unified Gateway
```bash
# Zainstaluj zależności
pip install fastapi uvicorn httpx

# Uruchom gateway
python unified_gateway.py
# Otwórz: http://localhost:8800
```

### 4. Przetwarzanie dokumentów do RAG
```bash
# Stwórz folder docs/pdfs i wrzuć tam swoje PDF
mkdir -p docs/pdfs

# Przetworz dokumenty
python pdf_rag_ingest.py docs/pdfs --output data/rag_index.jsonl

# Testuj wyszukiwanie
python pdf_rag_ingest.py docs/pdfs --search "MIGI" --output data/rag_index.jsonl
```

## 🗺️ MAPA SERWISÓW

| Serwis | Port | Opis | Status |
|--------|------|------|--------|
| **MTA Quest** | 5000 | AI Life Optimizer (główny produkt) | ✅ Działający |
| **Gateway** | 8800 | Unified API Gateway | ✅ Gotowy |
| **God Interface** | 8001 | Panel operatora MetaGeniusz | 🔨 W budowie |
| **MIGI Core** | 8004 | Planetary intelligence | 🔨 W budowie |
| **Hip-Hop Universe** | 8005 | Muzyka i content | 🔨 W budowie |
| **Rocket Fuel Girls** | 8003 | Adult content (compliance required) | ⚠️ Wymaga weryfikacji |
| **Drift Money** | 8007 | Token economy | 🔨 W budowie |
| **Portfolio** | 8008 | Showcase projektów | 🔨 W budowie |

## 🔐 SYSTEM RÓL

```
MetaGeniusz (5)    ← TY - pełen dostęp do wszystkiego
     ↓
GPT-President (4)  ← AI agent najwyższego szczebla
     ↓  
GPT-King (3)       ← AI agent regionalny
     ↓
GPT-Organization (2) ← AI agent instytucjonalny
     ↓
User (1)           ← Zwykły użytkownik
```

## 🛣️ ENDPOINTS GATEWAY

### Core System
- `GET /health` - Status gateway
- `GET /services` - Lista wszystkich serwisów
- `GET /god/dashboard` - God Interface (MetaGeniusz only)
- `GET /god/migi/state` - Stan systemu MIGI

### Główny Produkt
- `GET /` - MTA Quest landing page  
- `POST /api/success-probability` - Kalkulator sukcesu
- `POST /api/quick-insight` - Szybka analiza AI

### Content & Media  
- `GET /mixtape/latest` - Najnowsze miksy
- `GET /hhu/tracks` - Hip-hop utwory
- `GET /rfg/gallery` - Rocket Fuel Girls (age verification)

### Economy
- `GET /drift/balance/{user_id}` - Saldo tokenów
- `POST /drift/transfer` - Transfer tokenów

### Admin
- `GET /admin/audit` - Audyt systemu (GPT-King+)
- `GET /admin/ecosystem-status` - Status ekosystemu (GPT-President+)

## 📊 PLAN ROZWOJU

### 🟢 PHASE 1 - FOUNDATION (Gotowe)
- ✅ MTA Quest MVP działający
- ✅ Unified Gateway struktura
- ✅ Workspace mapping 22 repo
- ✅ RAG system dla dokumentów

### 🔵 PHASE 2 - INTEGRATION (30 dni)
- 🔨 God Interface z MIGI dashboard
- 🔨 Hip-Hop Universe content
- 🔨 Portfolio showcase  
- 🔨 Basic Drift Economy

### 🟡 PHASE 3 - CONTENT (60 dni)
- ⚠️ Rocket Fuel Girls (compliance first!)
- 🔨 Anonymous Agent 777
- 🔨 Global Vision monitoring
- 🔨 PinkPlay ecosystem

### 🟠 PHASE 4 - SCALE (90 dni)
- 🔨 Multi-user support
- 🔨 Payment processing
- 🔨 Mobile apps
- 🔨 Public API

## 🌍 DOMENY DOCELOWE

```
mtaquestwebskidx.com           # Główna domena
├── god.mtaquestwebskidx.com   # God Interface
├── api.mtaquestwebskidx.com   # API Gateway  
├── hub.mtaquestwebskidx.com   # Content Hub
├── drift.mtaquestwebskidx.com # Token Economy
└── portfolio.mtaquestwebskidx.com # Portfolio
```

## ⚠️ COMPLIANCE WARNINGS

**Rocket Fuel Girls & PinkPlay** wymagają:
- ✅ Weryfikacja wieku (18+)
- ✅ Content moderation
- ✅ GDPR compliance 
- ✅ DSA transparency reporting
- ✅ Notice-and-action procedures

**Nie uruchamiaj bez compliance!**

## 🎯 NASTĘPNE KROKI

1. **Uruchom synchronizację**: `python sync_repos.py`
2. **Testuj MTA Quest**: `python mta_quest_api.py` 
3. **Sprawdź gateway**: `python unified_gateway.py`
4. **Dodaj swoje PDF**: Skopiuj analizy do `docs/pdfs/`
5. **Przetestuj RAG**: `python pdf_rag_ingest.py docs/pdfs --search "GOK"`

---

**Meta-Genius Digital Empire** - gdzie 22 repozytoria staje się jednym ekosystemem. 🚀