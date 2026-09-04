# World Trade Radar — AI Sistem Mimari Denetimi

Tarih: 2026-09-01  
Kapsam: mevcut repository (kaynak kod). `node_modules/`, `backend/venv/` ve `.env.local` içeriği rapora alınmadı.  
Yöntem: kod okuma. Tahmin yok; her iddia dosya yoluyla bağlandı.  
Bu belge analiz içindir. Kod değiştirilmedi.

---

## Yönetici özeti

Bu proje **hibrit bir ticari asistan**. Kullanıcıya “ajan + LLM” gibi görünür; gerçekte iki ayrı motor vardır:

1. **Kural motoru (asıl danışmanlık):** Python şablonları (`backend/agents/session.py`) ürün/stok/kapasite slot’larını doldurur, niyeti regex ile sınıflandırır, fiyat/kanal/mail metnini **llama3’e sormadan** üretir. Alıcı/tedarikçi kartları **Ollama bge-m3 + Supabase pgvector RAG** ile gelir.
2. **LLM motoru (yan yol):** Yerel **Ollama `llama3`** yalnızca `sector_chat` ve `product_matching` ajanlarında serbest metin üretir. System prompt `backend/prompts.py` içindeki `VOICE_SYSTEM` + rol prompt’udur.

Ses zinciri: **tarayıcı Web Speech API (STT, bulut/OS)** → metin danışmanlık API’si → **yerel Coqui XTTS v2 (kadın konuşmacı, GPU)** veya Piper yedek (CPU) → WAV.

Hedef persona (“üst düzey CMO + ticari direktör + dış ticaret danışmanı + stratejik iş ortağı”) prompt metninde kısmen vardır; **çalışma zamanında çoğu tur LLM’e hiç gitmez**, şirket CRM bağlamı bağlanmaz, web araştırması yoktur, tool-calling yoktur.

---

## 1. AI modeli

### Hangi modeller?

| Rol | Model | Nerede | Nasıl çağrılır |
|-----|--------|--------|----------------|
| Sohbet / serbest metin | `llama3` (Ollama etiketi) | Yerel Ollama süreci | HTTP `POST {OLLAMA_BASE_URL}/api/chat` (404 ise `/api/generate`) |
| Embedding / RAG | `bge-m3` (1024 boyut) | Yerel Ollama | HTTP `POST …/api/embed` (404 ise `/api/embeddings`) |
| TTS (asıl) | Coqui `tts_models/multilingual/multi-dataset/xtts_v2` | FastAPI süreci, PyTorch | `backend/tts_local.py` |
| TTS (yedek) | Piper `tr_TR-dfki-medium` | FastAPI süreci, ONNX | `backend/models/tts/` |
| STT | Tarayıcı `SpeechRecognition` / `webkitSpeechRecognition` | Kullanıcının tarayıcısı | `lib/speech.ts` |

Kaynak: `backend/main.py` satır 75–77, 185–215, 281–343, 354–396; `backend/tts_local.py` satır 22–24; `lib/speech.ts` satır 87–97.

### Local mi, cloud API mi?

- **llama3 ve bge-m3:** local. Varsayılan `OLLAMA_BASE_URL = http://localhost:11434`. OpenAI / Anthropic / vLLM / llama.cpp / LM Studio istemcisi yok.
- **XTTS / Piper:** local. `tts_local.py` başlığı: “Bulut yok.”
- **STT:** local kod değil. Chromium/Edge Web Speech API; tarayıcı çoğu kurulumda Google/OS tanıma servisine gider (`network` hata metni `components/mic-button.tsx` içinde).

### Altyapı envanteri

| Altyapı | Var mı? |
|---------|---------|
| Ollama | Evet — tek LLM/embedding köprüsü |
| llama.cpp | Hayır (bağımlılık yok) |
| vLLM | Hayır |
| Transformers | Evet, **yalnızca TTS** (`backend/requirements-tts.txt`: `coqui-tts`, `transformers`, `torch`) |
| LM Studio | Hayır |
| Function calling / tools JSON | Hayır; Ollama payload’da `tools` yok |

### Model nerede başlatılıyor?

- **Ollama:** bu repo modeli yüklemez. Operatör `ollama serve` + `ollama pull llama3` + `ollama pull bge-m3` çalıştırır. FastAPI yalnızca HTTP istemcisidir (`lifespan` içinde `httpx.AsyncClient`, timeout 180s: `backend/main.py` 757–765).
- **XTTS / Piper:** FastAPI ayağa kalkınca daemon thread `warmup_tts()` (`main.py` 763). Önce Piper, sonra XTTS (`tts_local.py` `warmup` / `try_xtts`).

### Model nerede çağrılıyor?

- Embedding: `embed_with_ollama` → `/search`, `/embed`, ingest, ve ajan `deps.embed`.
- Chat: `generate_with_llama3` → `AgentDeps.generate` → fiilen `run_sector_chat` ve `run_product_matching`.
- Trade Advisor / Buyer / Supplier / Intake / Chat: **generate çağırmaz**; şablon + RAG.

### Konfigürasyon

| Değişken | Varsayılan | Dosya |
|----------|------------|--------|
| `OLLAMA_CHAT_MODEL` | `llama3` | `backend/main.py` |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | `backend/main.py` |
| `OLLAMA_CHAT_TEMPERATURE` | 0.8, [0.7, 0.9] | `backend/prompts.py` `chat_temperature()` |
| `OLLAMA_EMBED_MODEL` | sabit `"bge-m3"` | `backend/main.py` |
| `TTS_ENGINE` | `auto` | `tts_local.py` |
| `TTS_DEVICE` | cuda varsa cuda | `tts_local.py` `_device()` |
| `TTS_SPEAKER` | yoksa `Tammie Ema` | `tts_local.py` |
| `TTS_PIPER_VOICE` | `tr_TR-dfki-medium` | `tts_local.py` |
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | `lib/api.ts` |
| Supabase URL + `SUPABASE_SERVICE_ROLE_KEY` | kök `.env.local` | `main.py` `load_env_local()` |

Ollama `stream: False`. Token streaming yok. Consult “stream” ajan adımı SSE’sidir.

---

## 2. Ses sistemi

### Kullanıcı sesi nasıl alınıyor?

`components/mic-button.tsx` mikrofon izni ister, `lib/speech.ts` üzerinden Web Speech Recognition başlatır.

- Dil: `SPEECH_LANG = "tr-TR"`
- `continuous: true`, `interimResults: true`
- VAD: 1500 ms sessizlik (`VAD_SILENCE_MS`) sonra `onCommit`
- Echo guard: 350 ms (`VAD_ECHO_GUARD_MS`) — TTS bitince yankıyı yutmak için
- Transcript birleştirme: `composeVoiceTranscript`, `joinUniqueUtterance`

Whisper, Faster-Whisper, Deepgram, Azure Speech **yok**.

### STT dosyaları

- `lib/speech.ts` — API sarmalayıcı, oynatma, VAD sabitleri
- `components/mic-button.tsx` — dinleme yaşam döngüsü
- `components/consult-desk.tsx` — `onCommit={(text) => submit(text, true)}`

### TTS — AI cevabı nasıl sese dönüşüyor?

1. `SpeakButton` (`components/speak-button.tsx`) `speakText(text)` çağırır.
2. `lib/speech.ts` `fetchSpeakAudio` → Next `POST /api/speak`.
3. `app/api/speak/route.ts` FastAPI `POST {API}/speak` proxy (timeout 180s, WAV stream).
4. `backend/main.py` `speak()`: yalnızca `lang` `tr*` kabul; `asyncio.to_thread(synthesize_wav, body.text)`.
5. `tts_local.py` `LocalTTS.synthesize`: cümle böl, kısaltma fonetiği, disk cache (`backend/cache/tts/{sha256}.wav`).

### Kadın sentetik ses nasıl tanımlı?

`PREFERRED_SPEAKERS` sırası (`tts_local.py` 25–31):

1. **Tammie Ema** (varsayılan; XTTS çok dilli kadın konuşmacı)
2. Ana Florence
3. Sofia Hellen
4. Viktor Eka
5. Damien Black

`TTS_SPEAKER` doluysa o isim kullanılır; değilse listedeki ilk mevcut XTTS speaker. Dil her zaman `TTS_LANGUAGE = "tr"`. Piper yedekte `tr_TR-dfki-medium` (DFKI Türkçe, orta kalite; cinsiyet modeli XTTS kadar açık etiketlenmez).

### Local mi cloud mu? Streaming?

- TTS **local**. WAV tamamı üretilir, sonra tarayıcı `HTMLAudioElement` oynatır.
- XTTS sentez streaming yok (cümle chunk’ları birleştirilir, tek WAV).
- Consult SSE, ajan adımları içindir; ses token’ı akmaz.
- Piper `use_cuda=False` — CPU.

### Uçtan uca ses zinciri (koddan)

```
Kullanıcı konuşur
  → MicButton + Web Speech API (tr-TR)     [STT, tarayıcı]
  → textarea + onCommit
  → consultStream POST /consult/stream      [metin]
  → orchestrator → agent → (şablon ve/veya llama3)
  → advice string
  → SpeakButton autoPlay=voiceHandsFree
  → POST /api/speak → FastAPI /speak
  → XTTS v2 GPU (konuşmacı Tammie Ema) veya Piper
  → audio/wav
  → kadın AI sesi
  → onSpeakFinished → 450ms sonra mikrofon yeniden (hands-free)
```

Hands-free: `consult-desk.tsx` `submit(..., fromVoice=true)` → `autoPlay` → TTS bitince `armMicListen()`.

---

## 3. Prompt sistemi

Tek dosya: **`backend/prompts.py`**. Ayrı “developer prompt” ürünü yok; `VOICE_SYSTEM` gizli system talimatıdır (yorum: “yalnızca /api/chat system rolü”).

### Katmanlar

```
VOICE_SYSTEM          (persona + dil kilidi + yasaklar)
    +
rol prompt’u          with_voice(role)
    =
system message        Ollama messages[0]
```

| Sembol | Görev | Kim kullanır? |
|--------|--------|----------------|
| `VOICE_SYSTEM` | Direktör persona, TR kilit, keşif yasağı, sektör kalıpları | Tüm `with_voice(...)` |
| `CONSULT_SYSTEM_PROMPT` | Varsayılan generate system | `generate_with_llama3` default; `AgentDeps.generate` |
| `CHAT_PROMPT` | Selam iki cümle | **Tanımlı; `chat.py` kullanmıyor** (sabit `GREETING_REPLY`) |
| `SECTOR_PROMPT` | Sektör sohbeti | `agents/sector.py` |
| `INTAKE_PROMPT` | Keşif | **Tanımlı; `intake.py` LLM çağırmıyor** |
| `TRADE_ADVISOR_PROMPT` | Direktör notu | **Tanımlı; `trade_advisor.py` şablon kullanıyor** |
| `MATCHING_PROMPT` | Eşleştirme, max 2 cümle | `product_matching.py` |
| `BUYER_PROMPT` / `SUPPLIER_PROMPT` | Filtre kalıbı | **Tanımlı; finder `party_action_reply` kullanıyor** |
| `TURKISH_RETRY_LOCK` | İngilizce kaçarsa ikinci generate | `generate_with_llama3` |
| Sabit cevaplar | `GREETING_REPLY`, `INTAKE_REPLY`, … | Kural ajanları |

`polish_chat_reply()` llama3 çıktısını kırpar (HS sızıntısı, İngilizce, anketör kapanışı). Şablon cevaplar bu filtreden geçmez.

### Context nasıl oluşur?

1. **Oturum notu** `session_notes()`: Hitap, Ürün, Kapasite, Stok, Pazar — system’e eklenir (`main.py` `_consult_deps`).
2. **Sohbet geçmişi** son 24 tur (`MAX_TURNS`), Ollama `messages` listesine.
3. **RAG kart özeti** `format_consult_context()`: 4–6 trade_item satırı (isim, rota, isteğe HS). Product Matching user turn’üne eklenir.
4. **Kullanıcı mesajı** `wrap_user_prompt` / `build_user_turn`: ham soru + not; meta talimat yok.

Şablon ajanlarda (advisor/finder) “context” asıl olarak `matches` listesidir; metin `advisor_answer` / `party_action_reply` içine gömülür.

---

## 4. Agent / orchestrator

Giriş: `run_orchestrator` (`backend/agents/orchestrator.py`).

Akış:

1. `agent_runs` satırı aç (`store.py`)
2. `classify_intent(question, session)` (`intent.py`) — regex + skor, LLM yok
3. `_RUNNERS[intent](question, deps, emit)`
4. `strip_echoed_query`
5. `AgentOutcome` + `agent_runs` güncelle

Ajanlar birbirini mesajlaşarak çağırmaz. Orchestrator tek dispatcher. İstisna: **intake ürünü alınca `run_trade_advisor` çağırır** (`intake.py`).

### Ajan tablosu

| Intent | Dosya | Ne yapar | Ne zaman | Modele bağlanır mı? |
|--------|-------|----------|----------|---------------------|
| `chat` | `agents/chat.py` | Sabit selam veya `continue_intake` | Small talk; ürün cümlesi | Hayır |
| `intake` | `agents/intake.py` | Slot doldur; ürün varsa advisor | Ürün/kapasite, keşif | Advisor şablon; LLM yok |
| `sector_chat` | `agents/sector.py` | Serbest sektör sohbeti, arama yok | Niyet veri araması değil | **llama3 + SECTOR_PROMPT** |
| `trade_advisor` | `agents/trade_advisor.py` | RAG + `advisor_answer` (mediate/rapor/mail/fuar) | Strateji, taslak, satış, ürün sonrası | Embedding evet; **chat hayır** |
| `buyer_finder` | `agents/supplier_finder.py` | RAG alıcı + `party_action_reply` | alıcı/müşteri | Embedding evet; **chat hayır** |
| `supplier_finder` | aynı dosya, `role=supplier` | RAG tedarikçi | tedarikçi/üretici | Embedding evet; **chat hayır** |
| `product_matching` | `agents/product_matching.py` | RAG + llama3 2 cümle | HS/benzer/muadil | **bge-m3 + llama3** |

Niyet sırası (`intent.py`): buyer_search → supplier_search → strategy/method/draft/sell → kapasite follow-up → choice pick → awaiting intake → small talk → general intake → kısa ürün → aksi halde sector_chat veya terim skoru.

`AgentDeps` (`types.py`): `http`, `supabase`, `embed`, `match`, `generate`, `format_context`, eşik, `session`. Bu “tool” değil; Python callback.

Adımlar UI’da `components/agent-flow.tsx` ile gösterilir (SSE `type: step`).

---

## 5. Tool sistemi

**Klasik tool-calling yok.** Ollama isteğinde `tools` / `function_call` yok. Model “search_web()” diyemez.

Fiili “araçlar” orchestrator’un Python’dan çağırdığı fonksiyonlardır:

| İşlem | Kim çağırır | Sonuç tekrar LLM’e gider mi? |
|-------|-------------|------------------------------|
| `embed_with_ollama` | advisor, finder, matching | Hayır (vektör) |
| `match_trade_items` RPC | `gather_match_rows` | Matching’de evet (metin özet); advisor’da şablona |
| `lexical_trade_rows` | `retrieve.py` | Şablon/kart |
| `domain_channel_rows` | 3D / çelik kart enjeksiyonu | Kart + şablon |
| `generate_with_llama3` | sector, matching | Kendisi model |
| Kota `consume_quota` | middleware | Kullanıcıya 429 |
| TTS `synthesize_wav` | `/speak` | Model’e değil, kulağa |

Structured output: Pydantic HTTP şemaları (`ConsultResponse`, `TradeItemMatch`). LLM JSON şeması yok.

---

## 6. Memory

### Kısa süreli (oturum)

`SessionState` (`session.py`): `name`, `product`, `capacity`, `stock`, `market`, `last_ask`, `last_advisor_*`, `messages` (max 24).

Saklama: süreç içi `_STORE: dict[str, SessionState]` — **uvicorn restart siler**. Frontend `history` + `session_id` gönderir; `hydrate()` slot’ları geçmişten `infer_slots` ile yeniden kurar.

Tarayıcı: `lib/consult-sessions.ts` → `localStorage` (`wtr-consult-sessions`).

### Uzun süreli

- `agent_logs`: her `/consult` input/output JSON (advice, context özeti, intent, steps). Kullanıcıya “hafıza” olarak inject edilmez; `/consult/history` ve Geçmiş sayfası okur.
- `agent_runs`: adım günlüğü.
- **Conversation / messages / memory tablosu yok.**
- Cross-session “kullanıcıyı hatırla” yok (isim bile oturum slot’u).

### RAG

Evet, ticaret kalemi RAG’i:

1. `retrieval_query` ürün + sektör genişletmesi (`retrieve.py`)
2. bge-m3 1024-d
3. `match_trade_items` — pgvector cosine, HNSW (`supabase_schema.sql`)
4. Lexical `ilike` yedek
5. `item_matches_product` sektör süzgeci
6. 3D/çelik kanal kartları

Bu, **kullanıcı hafızası RAG’i değil**; küresel `trade_items` semantik aramasıdır.

Embedding DB: Supabase `trade_items.embedding vector(1024)`.

---

## 7. Supabase / database

Şema kaynakları: `supabase_schema.sql`, `backend/billing.sql`, `backend/agent_runs.sql`, `backend/rls_paywall.sql`.

| Tablo | Ne | Kullanıcı/şirket/konuşma karşılığı |
|-------|-----|-------------------------------------|
| `organizations` | İthalatçı/ihracatçı/tüccar kartı | “Firma” — **oturumdaki kullanıcının şirketi değil** |
| `trade_items` | Ürün kalemi + embedding | “Ürün” katalog |
| `agent_logs` | Ajan I/O | Konuşma arşivi (jsonb), mesaj tablosu değil |
| `agent_runs` | Orchestrator run | Ajan izi |
| `plans` | free / pro | Abonelik planı |
| `accounts` | Kiracı, `slug` | Zayıf “kullanıcı”; auth.users yok |
| `subscriptions` | Plan durumu | Abonelik |
| `quota_usage` | Günlük arama/token | Kota |
| `billing_events` | Checkout olayları | Ödeme günlüğü |

**Yok:** `users`, `profiles`, `projects`, `conversations`, `messages`, `memories`, `company_profiles`.

RPC: `match_trade_items`, `consume_quota`.

RLS tüm bu tablolarda açık. Yorum: FastAPI **service_role RLS’i atlar**; anon/authenticated istemci varsayılan politikasız erişemez. Frontend doğrudan Supabase konuşmaz.

Paywall: uygulama katmanı `billing/redact.py` — Free planda e-posta, web, vergi, hacim/değer maskelenir (`rls_paywall.sql` yorumu).

---

## 8. Web araştırması

**Yok.** Tavily, Serper, Brave, Bing, Google Custom Search, scraping ajanı yok.

Ağın kullandığı dış HTTP:

- Ollama localhost
- Supabase REST
- (TTS ilk indirme HuggingFace/Coqui — runtime danışmanlık araması değil)

Arama sonuçları kullanıcıya “kaynak linki” olarak gitmez. Kartlar `trade_items` + sentetik kanal satırlarıdır.

---

## 9. Dosya / veri analizi

Kullanıcı sohbette Excel/CSV/PDF/görsel **yükleyemez**. ConsultDesk file input yok.

İlgili ama sohbet dışı:

- `backend/seed_data.py` + CSV/JSON — operatör seed
- `POST /trade-items/bulk` — JSON kalem listesi, her satır embed edilir (`main.py`)
- PDF/görüntü OCR/VLM yok

AI bu dosyalardan “analiz raporu” çıkarmaz.

---

## 10. Şirket bağlamı

AI, **giriş yapmış firmanın ERP/CRM’ini bilmez**.

Aktarılanlar:

| Veri | Nasıl |
|------|--------|
| Kullanıcının ürünü | Konuşmadan slot (`extract_product_slot`) |
| Kapasite / stok | Aynı |
| Hitap adı | Selam sonrası slot |
| Eşleşen firmalar | RAG `organizations` join + kanal kartları |
| Kullanıcı hesabı | `X-Account-Slug` (default `demo`) — kota; prompt’a şirket profili değil |
| `ConsultRequest.organization_id` | RAG’i bir firmaya filtreleyebilir; UI bunu doldurmaz (`lib/api.ts` consult body’de yok) |

`organizations` tablosu B2B hedef/kaynak firmalardır, “benim şirketim” profili değildir.

---

## 11. Frontend — danışmanlık ↔ AI

Ana yüzey: `app/page.tsx` → `components/consult-desk.tsx`.

| Konu | Dosya |
|------|--------|
| Chat gönder / history | `consult-desk.tsx` `submit`, `messages` |
| Mikrofon | `mic-button.tsx` |
| Voice state (hands-free, dinle, TTS meşgul) | `voiceHandsFree`, `listenKey`, `ttsBusy` |
| Loading | `loading` → “Ajanlar çalışıyor…” |
| Streaming | `lib/api.ts` `consultStream` — SSE `step` / `complete` / `error` |
| Adım UI | `agent-flow.tsx` |
| Mesaj render | `Briefing` içinde user/assistant |
| TTS düğmesi / autoPlay | `speak-button.tsx` |
| Sidebar oturumlar | `chat-sidebar.tsx`, `consult-sessions.ts` |
| API istemcisi | `lib/api.ts` |
| Geçmiş sayfası | `app/gecmis/page.tsx`, `consult-history.tsx` |

LLM token stream yok; tam `advice` `complete` event’inde gelir.

---

## 12. Backend — istek katmanları

Gerçek zincir:

```
ConsultDesk.submit
  → fetch NEXT_PUBLIC_API_URL/consult/stream   (lib/api.ts)
  → FastAPI consult_stream                     (backend/main.py)
  → QuotaMiddleware                            (billing/middleware.py)
  → _run_consult_pipeline
       hydrate(session_id, history)
       _consult_deps (embed, match, generate)
       run_orchestrator
         classify_intent
         selected agent
           [RAG] embed bge-m3 → match_trade_items + lexical
           [çoğu tur] session.py şablon
           [sector/matching] generate_with_llama3 → Ollama llama3
  → SSE steps + ConsultResponse
  → (isteğe bağlı) SpeakButton
       → Next /api/speak → FastAPI /speak → XTTS/Piper
```

Paralel REST: `POST /consult` (SSE’siz), `POST /search` (yalnız RAG), `POST /embed`.

---

## 13. AI’ın beyni — müdahale dosyaları

Davranışı değiştirmek için **öncelik sırası**:

| Dosya | Görevi | Neden kritik |
|-------|--------|----------------|
| `backend/prompts.py` | Persona, yasaklar, niyet regex, polish, Ollama system | LLM turları ve sınıflandırıcı kelimeleri |
| `backend/agents/session.py` | Slot, şablon cevaplar, fiyat playbook, mail | Asıl “danışman” metni buradan |
| `backend/agents/intent.py` | Hangi ajan | Yanlış niyet = yanlış motor |
| `backend/agents/retrieve.py` | RAG query, eşik, sektör kartları | Hangi firmalar görünür |
| `backend/agents/orchestrator.py` | Dispatch | Yeni ajan buraya eklenir |
| `backend/agents/trade_advisor.py` | RAG + `advisor_answer` | Ana ticari tur |
| `backend/agents/supplier_finder.py` | Alıcı/tedarikçi listesi | Buyer Finder |
| `backend/agents/intake.py` | Keşif → advisor köprüsü | Ürün kabulü |
| `backend/agents/chat.py` | Selam | İlk izlenim |
| `backend/agents/sector.py` | llama3 sektör | Serbest sohbet |
| `backend/agents/product_matching.py` | llama3 + RAG | HS/benzer |
| `backend/main.py` | Ollama/Supabase/TTS HTTP, generate sarmalayıcı | Altyapı |
| `backend/agents/types.py` | Intent birliği | Yeni ajan tipi |
| `supabase_schema.sql` | Vektör + RPC | RAG veri modeli |

LLM’e “daha CMO ol” demek **yalnızca sector/matching turlarını** değiştirir. Advisor turu için `session.py` + `prompts.py` VOICE (dokümantasyon) + şablonlar gerekir.

---

## 14. Dokunulmaması gerekenler (ses ve kırılgan yüzey)

Aşağıdakiler çalışır durumda; ticari persona işi için **gerekmedikçe değiştirilmemeli**:

| Dosya | Neden |
|-------|--------|
| `backend/tts_local.py` | GPU XTTS, speaker, cache, Piper fallback; reload XTTS thread kilitler |
| `backend/models/tts/` | Piper ONNX |
| `app/api/speak/route.ts` | Proxy; timeout/header kırılırsa ses kesilir |
| `lib/speech.ts` | STT + echo guard + nesil iptali |
| `components/mic-button.tsx` | VAD / commit |
| `components/speak-button.tsx` | autoPlay döngüsü |
| `backend/requirements-tts.txt` | CUDA/torch pin |
| `backend/billing/middleware.py` | Kota; yanlış dokunuş 429 |
| `backend/billing/redact.py` | Pro gizlilik |
| CORS/`QuotaMiddleware` sırası `main.py` | İstek kırılır |
| `.env.local` | Secret |

Frontend layout (`site-header`, PWA, paywall UI) persona ile ilgili değil.

---

## 15. Performans — RTX A6000 ve 256 GB RAM

Bu repo **A6000’i isimle bağlamaz**. Kullanım süreçlere göre:

### GPU (kodun bağladığı yer)

**XTTS:** `torch.cuda.is_available()` ise `TTS(... gpu=True).to("cuda")` (`tts_local.py`). Canlı log: `XTTS yüklendi (cuda, dil=tr, konuşmacı=Tammie Ema)`. Bu, A6000 üzerinde **gerçek inference**. XTTS v2 çok dilli; VRAM onlarca GB’ı doldurabilir; 256 GB sistem RAM’i model ağırlığı için değil, CPU yedeği / Piper / Ollama kv için.

**Piper:** `use_cuda=False` — CPU.

**llama3 / bge-m3:** FastAPI GPU’ya tensor koymaz. Ollama ayrı süreç; CUDA varsa Ollama kendi GPU’sunu kullanır (Modelfile/`OLLAMA_NUM_GPU` bu repo’da yok). Sağlık kontrolü yalnızca HTTP `/api/tags`.

Sonuç: **GPU kesinlikle TTS için kullanılıyor.** LLM GPU’su Ollama kurulumuna bağlı; uygulamadan görünmez.

### CPU / RAM

- FastAPI + httpx + Supabase istemcisi: hafif
- `asyncio.to_thread(synthesize_wav)`: TTS CPU/GPU’yu bloklamadan thread
- `_STORE` sohbetleri RAM
- WAV cache disk
- 256 GB: bu kod tabanı doldurmaz; Ollama büyük llama3 + XTTS + tarayıcı yan yana sığar

### I/O

- Her danışmanlık: 1+ Ollama embed (1024-d) + RPC + isteğe lexical
- llama3: sector/matching’de ek 180s timeout
- Kota: arama + `prompt_eval_count+eval_count` (Ollama) veya karakter/4 tahmini

---

## 16. Güvenlik

| Konu | Durum |
|------|--------|
| API key | Kök `.env.local`; `load_env_local` `os.environ` yazar. Frontend `NEXT_PUBLIC_*` tarayıcıya gider. Service role **yalnızca backend**. |
| Secret raporu | Bu belge anahtar içermez. |
| Auth | Supabase Auth / JWT login yok. Kimlik = `X-Account-Slug` (localStorage, default `demo`). Sahte slug = başka kota. |
| Kullanıcı izolasyonu | Session RAM + localStorage. `agent_logs` listesi hesap filtresiz (service_role). |
| RLS | Açık; API bypass. Doğrudan anon key ile tablo okunmamalı. |
| Prompt injection | `wrap_user_prompt` talimat eklemez; `VOICE_SYSTEM` kullanıcının system’e yazmasını engellemez (kullanıcı içeriği user rolünde). `strip_echoed_query` echo’yu keser. |
| CORS | `allow_origins=["*"]` + credentials — gevşek. |
| TTS | Yalnızca TR. |
| Paywall | Redaksiyon uygulama katmanı; DB’de satır durur. |

---

## 17. Hedef AI ile boşluklar

Hedef: üst düzey CMO + ticari direktör + dış ticaret danışmanı + stratejik iş ortağı.

| Beklenti | Mevcut | Boşluk |
|----------|--------|--------|
| Persona | VOICE_SYSTEM’de direktör/CMO tonu var | Advisor turunda LLM persona **uygulanmaz**; sabit playbook |
| Reasoning | llama3 0.8 temperature, 2–3 cümle polish | Çok adımlı düşünme, rakip simülasyonu, senaryo yok |
| Memory | 24 tur + slot | Şirket hafızası, karar günlüğü, CRM yok |
| Orchestrator | Regex dispatcher | Planlayan meta-ajan, tool loop yok |
| Tools | Python RAG | Web, e-posta gönder, CRM yaz, fiyat API yok |
| RAG | Ticaret kalemleri | Şirket dokümanı, politika, müşteri geçmişi yok |
| Şirket context | Slot’lar | Profil, katalog, hedef pazar kaydı yok |
| Web research | Yok | Pazar haberi / gümrük güncellemesi yok |
| Veri analizi | Yok | Excel/PDF/KPI yok |
| Çok turlu strateji | Rapor şablonu 3 blok | Dinamik CMO brifingi yok |
| Öğrenme | Yok | Feedback → prompt/RAG yok |
| Ses | Güçlü local TTS | STT tarayıcıya bağımlı; Whisper yok |
| Çok dillilik | Cevap TR kilit; mail EN/DE şablon | Gerçek çok dilli danışmanlık yok |

Persona “eksik” değil; **yanlış katmana yazılı**. Beyin `session.py` şablonları + ince RAG; llama3 dekor.

---

## 18. Sistem özeti (koddan, tahmin değil)

```
KULLANICI
  ↓
SES ALMA
  components/mic-button.tsx  (getUserMedia dolaylı, Web Speech)
  ↓
SPEECH-TO-TEXT
  lib/speech.ts  → window.webkitSpeechRecognition  (tr-TR, tarayıcı servisi)
  ↓
CONTEXT
  ConsultDesk history (localStorage) + session_id
  backend hydrate → SessionState slot’ları
  session_notes() (ürün/stok/kapasite)
  ↓
MEMORY
  RAM _STORE + last 24 messages
  (kalıcı: agent_logs jsonb; retrieval’a bağlı değil)
  ↓
ORCHESTRATOR
  agents/orchestrator.py ← classify_intent (intent.py, regex)
  ↓
AGENT / “TOOL”
  chat | intake | sector_chat | trade_advisor
  | buyer_finder | supplier_finder | product_matching
  RAG tool: retrieve.gather_match_rows
            → Ollama bge-m3 + Supabase match_trade_items + ilike
  ↓
LOCAL MODEL
  Çoğu tur: model yok (session.py şablon)
  sector_chat / product_matching: Ollama llama3 (localhost:11434)
  Embedding her RAG turunda: Ollama bge-m3
  Cloud chat API: yok
  ↓
AI RESPONSE
  advice string + context kartları
  SSE complete → Briefing
  ↓
TEXT-TO-SPEECH
  SpeakButton → /api/speak → FastAPI /speak
  tts_local.py  XTTS v2 (cuda) konuşmacı Tammie Ema
  yedek Piper tr_TR-dfki-medium (CPU)
  ↓
KADIN AI SESİ
  WAV playback (HTMLAudioElement)
```

---

## 19. Dosya listesi

### AI CORE
- `backend/main.py` — Ollama köprüsü, consult pipeline, health
- `backend/prompts.py` — system/persona, polish, temperature
- `backend/agents/session.py` — şablon beyin, slot, playbook
- `backend/agents/types.py` — Intent, AgentDeps

### VOICE
- `backend/tts_local.py`
- `backend/models/tts/` (Piper)
- `app/api/speak/route.ts`
- `lib/speech.ts`
- `components/mic-button.tsx`
- `components/speak-button.tsx`
- `backend/requirements-tts.txt`

### PROMPTS
- `backend/prompts.py` (`VOICE_SYSTEM`, `*_PROMPT`, sabit reply, regex niyet yardımcıları)

### ORCHESTRATOR
- `backend/agents/orchestrator.py`
- `backend/agents/intent.py`
- `backend/agents/store.py`

### AGENTS
- `backend/agents/chat.py`
- `backend/agents/intake.py`
- `backend/agents/sector.py`
- `backend/agents/trade_advisor.py`
- `backend/agents/supplier_finder.py`
- `backend/agents/product_matching.py`
- `components/agent-flow.tsx`

### TOOLS
- `backend/agents/retrieve.py` (RAG + lexical + kanal kartları)
- `backend/main.py` `match_trade_items`, `embed_with_ollama`
- `supabase_schema.sql` `match_trade_items` RPC  
(Function-calling tool listesi yok.)

### MEMORY
- `backend/agents/session.py` (`_STORE`, `hydrate`, `session_notes`)
- `lib/consult-sessions.ts`
- `backend/main.py` `log_agent` → `agent_logs`

### DATABASE
- `supabase_schema.sql`
- `backend/billing.sql`
- `backend/agent_runs.sql`
- `backend/rls_paywall.sql`
- `backend/seed_data.py`

### WEB
- (yok)

### FRONTEND
- `app/page.tsx`
- `components/consult-desk.tsx`
- `components/chat-sidebar.tsx`
- `components/consult-history.tsx`
- `lib/api.ts`
- `lib/quota.ts`
- `components/billing-provider.tsx`

### BACKEND
- `backend/main.py`
- `backend/billing/` (`engine.py`, `middleware.py`, `redact.py`, `meter.py`)
- `backend/requirements.txt`

---

## 20. Sonuç — minimum dokunuş listesi

**Soru:** Bu projeyi hedeflenen üst düzey ticari AI danışmana dönüştürmek için minimum hangi dosyalarda değişiklik yapılmalı?

**Cevap:** Ses yığınına dokunmadan, davranış üç katmanda değişir. Minimum set:

1. **`backend/agents/session.py`**  
   Şablon danışman (fiyat, kanal, mail, mediate). CMO/direktör muhakemesi buraya veya buradan LLM’e taşınmadan “canlı” olmaz.

2. **`backend/prompts.py`**  
   Persona, yasaklar, niyet ipuçları. llama3’ün gerçekten konuştuğu turlar için tek system kaynağı.

3. **`backend/agents/trade_advisor.py` (+ isteğe `supplier_finder.py`)**  
   Bugün llama3’ü atlıyor. Stratejik muhakeme isteniyorsa RAG kartlarını `generate`’e vermek veya hibrit (şablon iskelet + LLM gövde) burada bağlanır.

4. **`backend/agents/intent.py` + `orchestrator.py`**  
   Yeni yetenek (web, şirket belgesi, analiz) yeni intent/ajan olmadan regex’e sığmaz.

5. **`backend/agents/retrieve.py` + `supabase_schema.sql`**  
   Şirket/ürün/müşteri hafızası RAG’e girecekse vektör şeması ve süzgeç burada.

6. **`backend/main.py` `_consult_deps` / `generate_with_llama3`**  
   Şirket profili, uzun bellek, tool sonuçlarını system’e ekleme noktası; Ollama tool-calling ileride buraya.

**Bilerek hariç (minimum yol):** `tts_local.py`, `lib/speech.ts`, mic/speak bileşenleri, billing middleware.

**Yeni veri olmadan yapılamayanlar:** web research, Excel/PDF analizi, gerçek şirket CRM’i — yeni tablolar + yeni ajan dosyaları gerekir; mevcut 6 dosya yetmez.

Özet cümle: **ses ve RAG altyapısı hazır; “beyin” büyük ölçüde kural motoru. Hedef danışman için önce `session.py` + advisor’ın LLM/context köprüsü, sonra bellek ve tool’lar.**
