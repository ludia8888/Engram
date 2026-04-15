# Engram 개발 설계 문서 v2.3

**The Physical Trace of AI Memory**

*Embedded, event-sourced long-term memory for LLM applications*

---

## 목차

1. [Executive Summary](#1-executive-summary)
2. [Quick Start](#2-quick-start)
3. [설계 철학](#3-설계-철학)
4. [시스템 아키텍처](#4-시스템-아키텍처)
5. [데이터 모델](#5-데이터-모델)
6. [저장소 스키마](#6-저장소-스키마)
7. [Public API](#7-public-api)
8. [타입 시스템](#8-타입-시스템)
9. [Write Path: 추출 파이프라인](#9-write-path-추출-파이프라인)
10. [Read Path: Retrieval 파이프라인](#10-read-path-retrieval-파이프라인)
11. [Context Builder](#11-context-builder)
12. [이벤트 소싱 + 스냅샷](#12-이벤트-소싱--스냅샷)
13. [Raw Log (Tier 1)](#13-raw-log-tier-1)
14. [히스토리와 시간 질의](#14-히스토리와-시간-질의)
15. [Actions 경계](#15-actions-경계)
16. [멀티테넌시와 보안 경계](#16-멀티테넌시와-보안-경계)
17. [동시성 및 라이프사이클](#17-동시성-및-라이프사이클)
18. [성능 목표와 측정 원칙](#18-성능-목표와-측정-원칙)
19. [파일 구조](#19-파일-구조)
20. [구현 순서](#20-구현-순서)
21. [테스트 전략](#21-테스트-전략)
22. [벤치마크 계획](#22-벤치마크-계획)
23. [알려진 한계](#23-알려진-한계)
24. [Appendix: 주요 코드 스켈레톤](#24-appendix-주요-코드-스켈레톤)

---

## 1. Executive Summary

### 1.1 한 줄 정의

```text
Engram = Raw Log를 보존하는 embedded memory engine
       + canonical event store
       + rebuildable projections
       + known-time / valid-time retrieval
```

### 1.2 v2.3의 핵심 재정의

v2.3는 "기능을 더 넣는 버전"이 아니라 "무엇이 진실이고 무엇이 파생인지"를 다시 고정하는 버전이다.

- `Raw Log`는 사용자가 실제로 한 대화의 영구 기록이다.
- `Canonical Store`는 대화에서 추출된 구조화 이벤트의 원장이다.
- `Projection`은 현재 상태, FTS, 벡터 인덱스처럼 다시 만들 수 있는 가속 계층이다.
- 시간 질의는 하나가 아니다.
  - `known-time`: 시스템이 그 시점에 실제로 알고 있던 상태
  - `valid-time`: 지금 기준으로 재구성한, 그 시점에 실제로 유효했을 가능성이 있는 상태

### 1.3 기존 문서 대비 가장 큰 변경점

| 축 | v2.2 | v2.3 |
|---|---|---|
| 시간 모델 | `occurred_at` 하나에 의미 과적재 | `observed_at`, `effective_at_*`, `recorded_at` 분리 |
| 시간 API | `get_at()` 단일 API | `get_known_at()`, `get_valid_at()` 분리 |
| projection | DB tx 안에서 증분 apply 암시 | commit 후 rebuild / immutable swap |
| reprocess | old run inactive + 새 run active | lineage와 projection rebuild를 분리 |
| action writeback | core 문서에 깊게 결합 | companion spec로 분리 |
| 동시성 | 암묵적 단일 프로세스 가정 | 단일 writer + file lock + 분리 reader 명시 |

### 1.4 Engram의 역할과 한계

```text
Engram = 구조화된 장기 기억 엔진

Engram core는:
- 원본 대화를 영구 저장한다.
- 대화에서 구조화 이벤트를 만든다.
- 현재 상태와 관련 기억을 검색한다.
- "언제 알게 되었는가"와 "언제 유효한가"를 구분한다.

Engram core는 하지 않는다:
- 범용 문서 RAG를 대체하지 않는다.
- 분산 다중 writer 시스템을 지향하지 않는다.
- action writeback의 운영 복구까지 core 불변식에 포함하지 않는다.
```

---

## 2. Quick Start

### 2.1 현재 구현 기준 설치

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

현재 코드 기준:
- 패키지 메타데이터 이름은 `engram`
- Python `>=3.12`
- `sqlite3`
- `pytest`, `hypothesis`는 개발용 의존성

로드맵 의존성:
- `sqlite-vec` 또는 호환 벡터 확장
- 로컬 임베딩 또는 API 임베딩 제공자
- 추출용 LLM provider

### 2.2 현재 구현된 최소 사용 예

현재 저장소 코드에서 실제로 동작하는 API는 아래 범위다.

- `turn()`
- `append()`
- `flush()`
- `get()`
- `get_known_at()`
- `known_history()`
- `search()`
- `context()`
- `raw_get()`, `raw_recent()`

주의:
- 현재 런타임의 `append()`는 `entity.*`와 `relation.*` 이벤트를 모두 허용한다.
- 다만 relation 전용 public API는 아직 없고, 현재 relation read는 `search()`, `context()`, internal projection에서만 반영된다.
- 현재 active relation은 양 endpoint entity가 같은 mode/time 기준에서 모두 유효할 때만 보인다.

```python
from engram import Engram

mem = Engram(user_id="alice")

ack = mem.turn(
    user="난 채식주의자야. 다음 주 도쿄 여행 가.",
    assistant="알겠습니다. 채식 가능한 도쿄 식당도 추천할게요.",
)

# 지금 보장되는 건 raw durability와 enqueue 결과
print(ack.turn_id, ack.durable_at, ack.queued)

# 현재 기본 extractor는 no-op이므로,
# structured memory가 필요하면 append() 또는 사용자 extractor + flush("canonical")를 쓴다.
mem.append(
    "entity.create",
    {
        "id": "user:alice",
        "type": "user",
        "attrs": {"diet": "vegetarian"},
    },
    source_role="manual",
    time_confidence="exact",
)

current = mem.get("user:alice")
past = mem.get_known_at("user:alice", ack.durable_at)
history = mem.known_history("user:alice")
```

### 2.3 현재 구현에서 중요한 사용 규칙

비개발자 관점에서 쉽게 말하면:

- `turn()`은 "메모리 엔진이 대화를 잃어버리지 않게 저장했다"는 뜻이다.
- `ack.queued=True`면 다음 단계 처리를 위해 큐에 들어갔다는 뜻이다.
- `ack.queued=False`면 원본 저장은 성공했지만 큐에는 못 들어간 상태다.
- 현재 구현에서는 `turn()`만으로 canonical 이벤트가 자동 생성되지는 않는다.
- `flush("canonical")`은 현재 큐에 있는 raw turn을 configured extractor로 처리한다.
- 기본 extractor는 no-op이라 extraction run만 남기고 event는 만들지 않는다.
- 현재 구조화 메모리는 `append()`와 `get_known_at()` 중심으로 검증된다.
- `flush("projection")`은 이미 커밋된 canonical 이벤트만 내부 projection snapshot에 반영한다.
- 앱이 다시 시작되면 raw에만 있고 아직 `현재 extractor version` 기준 successful extraction run이 없는 turn은 startup catch-up으로 다시 큐에 올라간다.
- `search()`와 `context()`는 현재 `known`/`valid` 모드와 pluggable semantic index까지 구현됐고, semantic은 현재 embedder version 기준 `vec_events`를 사용한다.

### 2.4 로드맵 예시 API

아래 예시는 문서가 목표로 하는 미래형 사용 방식이며, 현재 Phase 1 코드에는 아직 구현되지 않았다.

```python
class ChatBot:
    def __init__(self, user_id: str):
        self.mem = Engram(user_id=user_id)
        self.history = []

    async def chat(self, user_input: str) -> str:
        memory_context = self.mem.context(
            query=user_input,
            time_mode="known",
            max_tokens=2000,
        )

        messages = [
            {"role": "system", "content": memory_context},
            *self.history,
            {"role": "user", "content": user_input},
        ]
        response = await llm.generate(messages)

        self.mem.turn(user_input, response)

        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": response})
        return response
```

---

## 3. 설계 철학

### 3.1 원칙

1. **Raw first**: 원본 대화는 절대 버리지 않는다.
2. **Canonical before projection**: DB 원장이 먼저고, 상태 캐시와 인덱스는 나중이다.
3. **Known-time와 valid-time을 섞지 않는다**.
4. **Projection은 rebuild 가능해야 한다**.
5. **Reprocess는 정상 기능이지 예외 복구가 아니다**.
6. **동시성 계약을 문서에 먼저 적는다**.
7. **불확실한 시간은 억지로 확정하지 않는다**.
8. **비용보다 정합성을 먼저 맞춘다**.

### 3.2 비원칙

- "항상 완전한 시간여행"을 약속하지 않는다.
- "추출기가 알아서 실제 사건 시간을 다 맞춘다"고 가정하지 않는다.
- "action writeback까지 core 한 문서로 안전하게 끝난다"고 주장하지 않는다.
- "동일 사용자에 대한 다중 writer"를 지원하지 않는다.

### 3.3 핵심 일관성 계약

| 레벨 | 의미 | 예시 |
|---|---|---|
| L0 Raw Durable | 대화가 원본 로그에 저장됨 | `turn()` 반환 |
| L1 Canonical | extraction run + event + mapping이 커밋됨 | `flush(level="canonical")` |
| L2 Projection | state/fts/vector가 rebuild되어 조회에 반영됨 | `flush(level="projection")` |
| L3 Index Fresh | semantic/entity index까지 최신 | `flush(level="index")` |

이 문서의 중요한 약속은 API마다 어느 레벨까지 보장하는지 명시하는 것이다.

---

## 4. 시스템 아키텍처

### 4.1 상위 구조

```text
Application
  |
  v
Engram Core
  |
  +-- L0 Raw Ingest
  |     - segmented raw log
  |     - durable ack
  |
  +-- L1 Canonical Store
  |     - extraction_runs
  |     - events
  |     - event_entities
  |     - dirty_ranges
  |
  +-- L2 Rebuildable Projections
  |     - state snapshots
  |     - relation index
  |     - FTS
  |     - vector index
  |
  +-- Read Path
  |     - event-seeded retrieval
  |     - context builder
  |
  +-- Recovery / Catch-up
        - raw gap scanner
        - projector rebuild
```

### 4.2 계층별 책임

#### L0 Raw Ingest
- `RawTurn`을 세그먼트 파일에 append
- checksum/manifest 갱신
- durable ack 반환

#### L1 Canonical Store
- `ExtractionRun`
- `Event`
- `event_entities`
- `dirty_ranges`

여기까지가 "정답의 근거"다.

#### L2 Projection
- 현재 상태 캐시
- 관계 캐시
- entity FTS
- event vector index

여기는 "빠르게 보기 위한 복사본"이다.

### 4.3 Actions의 위치

v2.3에서 action writeback은 core 내부에 억지로 결합하지 않는다.

- core는 action 감사 이벤트를 받아 저장할 수 있다.
- 실제 승인, lease, reconcile, external idempotency는 companion spec에서 다룬다.

---

## 5. 데이터 모델

### 5.1 Event

```python
@dataclass
class Event:
    id: str
    seq: int
    observed_at: datetime
    effective_at_start: datetime | None
    effective_at_end: datetime | None
    recorded_at: datetime
    type: str
    data: dict[str, Any]
    extraction_run_id: str | None
    source_turn_id: str | None
    source_role: Literal["user", "assistant", "tool", "system", "manual"]
    confidence: float | None
    reason: str | None
    time_confidence: Literal["exact", "inferred", "unknown"]
    caused_by: str | None
    schema_version: int
```

핵심 규칙:

- `observed_at`은 "그 사실을 시스템이 들은 시간"이다.
- `effective_at_*`는 "그 사실이 실제 세계에서 적용되는 시간"이다.
- `recorded_at`은 "이 이벤트가 DB에 커밋된 시간"이다.
- 추출기가 실제 시간을 모르겠으면 `effective_at_*`를 비워 둔다.
- `known-time` 질의는 `recorded_at` 기준이다.
- `valid-time` 질의는 `effective_at_*` 기준이다.

### 5.2 ExtractionRun

```python
@dataclass
class ExtractionRun:
    id: str
    source_turn_id: str
    extractor_version: str
    observed_at: datetime
    processed_at: datetime
    status: Literal["SUCCEEDED", "FAILED", "SKIPPED"]
    error: str | None
    event_count: int
    superseded_at: datetime | None
    projection_version: int | None
```

run visibility 규칙:

- `current-active run`: `superseded_at is null`
- `known-time visible run @ t`: `processed_at <= t` 이고 (`superseded_at is null` 또는 `superseded_at > t`)
- `valid-time current run`: 현재 시점에서 active인 run

### 5.3 RawTurn

```python
@dataclass
class RawTurn:
    id: str
    session_id: str | None
    observed_at: datetime
    user: str
    assistant: str
    metadata: dict[str, Any]
```

### 5.4 Entity와 Temporal View

```python
@dataclass
class Entity:
    id: str
    type: str
    attrs: dict[str, Any]
    created_recorded_at: datetime
    updated_recorded_at: datetime

@dataclass
class TemporalEntityView:
    entity_id: str
    entity_type: str
    attrs: dict[str, Any]
    unknown_attrs: list[str]
    supporting_event_ids: list[str]
    basis: Literal["known", "valid"]
    as_of: datetime
```

`TemporalEntityView`를 둔 이유는 `valid-time`이 항상 완벽히 확정적이지 않기 때문이다.

### 5.5 Relation

```python
@dataclass
class Relation:
    source: str
    target: str
    type: str
    attrs: dict[str, Any]
    observed_at: datetime
    effective_at_start: datetime | None
    effective_at_end: datetime | None
```

### 5.6 Event Payload 규칙

payload는 전부 `attrs` 중첩 형태를 사용한다.

```python
"entity.create"  -> {"id": "...", "type": "...", "attrs": {...}, "meta": {...}?}
"entity.update"  -> {"id": "...", "attrs": {...}, "meta": {...}?}
"entity.delete"  -> {"id": "...", "meta": {...}?}

"relation.create" -> {"source": "...", "target": "...", "type": "...", "attrs": {...}, "meta": {...}?}
"relation.update" -> {"source": "...", "target": "...", "type": "...", "attrs": {...}, "meta": {...}?}
"relation.delete" -> {"source": "...", "target": "...", "type": "...", "meta": {...}?}
```

중요:
- `_changed_from` 같은 파생 정보는 canonical payload에 저장하지 않는다.
- 히스토리 조회 시 replay로 계산한다.

### 5.7 EventEntity

```python
@dataclass
class EventEntity:
    event_id: str
    entity_id: str
    role: Literal["subject", "source", "target", "owner", "mentioned"]
```

### 5.8 DirtyRange

```python
@dataclass
class DirtyRange:
    owner_id: str
    from_recorded_at: datetime | None
    from_effective_at: datetime | None
    reason: str
```

### 5.9 Snapshot

```python
@dataclass
class Snapshot:
    id: str
    basis: Literal["known"]
    created_at: datetime
    last_seq: int
    projection_version: int
    max_recorded_at_included: datetime
    max_effective_at_included: datetime | None
    state_blob: bytes
    relation_blob: bytes
```

v2.3에서 필수 스냅샷은 `known` basis만 요구한다.
`valid` basis snapshot은 최적화 옵션이다.

### 5.10 TurnAck

```python
@dataclass
class TurnAck:
    turn_id: str
    observed_at: datetime
    durable_at: datetime
    queued: bool
```

---

## 6. 저장소 스키마

### 6.1 핵심 SQLite 테이블

```sql
CREATE TABLE extraction_runs (
    id                 TEXT PRIMARY KEY,
    source_turn_id     TEXT NOT NULL,
    extractor_version  TEXT NOT NULL,
    observed_at        TEXT NOT NULL,
    processed_at       TEXT NOT NULL,
    status             TEXT NOT NULL CHECK(status IN ('SUCCEEDED', 'FAILED', 'SKIPPED')),
    error              TEXT,
    event_count        INTEGER NOT NULL,
    superseded_at      TEXT,
    projection_version INTEGER
);

CREATE UNIQUE INDEX uq_run_dedupe
    ON extraction_runs(source_turn_id, extractor_version)
    WHERE status = 'SUCCEEDED';

CREATE TABLE superseded_runs (
    old_run_id      TEXT PRIMARY KEY,
    new_run_id      TEXT NOT NULL,
    superseded_at   TEXT NOT NULL
);

CREATE TABLE events (
    id                  TEXT PRIMARY KEY,
    seq                 INTEGER NOT NULL UNIQUE,
    observed_at         TEXT NOT NULL,
    effective_at_start  TEXT,
    effective_at_end    TEXT,
    recorded_at         TEXT NOT NULL,
    type                TEXT NOT NULL,
    data                TEXT NOT NULL CHECK(json_valid(data)),
    extraction_run_id   TEXT,
    source_turn_id      TEXT,
    source_role         TEXT NOT NULL,
    confidence          REAL,
    reason              TEXT,
    time_confidence     TEXT NOT NULL CHECK(time_confidence IN ('exact', 'inferred', 'unknown')),
    caused_by           TEXT,
    schema_version      INTEGER NOT NULL
);

CREATE INDEX idx_events_recorded_at ON events(recorded_at);
CREATE INDEX idx_events_effective_start ON events(effective_at_start);
CREATE INDEX idx_events_effective_end ON events(effective_at_end);
CREATE INDEX idx_events_run_id ON events(extraction_run_id);

CREATE TABLE event_entities (
    event_id     TEXT NOT NULL,
    entity_id    TEXT NOT NULL,
    role         TEXT NOT NULL,
    PRIMARY KEY(event_id, entity_id, role)
);

CREATE INDEX idx_event_entities_entity ON event_entities(entity_id);

CREATE TABLE dirty_ranges (
    id                 TEXT PRIMARY KEY,
    owner_id           TEXT NOT NULL,
    from_recorded_at   TEXT,
    from_effective_at  TEXT,
    reason             TEXT NOT NULL,
    created_at         TEXT NOT NULL
);

CREATE TABLE snapshots (
    id                         TEXT PRIMARY KEY,
    basis                      TEXT NOT NULL CHECK(basis IN ('known')),
    created_at                 TEXT NOT NULL,
    last_seq                   INTEGER NOT NULL,
    projection_version         INTEGER NOT NULL,
    max_recorded_at_included   TEXT NOT NULL,
    max_effective_at_included  TEXT,
    state_blob                 BLOB NOT NULL,
    relation_blob              BLOB NOT NULL
);
```

### 6.2 Projection 테이블

```sql
CREATE VIRTUAL TABLE entity_fts USING fts5(
    entity_id UNINDEXED,
    text,
    tokenize='unicode61'
);

CREATE TABLE vec_events (
    event_id           TEXT NOT NULL,
    embedder_version   TEXT NOT NULL,
    dim                INTEGER NOT NULL,
    embedding          BLOB NOT NULL,
    indexed_at         TEXT NOT NULL,
    PRIMARY KEY(event_id, embedder_version)
);
```

주의:
- `entity_fts`, `vec_events`는 canonical truth가 아니다.
- `vec_events`는 embedder version별로 다시 만들 수 있어야 한다.

### 6.3 Raw Log 저장 구조

```text
raw/
  manifest.json
  active-000001.jsonl
  archived/
    000000.jsonl.gz
    000001.jsonl.gz
```

`manifest.json`은 최소한 아래를 가진다.

```json
{
  "active_segment": "active-000001.jsonl",
  "last_committed_turn_id": "turn_123",
  "last_rotation_at": "2026-04-15T09:00:00Z"
}
```

### 6.4 PRAGMA 기본값

```sql
PRAGMA journal_mode = WAL;
PRAGMA synchronous = FULL;
PRAGMA foreign_keys = ON;
PRAGMA busy_timeout = 5000;
```

v2.3에서는 raw durability와 canonical store 정합성을 우선하기 때문에 `FULL`을 기본으로 둔다.

---

## 7. Public API

### 7.1 상태 표기 원칙

- `Implemented (Phase 1)`: 현재 저장소 코드에 이미 있음
- `Planned (Phase 2+)`: 문서에만 있고 아직 구현되지 않음

### 7.2 `Engram` 클래스

Status: `Implemented (Phase 2)`

```python
class Engram:
    def __init__(
        self,
        user_id: str = "default",
        path: str | None = None,
        session_id: str | None = None,
        queue_max_size: int = 10000,
        queue_put_timeout: float = 1.0,
        extractor: Extractor | None = None,
        embedder: Embedder | None = None,
    ): ...
```

현재 구현 메모:
- extractor를 넘기지 않으면 기본 `NullExtractor`가 사용된다.
- 즉 `flush("canonical")`은 동작하지만, 기본값으로는 extraction run만 기록하고 event는 만들지 않는다.
- embedder를 넘기지 않으면 기본 `HashEmbedder`가 사용된다.
- embedder는 `version`, `dim`, `embed_texts(texts)`를 가진 pluggable 인터페이스다.

### 7.3 Write API

Status: `Implemented (Phase 1)`

```python
def turn(
    self,
    user: str,
    assistant: str,
    *,
    observed_at: datetime | None = None,
    session_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> TurnAck:
    ...

def append(
    self,
    event_type: str,
    data: dict[str, Any],
    *,
    observed_at: datetime | None = None,
    effective_at_start: datetime | None = None,
    effective_at_end: datetime | None = None,
    source_role: Literal["user", "assistant", "tool", "system", "manual"] = "manual",
    source_turn_id: str | None = None,
    caused_by: str | None = None,
    confidence: float | None = None,
    reason: str | None = None,
    time_confidence: Literal["exact", "inferred", "unknown"] = "unknown",
) -> str:
    ...
```

규칙:
- `turn()`은 L0 raw durability까지만 즉시 보장한다.
- queue enqueue가 실패해도 raw append가 성공했다면 `turn()`은 예외 대신 `TurnAck(..., queued=False)`를 반환한다.
- `append()`는 L1 canonical commit을 동기 수행한다.
- 현재 런타임에서 `append()`가 허용하는 structured 이벤트는 `entity.create`, `entity.update`, `entity.delete`, `relation.create`, `relation.update`, `relation.delete`다.
- `append()`는 `caused_by`가 주어지면 해당 event id가 이미 canonical store에 존재할 때만 허용한다.
- relation 전용 public read API는 아직 없지만, relation 이벤트는 canonical replay, `search()`, `context()`, projector rebuild에 반영된다.
- `relation.update`는 prior `relation.create`가 없어도 active relation을 만든 것으로 해석한다.
- `valid` + `time_window=(start, end)`에서 relation은 `end` 시점 active가 아니라 `[start, end)` 구간 중 한 번이라도 active였으면 후보로 포함된다.

### 7.4 Flush API

Status: `Implemented (Phase 3 PR2 for index, Phase 2 PR2 for canonical, Phase 2 PR1 for projection)`

```python
def flush(
    self,
    level: Literal["raw", "canonical", "projection", "index"] = "projection",
) -> None:
    ...
```

현재 구현 규칙:
- `flush("raw")`는 즉시 반환한다. `turn()`이 raw append를 이미 동기 완료하기 때문이다.
- `flush("canonical")`은 현재 queue를 끝까지 소비하면서 configured extractor를 실행한다.
- extractor가 event를 만들면 `extraction_runs`, `events`, `event_entities`, `dirty_ranges`가 함께 커밋된다.
- 기본 extractor는 no-op이므로 `flush("canonical")`을 호출해도 event 없이 successful extraction run만 남는다.
- `flush("projection")`은 `dirty_ranges`를 읽어 projector rebuild를 완료할 때까지 수행한다.
- `flush("index")`는 현재 embedder version 기준으로 아직 인덱싱되지 않은 canonical event를 `vec_events`에 backfill한다.
- embedder version이 바뀌면 이전 vector row는 남겨두고, 검색은 현재 version row만 사용한다.
- `turn()`만 호출한 raw turn은 `flush("projection")`으로도 canonical/projection에 올라가지 않는다.

### 7.5 Query API

Status:
- `get()`, `get_known_at()`, `known_history()` = `Implemented (Phase 1)`
- `get_valid_at()`, `valid_history()` = `Implemented (Phase 4 PR1)`

```python
def get(self, entity_id: str) -> Entity | None:
    """현재 known-state projection."""

def get_known_at(self, entity_id: str, at: datetime) -> TemporalEntityView | None:
    """그 시점까지 실제로 커밋되어 알고 있던 상태."""

def get_valid_at(self, entity_id: str, at: datetime) -> TemporalEntityView | None:
    """현재 active canonical events를 기준으로 그 시점에 유효했던 상태."""

def known_history(self, entity_id: str, attr: str | None = None) -> list[HistoryEntry]:
    ...

def valid_history(self, entity_id: str, attr: str | None = None) -> list[HistoryEntry]:
    ...
```

제거된 API:
- `get_at()`는 v2.3에서 삭제한다. 의미가 둘로 갈라지기 때문이다.

### 7.6 Retrieval API

Status:
- `search(..., time_mode="known")` = `Implemented (Phase 3 PR1)`
- `context(..., time_mode="known")` = `Implemented (Phase 3 PR1)`
- `search(..., time_mode="valid")` = `Implemented (Phase 4 PR2)`
- `context(..., time_mode="valid")` = `Implemented (Phase 4 PR2)`
- semantic ranking = `Implemented (Phase 3 PR2)`
- causal ranking = `Implemented (Phase 6 PR1)`

```python
def search(
    self,
    query: str,
    *,
    time_mode: Literal["known", "valid"] = "known",
    time_window: tuple[datetime, datetime] | None = None,
    k: int = 20,
) -> list[SearchResult]:
    ...

def context(
    self,
    query: str,
    *,
    time_mode: Literal["known", "valid"] = "known",
    time_window: tuple[datetime, datetime] | None = None,
    max_tokens: int = 2000,
    include_history: bool = True,
    include_raw: bool = False,
) -> str:
    ...
```

현재 구현 규칙:
- `search()`는 canonical 이벤트를 직접 읽는 event-seeded retrieval이다.
- lexical 점수와 semantic cosine 점수를 함께 사용한다.
- explicit causal 1-hop expansion을 함께 사용한다.
- 지원 축은 현재 `entity`, optional `semantic`, optional `temporal`, optional `causal`다.
- `time_mode="known"`는 `recorded_at` 기준 lexical retrieval이다.
- `time_mode="valid"`는 `effective_at_*` 기준 lexical retrieval이다.
- `time_mode="valid"`에서는 `effective_at_start`가 없는 이벤트를 seed에서 제외한다.
- semantic 검색은 현재 embedder version의 `vec_events`만 사용한다.
- semantic row가 없는 경우에도 검색은 lexical-only로 정상 동작한다.
- `context()`는 `search()` 결과를 바탕으로 `Memory Basis / Current State / Relevant Changes / Raw Evidence` 섹션을 만든다.
- relation event가 supporting event에 포함되면 source/target entity를 현재 상태 후보로 투영하고, active relation summary를 `Current State`에 함께 적는다.
- relation event는 query의 mode/time 기준에서 양 endpoint entity가 모두 active일 때만 active relation seed로 취급한다.
- `context(time_mode="valid", time_window=...)`의 relation summary는 `relations_active_in_window=...` 형식으로 표기하고, 구간 중 활성이라는 의미를 드러낸다.
- 같은 경우 entity attrs는 `attrs_as_of_window_end=...` / `unknown_attrs_as_of_window_end=...`로 표기한다. 즉 현재 구현은 relation은 window semantics, entity attrs는 `end_at` semantics를 쓰되, 출력 라벨에서 그 차이를 드러낸다.
- causal supporting event가 있으면 `Memory Basis`에 그 사실을 표시하고, `Relevant Changes`에는 `caused by:` / `led to:` 설명을 함께 렌더링한다.
- `include_raw=True`일 때만 `source_turn_id`를 따라 raw evidence를 붙인다.
- `context(time_mode="valid")`는 `Current State`에 `unknown_attrs`를 함께 노출한다.
- `Relevant Changes`는 entity attr 변경뿐 아니라 relation create/update/delete도 문장으로 렌더링한다.

### 7.7 Maintenance API

Status:
- `rebuild_projection()` = `Planned (Phase 2)`
- `reprocess()` = `Implemented (Phase 5 PR1)`

```python
def reprocess(
    self,
    *,
    from_turn_id: str | None = None,
    to_turn_id: str | None = None,
    extractor_version: str | None = None,
) -> int:
    ...

def rebuild_projection(
    self,
    *,
    owner_id: str | None = None,
    from_recorded_at: datetime | None = None,
    from_effective_at: datetime | None = None,
) -> None:
    ...
```

현재 구현 규칙:
- `reprocess()`는 raw append 순서 기준 inclusive range를 다시 extractor에 통과시킨다.
- `extractor_version` 인자를 주면 현재 연결된 extractor version과 같을 때만 허용한다.
- 새 successful run이 생기면 같은 `source_turn_id`의 이전 active successful run을 supersede한다.
- supersede는 `extraction_runs.superseded_at`와 `superseded_runs` 둘 다에 기록된다.
- reprocess는 canonical store와 lineage만 갱신하고, projection 반영은 `flush("projection")`이 맡는다.

### 7.8 Raw API

Status: `Implemented (Phase 1)`

```python
def raw_recent(self, limit: int = 20) -> list[RawTurn]:
    ...

def raw_get(self, turn_id: str) -> RawTurn | None:
    ...
```

---

## 8. 타입 시스템

### 8.1 시간 타입

모든 datetime은 다음 규칙을 따른다.

- timezone-aware UTC
- RFC3339 직렬화
- naive datetime 입력은 금지

### 8.2 SearchResult

```python
@dataclass
class SearchResult:
    entity_id: str
    score: float
    matched_axes: set[Literal["entity", "semantic", "temporal", "causal"]]
    supporting_event_ids: list[str]
    time_basis: Literal["known", "valid"]
```

### 8.3 HistoryEntry

```python
@dataclass
class HistoryEntry:
    entity_id: str
    attr: str
    old_value: Any
    new_value: Any
    observed_at: datetime
    effective_at_start: datetime | None
    effective_at_end: datetime | None
    recorded_at: datetime
    reason: str | None
    confidence: float | None
    basis: Literal["known", "valid"]
    event_id: str
```

---

## 9. Write Path: 추출 파이프라인

### 9.1 단계 개요

```text
turn()
  -> raw log append
  -> queue enqueue
  -> worker consumes
  -> filter
  -> extractor
  -> canonical tx commit
  -> dirty range mark
  -> projector rebuild
  -> index refresh
```

### 9.2 QueueItem

```python
@dataclass
class QueueItem:
    turn_id: str
    observed_at: datetime
    session_id: str | None
    user: str
    assistant: str
    metadata: dict[str, Any]
```

### 9.3 Extractor 계약

추출기의 역할은 두 가지를 분리해서 내놓는 것이다.

1. 지금 말한 사실을 구조화한다.
2. 실제 적용 시간에 대해 자신 있는 수준만 표시한다.

```python
{
  "events": [
    {
      "type": "entity.update",
      "data": {"id": "user:alice", "attrs": {"location": "Busan"}},
      "confidence": 0.92,
      "reason": "사용자가 본인의 현재 거주지를 명시함",
      "effective_at_start": "2026-04-29T00:00:00Z",
      "effective_at_end": null,
      "time_confidence": "inferred"
    }
  ]
}
```

중요한 규칙:
- `observed_at`는 무조건 `RawTurn.observed_at`
- `effective_at_*`는 추출기가 진짜로 추정할 수 있을 때만 채운다
- 모르면 비워 둔다
- assistant 발화는 기본적으로 낮은 신뢰 소스로 취급한다

### 9.4 Canonical Commit 계약

DB 트랜잭션 안에서는 아래만 수행한다.

1. `ExtractionRun` 기록
2. `Event` 기록
3. `event_entities` 기록
4. `dirty_ranges` 기록
5. 필요 시 `superseded_runs` 기록

트랜잭션 안에서 하지 않는 것:
- `state_cache` mutate
- `relation cache` mutate
- `entity_fts` 갱신
- `vec_events` 갱신

### 9.5 Reprocess 계약

reprocess는 "예전 run을 지우고 새 run으로 갈아끼우는 것"이 아니라 아래 순서다.

1. 새 run 생성
2. 새 event set를 canonical store에 기록
3. old/new run lineage 기록
4. 영향받은 owner에 dirty range 생성
5. projector가 rebuild 수행

즉 run lineage와 projection 반영은 분리된다.

중요:
- 새 run이 생겨도 과거 `known-time`이 곧바로 덮어써지지 않는다.
- `known-time`은 질의 시각 기준으로 그때 보였던 run만 사용한다.
- `valid-time`은 현재 active run 기준으로 재구성한다.

### 9.6 Startup Catch-up

시작 시 반드시 아래를 수행한다.

1. raw manifest 확인
2. 마지막 raw turn과 마지막 `현재 extractor version`의 successful extraction run 비교
3. gap이 있으면 queue에 다시 넣음
4. pending dirty range가 있으면 projector 재개

현재 구현된 최소형:
- startup catch-up은 `현재 extractor version`의 active successful extraction run 존재 여부를 gap 판단 기준으로 사용한다.
- raw에만 있고 아직 `현재 extractor version`의 successful extraction run이 없는 turn은 startup 시 `QueueItem`으로 다시 enqueue한다.
- pending `dirty_ranges`가 있으면 `Projector.rebuild_dirty()`를 먼저 실행해 snapshot을 복구한다.

---

## 10. Read Path: Retrieval 파이프라인

### 10.1 event-seeded retrieval

v2.3의 retrieval은 항상 이벤트에서 시작한다.

```text
query
  -> entity matcher
  -> semantic event search
  -> time filter
  -> causal expansion
  -> seed event set
  -> event_entities로 entity 투영
  -> rank
  -> context build
```

현재 구현된 최소형:
- query token과 event `type/data/reason/source_role`의 lexical match로 seed event를 만든다.
- `flush("index")`로 구축한 `vec_events`를 사용해 current embedder version 기준 semantic cosine score를 계산한다.
- explicit `caused_by` 링크를 따라 상류 원인과 하류 결과를 1-hop causal expansion 한다.
- 한국어 query는 조사/어미가 붙은 일부 어절에 대해 간단한 suffix normalization을 적용한다.
- entity event와 relation event를 모두 seed로 사용하고, `event_entities`를 통해 relation의 source/target까지 entity 후보에 투영한다.
- 결과는 entity별 supporting event 집합으로 반환한다.
- `known` mode의 `time_window`는 `recorded_at` 기준으로만 적용된다.
- `valid` mode의 `time_window`는 `effective_at_*` overlap 기준으로 적용된다.
- relation event는 `valid` point query에서는 해당 시점에 active일 때만 seed가 되고, `valid` window query에서는 구간 중 한 번이라도 active였으면 seed가 된다.
- `valid` mode는 `effective_at_start`가 없는 이벤트를 검색 seed에서 제외하고, 불확실성은 `context()`의 `unknown_attrs`에서 보완한다.
- semantic row가 없는 이벤트는 lexical-only 후보로 남는다.
- relation valid-window 검색은 correctness-first 구현이다. 현재는 visible event를 순회하면서 relation key/endpoint overlap을 Python helper로 계산하므로, relation 수가 많거나 window가 넓은 경우 이후 최적화 대상이 될 수 있다.
- causal expansion은 현재 mode의 visible event 집합 안에서만 수행되고, same-batch alias 해석이나 heuristic causal 추론은 하지 않는다.

### 10.2 Run visibility 필터

조회는 모두 run visibility를 타지만, mode에 따라 기준이 다르다.

- `known` mode:
  - `processed_at <= query_time`
  - `superseded_at is null or superseded_at > query_time`
  - failed/skipped run 제외
- `valid` mode:
  - 현재 active run만 사용
  - failed/skipped run 제외

즉 `vec_events`도 단독 조회하지 않고, 먼저 mode에 맞는 visible event 집합을 만든 뒤 그 event id에 대해서만 semantic score를 계산한다.

### 10.3 Time Mode

#### known mode
- 시간 필터 기준: `recorded_at`
- run visibility 기준: query 시점에 보였던 run
- 질문 해석: "그때 시스템이 뭘 알고 있었나?"

#### valid mode
- 시간 필터 기준: `effective_at_start/end`
- run visibility 기준: 현재 active run
- 질문 해석: "지금 기준으로 보면 그때 무엇이 유효했나?"

### 10.4 Ranking

스코어는 단일 축이 아니라 합성이다.

```text
event_score = lexical_score only
          or 0.6 * lexical_score + 0.4 * semantic_score
causal_score = seed_event_score * 0.5
entity_score = Σ supporting_event_scores
```

정규화 원칙:
- 현재 semantic은 cosine similarity를 0~1 범위로 clamp하여 사용한다
- current embedder version의 semantic row가 하나도 없으면 lexical-only 결과를 그대로 사용한다
- causal은 direct lexical/semantic seed가 있는 이벤트 주변만 1-hop 확장하고, causal-only 이벤트는 `causal_score`로만 점수를 받는다
- 이미 direct lexical/semantic seed로 들어온 이벤트는 causal 때문에 추가 가중치를 받지 않는다
- `matched_axes`를 결과에 남겨 디버깅 가능하게 한다

### 10.5 event_entities의 역할

이 테이블이 있어야 다음이 싸고 정확해진다.

- 특정 entity 관련 history
- relation과 entity 연결
- query에서 찾은 event를 entity 후보로 확장
- dirty range별 rebuild 범위 계산

---

## 11. Context Builder

### 11.1 출력 원칙

context는 단순 텍스트 덤프가 아니라, 모델이 오해하지 않게 정리된 설명이어야 한다.

- 현재 상태
- 관련 관계
- 관련 변경 이력
- 선택적 raw evidence
- 각 항목의 시간 기준과 확신도

### 11.2 섹션 예시

```text
## Memory Basis
- mode: known
- as_of: 2026-04-15T10:00:00Z

## Current State
- user:alice
  attrs: {"diet": "vegetarian", "location": "Busan"}
  relations: ["manager -> user:bob"]

## Relevant Changes
- location changed to Busan
  observed_at: 2026-05-01
  effective_at_start: 2026-04-29
  confidence: 0.92
  reason: user stated they had moved last week
- relation user:alice -[manager]-> user:bob attrs={"scope": "engram"}
  effective_at: 2026-05-01

## Raw Evidence
- [2026-05-01] "지난주에 부산으로 이사했어"
```

### 11.3 안전 장치

- raw/user text는 quoting 또는 escaping 후 넣는다
- confidence가 낮거나 effective time이 unknown이면 명시한다
- truncate는 문장/항목 단위로 자른다

---

## 12. 이벤트 소싱 + 스냅샷

### 12.1 EventStore의 기준 정렬

canonical store의 기본 정렬은 `seq`다.
하지만 의미가 다른 두 가지 질의가 있다.

- known replay: `recorded_at`, tie-breaker는 `seq`
- valid replay: `effective_at_start`, tie-breaker는 `recorded_at`, `seq`

즉 "seq 하나로 모든 시간 의미를 해결한다"는 가정을 버린다.

### 12.2 Known Snapshot

`known` snapshot은 아래를 anchor로 가진다.

- `last_seq`
- `max_recorded_at_included`
- `projection_version`

복원 알고리즘:

```python
def restore_known_to(at: datetime) -> StateCache:
    snap = latest_known_snapshot_before(at)
    state = load_snapshot(snap) if snap else empty_state()
    for event in visible_known_events_after_seq(snap.last_seq if snap else 0, at):
        apply_known(state, event)
    return state
```

여기서 중요한 점:
- `break`하지 않는다
- backdated effective time이나 늦게 커밋된 이벤트가 있어도 recorded order replay는 계속된다
- 나중에 supersede된 run이라도 `at` 시점에 visible이었다면 replay에 포함된다

### 12.3 Valid Replay

`valid` 질의는 v2.3에서 전역 스냅샷보다 정확성을 우선한다.

- 기본 구현은 entity-local replay다
- candidate event를 `event_entities`로 좁힌 뒤
- `effective_at_*`가 질의 시점을 덮는 이벤트만 적용한다
- `effective time unknown`인 이벤트는 `unknown_attrs`로 밀어낸다

현재 구현된 최소형:
- `get_valid_at()`는 entity-local replay만 구현한다.
- 정렬 기준은 `effective_at_start`, 그다음 `recorded_at`, `seq`다.
- `effective_at_start`가 없는 이벤트는 valid apply에서 제외하고 `unknown_attrs`로 보낸다.
- `search()/context()`도 현재는 같은 valid-time 최소 규칙을 사용한다.

### 12.4 Projector

projector는 dirty range를 읽고 L2를 rebuild한다.

```text
dirty_ranges
  -> affected owners 조회
  -> canonical events 재생성
  -> 새 immutable state snapshot 생성
  -> relation cache 생성
  -> fts / vec refresh
  -> swap
```

즉 projection은 "append할 때 바로 mutate"가 아니라 "새 스냅샷을 만든 뒤 교체"다.

---

## 13. Raw Log (Tier 1)

### 13.1 저장 전략

single gzip append는 버리고 segment 방식으로 간다.

- active segment는 plain JSONL
- rotate 시 gzip 압축
- 각 세그먼트는 checksum 보유
- manifest가 마지막 커밋 지점을 가리킴

### 13.2 append 규칙

```python
def append(turn: RawTurn) -> TurnAck:
    acquire_file_lock()
    write_jsonl_line(turn)
    fsync()
    update_manifest()
    fsync_manifest()
    return TurnAck(...)
```

### 13.3 recovery 규칙

시작 시:
- manifest와 실제 세그먼트 tail 비교
- 마지막 줄 checksum 검증
- 손상된 마지막 줄은 잘라내고 경고
- canonical gap이 있으면 catch-up enqueue

---

## 14. 히스토리와 시간 질의

### 14.1 known_history

`known_history`는 시스템이 실제로 알게 된 순서를 보여준다.

- 정렬 기준: `recorded_at`, `seq`
- run visibility 기준: `known-time visible run`
- 현재 시각에서 호출하면 이미 supersede된 run의 이벤트는 포함되지 않는다.
- backfill/reprocess 후에도 과거의 "지식 상태"는 바뀌지 않아야 한다

### 14.2 valid_history

`valid_history`는 지금의 active canonical events로 재구성한 유효 시간선을 보여준다.

- 정렬 기준: `effective_at_start`, `recorded_at`, `seq`
- reprocess 후 결과가 달라질 수 있다
- 이건 버그가 아니라 정의된 동작이다
- 현재 최소 구현은 `effective_at_start`가 없는 이벤트를 히스토리에서 제외한다

### 14.3 derived change 계산

`_changed_from`는 저장하지 않는다.
히스토리 렌더링 시 replay로 계산한다.

이렇게 해야:
- reprocess
- backfill
- manual correction

상황에서도 과거 비교가 현재 projection 오염 없이 다시 계산된다.

---

## 15. Actions 경계

v2.3 core 문서는 action writeback을 아래 수준까지만 다룬다.

- action 모듈이 남긴 감사 이벤트를 canonical store에 저장할 수 있다
- action 결과를 retrieval/context에 참고 정보로 노출할 수 있다

v2.3 core가 다루지 않는 것:
- approval workflow
- lease / heartbeat
- RUNNING stuck recovery
- external idempotency propagation
- reconcile job
- auth/authz
- connector-specific revert

즉 action은 "핵심 메모리 불변식"이 아니라 "확장 모듈"이다.

---

## 16. 멀티테넌시와 보안 경계

### 16.1 사용자 분리

- `user_id`는 외부 입력을 그대로 파일 경로에 쓰지 않는다
- 내부적으로 path-safe slug 또는 UUID로 매핑한다
- 한 사용자당 하나의 storage root를 가진다

### 16.2 권한 경계

raw API는 민감할 수 있으므로 서버 환경에서는 별도 권한이 필요하다.

- `context()`는 앱에서 주로 노출
- `raw_recent()`와 `raw_get()`는 운영/디버깅용
- export/delete는 관리자 또는 사용자 본인만 허용

### 16.3 삭제와 보존

v2.3 core는 최소한 다음 정책 지점을 문서에 남긴다.

- raw retention
- snapshot retention
- vector reindex after delete
- right-to-delete 시 canonical / projection / raw 동시 정리

---

## 17. 동시성 및 라이프사이클

### 17.1 writer / reader 모델

- 동일 사용자에 대해 writer는 1개만 허용
- writer는 file lock을 잡는다
- lockfile에는 최소한 `pid`, `created_at`를 저장하고, dead PID면 stale lock으로 자동 회수한다
- reader는 thread-local read-only connection 사용
- same-user multi-process writer는 지원하지 않는다

### 17.2 shared state 모델

projection은 mutable dict를 여러 스레드가 함께 만지는 모델이 아니다.

선택 규칙:
- projector가 새 immutable snapshot을 만든다
- 교체 시 짧은 lock 또는 atomic pointer swap
- reader는 항상 일관된 한 버전을 본다

### 17.3 queue 정책

`block`은 무기한 대기가 아니다.

```python
queue.put(item, timeout=queue_put_timeout)
```

실패 시:
- `QueueFullError`
- metric 증가
- 앱이 fallback 판단 가능

### 17.4 shutdown 규칙

1. 새 enqueue 차단
2. queue drain
3. canonical flush
4. projector finish 또는 timeout
5. raw/file/db lock 해제

---

## 18. 성능 목표와 측정 원칙

v2.3는 숫자를 약속하기보다 "정합성을 유지한 상태에서 측정"을 우선한다.

### 18.1 correctness-first 목표

| 작업 | 목표 |
|---|---|
| `turn()` raw durability | p99 20ms 이내 |
| `flush(level="canonical")` | 추출기 제외 오버헤드 명확화 |
| `context()` | 10k entity에서 p99 100ms 이내 목표 |
| startup catch-up | gap 크기에 선형 증가하되 자동 복구 |

### 18.2 최적화는 나중에

이 문서에서 숫자를 공격적으로 잡지 않는 이유:
- segmented raw log
- projector rebuild
- active-run join
- valid-time replay

가 모두 correctness를 위해 추가된 비용이기 때문이다.

---

## 19. 파일 구조

```text
engram/
  src/engram/
    engram.py
    types.py
    errors.py
    storage/
      schema.sql
      store.py
      raw_log.py
      snapshot.py
      projector.py
      locks.py
    extraction/
      filter.py
      llm_extractor.py
      pipeline.py
      worker.py
      recovery.py
    retrieval/
      pipeline.py
      entity_matcher.py
      semantic_index.py
      entity_index.py
      temporal.py
      causal.py
      ranker.py
    context/
      builder.py
    security/
      paths.py
      authz.py
  docs/
    engram_actions.md
```

핵심 메시지:
- `projector.py`와 `recovery.py`는 더 이상 부가 기능이 아니다
- `engram_actions.md`는 core 밖 companion spec이다

---

## 20. 구현 순서

### Phase 0. 의미 고정

- 시간 필드와 API 확정
- event payload를 `attrs` 중첩으로 통일
- writer/reader/file lock 계약 고정

### Phase 1. Canonical Foundation

- raw segment log
- extraction_runs / events / event_entities / dirty_ranges
- `append()`, `turn()`, `get_known_at()` 정확성

### Phase 2. Recovery and Projection

- startup catch-up
- projector rebuild
- immutable snapshot swap
- `flush(level=...)`

### Phase 3. Retrieval and Context

- event-seeded retrieval
- active-run-safe semantic index
- `context(time_mode=...)`

### Phase 4. Valid-time

- `get_valid_at()`
- `valid_history()`
- effective-time uncertainty handling

### Phase 5. Optional Actions

- companion spec 기반 별도 구현

---

## 21. 테스트 전략

### 21.1 꼭 필요한 테스트

1. 사용자가 5월 1일에 "지난주에 이사했다"고 말하면
   - `observed_at = 5월 1일`
   - `effective_at_start = 4월 말` 또는 `None`
   - `time_confidence`가 적절히 남는지
2. `get_known_at()`은 reprocess 후에도 과거 결과가 바뀌지 않는지
3. `get_valid_at()`은 최신 active run 기준으로 다시 계산되는지
4. backdated append 뒤에도 `restore_known_to()`가 `break` 없이 정확한지
5. superseded run event가 vec/fts/history/context에 새어 나오지 않는지
6. tx 실패 후 projection이 DB와 갈라지지 않는지
7. startup catch-up이 raw gap을 자동 복구하는지
8. same-user 두 writer가 동시에 열리면 하나가 lock에서 실패하는지
9. `flush(level="canonical")`과 `flush(level="projection")`의 의미가 테스트로 고정되는지

### 21.2 property test 불변식

- `seq`는 유일하며 단조 증가
- `known_history(now)`의 마지막 상태는 `get()`와 동일
- same `(source_turn_id, extractor_version)` 성공 run은 하나
- superseded run은 active query에 보이지 않음
- `event_entities` 없이도 정답이 바뀌지 않아야 하나, 있으면 더 싸야 함

---

## 22. 벤치마크 계획

### 22.1 품질 지표

정확도는 F1만으로 보지 않는다.

- false memory rate
- stale memory rate
- contradiction rate
- known/valid confusion rate
- reprocess determinism
- startup catch-up latency

### 22.2 레이턴시 지표

- raw append latency
- canonical commit latency
- projection rebuild latency
- context latency
- valid-time replay latency

### 22.3 recovery 지표

- crash 후 catch-up 완료 시간
- dirty range rebuild 시간
- vector/fts 재색인 시간

---

## 23. 알려진 한계

### 23.1 v2.3에서 의도적으로 남기는 한계

- valid-time은 이벤트가 시간을 모르면 완전하지 않다
- `get_valid_at()`은 `unknown_attrs`를 반환할 수 있다
- 분산 multi-writer는 지원하지 않는다
- action writeback 복구는 core 문서 범위 밖이다
- 한국어 FTS 품질은 tokenizer 선택에 따라 차이가 클 수 있다

### 23.2 정직한 제품 메시지

Engram의 강점:
- 원본 대화 보존
- structured event lineage
- rebuildable projection
- known-time / valid-time 분리

Engram의 약점:
- 구현이 단순 key-value memory보다 복잡하다
- valid-time은 추출 품질에 영향을 받는다
- action writeback까지 한 번에 해결하는 제품은 아니다

---

## 24. Appendix: 주요 코드 스켈레톤

### 24.1 Engram 조립

```python
class Engram:
    def __init__(self, user_id: str = "default", path: str | None = None, ...):
        base = resolve_storage_root(user_id, path)

        self.lock = acquire_writer_lock(base)
        self.writer = open_writer_connection(base / "engram.db")
        self.reader_factory = ReaderFactory(base / "engram.db")

        self.raw_log = SegmentedRawLog(base / "raw")
        self.store = EventStore(self.writer)
        self.projector = Projector(self.writer, self.reader_factory)
        self.recovery = RecoveryService(self.raw_log, self.store, self.projector)
        self.extractor = NullExtractor()
        self.canonical_worker = CanonicalWorker(self.store, self.extractor)

        self.queue = WriteQueue(timeout=1.0)
        self.recovery.catch_up_on_startup()

    def turn(self, user: str, assistant: str, **kwargs) -> TurnAck:
        turn = RawTurn(
            id=make_turn_id(),
            observed_at=kwargs.get("observed_at") or utcnow(),
            session_id=kwargs.get("session_id"),
            user=user,
            assistant=assistant,
            metadata=kwargs.get("metadata") or {},
        )
        ack = self.raw_log.append(turn)
        self.queue.put(QueueItem.from_turn(turn))
        return ack

    def append(self, event_type: str, data: dict, **kwargs) -> str:
        event = Event(
            id=make_event_id(),
            seq=self.store.next_seq(),
            observed_at=kwargs.get("observed_at") or utcnow(),
            effective_at_start=kwargs.get("effective_at_start"),
            effective_at_end=kwargs.get("effective_at_end"),
            recorded_at=utcnow(),
            type=event_type,
            data=data,
            extraction_run_id=None,
            source_turn_id=kwargs.get("source_turn_id"),
            source_role=kwargs.get("source_role", "manual"),
            confidence=kwargs.get("confidence"),
            reason=kwargs.get("reason"),
            time_confidence=kwargs.get("time_confidence", "unknown"),
            caused_by=None,
            schema_version=1,
        )
        with self.store.transaction() as tx:
            self.store.append_event(tx, event)
            self.store.append_event_entities(tx, derive_event_entities(event))
            self.store.mark_dirty(tx, derive_dirty_ranges(event))
        self.projector.schedule_rebuild_for_event(event)
        return event.id
```

### 24.2 known-time / valid-time 예시

```python
view = mem.get_known_at("user:alice", dt("2026-05-01T10:00:00Z"))

view = mem.get_valid_at("user:alice", dt("2026-04-29T12:00:00Z"))
if view and view.unknown_attrs:
    print("이 시점의 일부 속성은 실제 적용 시간이 불확실함")
```

### 24.3 Projector 예시

```python
class Projector:
    def rebuild_owner(self, owner_id: str) -> None:
        events = self.store.active_events_for_owner(owner_id)
        new_state = replay_known_current(events)
        new_relations = replay_relations(events)
        new_fts_docs = build_fts_docs(new_state)
        new_vec_rows = build_event_embeddings(events)
        self.swap(owner_id, new_state, new_relations, new_fts_docs, new_vec_rows)
```

### 24.4 이 문서의 최종 의미

v2.3의 목적은 "AI 메모리를 더 똑똑하게 보이게 하는 것"이 아니라, 나중에 시스템이 커져도 **무엇이 원본이고, 무엇이 재구성이고, 무엇이 아직 모르는 사실인지**를 헷갈리지 않게 만드는 것이다.

그래야 재처리, 시간 질의, 복구, 검색, 감사가 한 방향으로 정렬된다.
