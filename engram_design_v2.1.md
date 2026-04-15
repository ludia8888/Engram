# Engram 개발 설계 문서 v2.2

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
14. [충돌 감지 및 히스토리](#14-충돌-감지-및-히스토리)
15. [액션 Writeback](#15-액션-writeback)
16. [멀티테넌시](#16-멀티테넌시)
17. [동시성 및 라이프사이클](#17-동시성-및-라이프사이클)
18. [성능 목표](#18-성능-목표)
19. [파일 구조](#19-파일-구조)
20. [구현 순서](#20-구현-순서)
21. [테스트 전략](#21-테스트-전략)
22. [벤치마크 계획](#22-벤치마크-계획)
23. [알려진 한계](#23-알려진-한계)
24. [Appendix: 주요 코드 스켈레톤](#24-appendix-주요-코드-스켈레톤)

---

## 1. Executive Summary

### 1.1 한 줄 정의

```
Engram = mem0의 지능 + SQLite의 단순함 + Git의 시간여행
       = 임베디드로 돌아가는 event-sourced AI memory + action ledger
```

### 1.2 핵심 포지션

| | mem0 | Letta (MemGPT) | Engram |
|---|------|----------------|--------|
| 배포 모델 | Cloud-first | Server | **Embedded (SQLite)** |
| 저장 모델 | Vector + Graph | Tiered context | **Event-sourced + Vector + Graph** |
| 시간여행 | ✗ | ✗ | **✓ (get_at)** |
| 재추출 | 어려움 | ✗ | **✓ (reprocess)** |
| 외부 의존성 | Qdrant + Neo4j | Postgres | **없음 (sqlite + sqlite-vec)** |
| 원본 보존 | 부분적 | ✗ | **✓ (Tier 1 Raw Log)** |

### 1.3 핵심 가치

| 기존 LLM 메모리 | Engram |
|----------------|--------|
| 배치 압축 (손실) | 실시간 추출 + 원본 보존 |
| 텍스트 덩어리 | 이벤트 + 엔티티 + 그래프 + 인과 체인 |
| 쿼리 불가 | Entity + Semantic + Temporal + Causal 4축 |
| 세션 끊기면 손실 | Raw Log durability 보장 |
| 원본 손실 | 2-Tier (Raw + Structured) |
| 재추출 불가 | reprocess() first-class |
| 외부 DB 클러스터 필요 | SQLite 단일 파일 |

### 1.4 Engram의 역할과 한계

```
Engram = 구조화된 장기 기억 + 검색 + 시간여행 + run-aware typed action ledger

Engram ≠ 전체 대화 대체 (→ Raw Log로 보완)
Engram ≠ RAG 교체 (→ 문서 검색과 별개)
Engram ≠ 컨텍스트 윈도우 대체 (→ Hybrid 구성)
Engram ≠ 분산 스토리지 (→ 임베디드가 정체성)
Engram ≠ 범용 RPA 엔진 (→ 실제 writeback은 typed handler가 수행)
```

---

## 2. Quick Start

### 2.1 설치

```bash
pip install engram-db
```

의존성:
- Python 3.10+
- `sqlite3` (표준 라이브러리)
- `sqlite-vec` (벡터 인덱스용)
- `sentence-transformers` (로컬 임베딩, 옵션)
- `openai` 또는 `anthropic` (LLM 추출용, 옵션)

### 2.2 기본 사용

```python
from engram import Engram

mem = Engram(user_id="alice")

# 대화 기록
mem.turn(
    user="난 채식주의자야. 다음 주 도쿄 여행 가.",
    assistant="알겠습니다. 채식 가능한 도쿄 레스토랑 추천할까요?"
)

# 다음 대화에서 컨텍스트 주입
context = mem.context(query="저녁 뭐 먹을까?")
# → alice: vegetarian, traveling_to: Tokyo 등 relevance-ranked
```

### 2.3 Production 통합

```python
from engram import Engram

class ChatBot:
    def __init__(self, user_id: str):
        self.mem = Engram(user_id=user_id)
        self.history = []

    async def chat(self, user_input: str) -> str:
        # 1. Relevance-based 컨텍스트 생성
        memory_context = self.mem.context(
            query=user_input,
            max_tokens=2000,
        )

        # 2. LLM 호출
        messages = [
            {"role": "system", "content": memory_context},
            *self.history,
            {"role": "user", "content": user_input},
        ]
        response = await llm.generate(messages)

        # 3. 기록 (Fire-and-Forget, ~0.1ms)
        self.mem.turn(user_input, response)

        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": response})
        return response

    def close(self):
        self.mem.close()
```

---

## 3. 설계 철학

### 3.1 원칙

1. **임베디드 우선**: 외부 서버·DB·클러스터 없음. SQLite 단일 파일.
2. **이벤트 소싱**: 모든 상태는 이벤트에서 파생. 원본이 진실의 근원.
3. **2-Tier 저장**: Raw (손실 없음) + Structured (lossy projection).
4. **Fire-and-Forget 쓰기**: `turn()`은 큐에 넣고 즉시 리턴.
5. **4축 Retrieval**: Entity, Semantic, Temporal, Causal을 합성.
6. **Token budget 엄수**: `context()`는 지정된 토큰 이내로 패킹.
7. **재추출 가능**: `reprocess()`로 추출 로직 개선 반영.
8. **멀티테넌트 기본**: user_id 격리 설계 시점부터.
9. **타입 완결성**: 전 API mypy strict 통과.

### 3.2 비(非)원칙

- ❌ "Zero LLM cost"를 주장하지 않음. LLM 추출은 비용 발생.
- ❌ "완전 무손실"을 주장하지 않음. Tier 2는 lossy projection.
- ❌ "분산"을 지원하지 않음. 임베디드 제품임.
- ❌ Raw log를 RAG 대체로 포지셔닝하지 않음.

### 3.3 레이턴시 예산

| 작업 | 목표 | 허용 최대 |
|------|------|-----------|
| `turn()` (raw append + queue enqueue) | 1ms | 5ms |
| `context(query=...)` | 20ms | 50ms |
| `get(entity_id)` | 0.1ms | 1ms |
| `search(query)` | 30ms | 100ms |
| `recall(entity_id)` | 10ms | 50ms |
| 콜드 스타트 | 100ms | 500ms |
| 백그라운드 추출 (비용 은닉) | 1-2s | 5s |

---

## 4. 시스템 아키텍처

### 4.1 상위 구조

```
┌─────────────────────────────────────────────────────────────┐
│                     Application Layer                        │
│             (ChatBot, Agent, Assistant 등)                   │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ↓
┌─────────────────────────────────────────────────────────────┐
│                     Engram SDK                               │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                 Public API Surface                   │   │
│  │  turn() / context() / search() / recall() / get()   │   │
│  │  propose_action() / execute_action()                │   │
│  └────────────────┬────────────────────────────────────┘   │
│                   │                                         │
│      ┌────────────┬──────────────┬───────────────┐         │
│      ↓            ↓              ↓               │         │
│  (write path)  (read path)   (action path)      │         │
│  ┌───────────┐ ┌─────────────────────┐ ┌────────────────┐ │
│  │ Write     │ │ Retrieval Pipeline  │ │ Action Service │ │
│  │ Queue     │ │  event-seeded       │ │ intent / exec  │ │
│  │   ↓       │ │  retrieval          │ │   ↓            │ │
│  │ Worker    │ │   ↓                 │ │ PolicyEngine   │ │
│  │   ↓       │ │ Hybrid Ranker       │ │   ↓            │ │
│  │ Extractor │ │   ↓                 │ │ Handlers       │ │
│  │   ↓       │ │ Context Builder     │ │   ↓            │ │
│  │ Indexer   │ └──────────┬──────────┘ │ External I/O   │ │
│  └────┬──────┘            │            └──────┬─────────┘ │
│       │                   │                   │            │
│       ↓                   ↓                   ↓            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                  Storage Layer                       │   │
│  ├─────────────────────────────────────────────────────┤   │
│  │                                                     │   │
│  │  Tier 2 (SQLite):                                  │   │
│  │    • extraction_runs + events (이벤트 소싱)          │   │
│  │    • snapshots (상태 캐시)                          │   │
│  │    • action_types / intents / executions          │   │
│  │    • external_objects / bindings / effects        │   │
│  │    • vec_entities (sqlite-vec)                     │   │
│  │    • vec_events (sqlite-vec)                       │   │
│  │    • entity_idx (FTS5)                             │   │
│  │                                                     │   │
│  │  Tier 1 (File):                                    │   │
│  │    • raw.jsonl.gz (원본 대화, append-only)          │   │
│  │                                                     │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 2-Tier 저장소

```
┌───────────────────────────────────────────────┐
│          Tier 1: Raw Log (손실 없음)          │
├───────────────────────────────────────────────┤
│ • 모든 turn을 그대로 기록                       │
│ • 로컬 gzip jsonl, append-only                │
│ • `turn()` 응답 전 fsync로 durability 보장      │
│ • 선택적 S3 mirror는 백업용 (ACK 경로 아님)      │
│ • reprocess의 canonical source                │
│ • ~5MB/년/유저                                │
└───────────────────┬───────────────────────────┘
                    │
                    ↓ Extractor
                    ↓ (filter → extraction_run → events)
                    ↓
┌───────────────────────────────────────────────┐
│       Tier 2: Structured Memory (lossy)       │
├───────────────────────────────────────────────┤
│ • Extraction runs + Events (append-only)      │
│ • Snapshots (빠른 복원용)                      │
│ • Action ledger + external object graph       │
│ • Semantic index (벡터)                       │
│ • Entity index (FTS)                          │
│ • 쿼리/검색 최적화                             │
└───────────────────────────────────────────────┘
```

### 4.3 Write Path

```
user.turn(user_msg, assistant_msg)
    │
    ├─→ (sync) Local Raw Log append + fsync
    │          ↓
    │        RawTurn(turn_id, timestamp) ← durability 보장
    │
    ├─→ (async, optional) S3 mirror enqueue
    │
    └─→ (sync) Queue.put(turn_id, turn_timestamp, user, assistant)
              │
              ↓ (async, background worker)
              │
        EventStore.already_processed(turn_id, extractor_version)?
              │
         ┌────┴────┐
       yes       no
         │         │
      (skip)      ↓
                  │
        LocalFilter.classify()
              │
         ┌────┴────┐
       skip      extract
         │         │
      (drop)      ↓
                  │
        ExtractionRun.create(turn_id, extractor_version)
                  │
                  ↓
              LLMExtractor.extract(run_id, turn_id, occurred_at=turn_timestamp)
                  │
                  ↓ events[]
                  │
              ConflictDetector
              (add _changed_from)
                  │
                  ↓
              동시 업데이트:
                • extraction_runs append
                • EventStore.append()
                • superseded_runs 기록
                • StateCache.apply()
                • RelationIndex.update()
                • SemanticIndex.embed_and_store()
                • EntityIndex.update()
                • SnapshotManager.check()
```

### 4.4 Read Path

```
user.context(query="저녁 뭐 먹을까?")
    │
    ↓
RetrievalPipeline.run(query)
    │
    ├─→ EntityMatcher.match_events()      → [seed event ids]
    ├─→ SemanticIndex.search_events()     → [semantic event ids]
    ├─→ EventStore.events_in_range()      → [time-window event ids]
    └─→ CausalExpander.expand_events()    → [causal event ids]
    │
    ↓ seed event merge
    │
    ↓ seed event → affected entity projection
    │
HybridRanker.rank(candidates, weights)
    │
    ↓ scored candidates
    │
ContextBuilder.build(ranked, max_tokens)
    │
    ├─→ 현재 상태 (top 엔티티)
    ├─→ 관계 (1-hop)
    ├─→ 주요 변경 히스토리
    └─→ 최근 원본 (선택)
    │
    ↓
Final context string
```

### 4.5 Action Writeback Path

```
agent.propose_action("update_crm_field", payload)
    │
    ↓
ActionRegistry.lookup(action_type)
    │
    ↓
PolicyEngine.evaluate(intent)
    │
    ├─→ auto approval           → intent.status = APPROVED
    └─→ human approval required → intent.status = PROPOSED
                                   │
                              approve_action()
                                   │
                                   ↓
                             intent.status = APPROVED
                                   │
                                   ↓
                             execute_action()
                                   │
                        latest attempt exists?
                           │               │
                     RUNNING / SUCCEEDED    no or FAILED
                           │               │
                        return it      create execution attempt
                                            │
                                 ┌──────────┴──────────┐
                               success               fail
                                 │                    │
                     execution.status=SUCCEEDED   execution.status=FAILED
                                 │
                ├─→ external_objects upsert
                ├─→ external_bindings / action_effects 기록
                ├─→ EventStore.append(action.*)
                └─→ reversible이면 revert_action() 가능
```

---

## 5. 데이터 모델

### 5.1 이벤트 (Source of Truth)

```python
@dataclass
class Event:
    id: str                        # UUID
    occurred_at: datetime         # 사건이 실제로 일어난 시각 (canonical)
    recorded_at: datetime         # DB에 기록된 시각
    type: str                     # 아래 타입 참조
    data: dict                    # type별 스키마
    caused_by: str | None         # 인과 체인용 (upstream event id)
    extraction_run_id: str | None # LLM 추출 이벤트면 어떤 run에서 왔는지
```

중요한 규칙:
- `occurred_at`은 시간여행과 히스토리의 기준이 되는 **정규 시간**이다.
- LLM/하이브리드 추출 이벤트는 항상 `RawTurn.timestamp`를 `occurred_at`로 사용한다.
- `recorded_at`은 큐 지연, 재처리, 마이그레이션 등 실제 적재 시점을 보여주는 운영 메타데이터다.
- 수동 `append()` 이벤트는 기본적으로 `occurred_at=recorded_at=now`이며, 필요 시 `occurred_at`를 명시할 수 있다.

### 5.2 추출 실행 (Extraction Run)

```python
@dataclass
class ExtractionRun:
    id: str
    source_turn_id: str
    extractor_version: str
    occurred_at: datetime        # RawTurn.timestamp 복사
    processed_at: datetime       # 실제 추출 완료 시각
    status: Literal["SUCCEEDED", "FAILED", "SKIPPED"]
    error: str | None
    event_count: int
```

재처리 중복 방지는 이벤트 개별 행이 아니라 `ExtractionRun` 단위로 판단한다.
즉, 같은 `(source_turn_id, extractor_version)`에 대해 `SUCCEEDED` run이 이미 있으면 skip한다.

### 5.3 이벤트 타입

```python
# 스키마 진화
"schema.entity_type.add"
"schema.entity_type.attr.add"
"schema.relation_type.add"

# 엔티티 라이프사이클
"entity.create"               # data: {id, type, attrs}
"entity.update"               # data: {id, attrs, _changed_from?}
"entity.delete"               # data: {id}

# 관계
"relation.create"             # data: {source, target, type, attrs?}
"relation.update"
"relation.delete"

# 액션 writeback
"action.intent.create"        # data: {intent_id, action_type, payload, reason?, targets?, idempotency_key}
"action.intent.approve"       # data: {intent_id, approver, note?}
"action.intent.reject"        # data: {intent_id, approver, note?}
"action.intent.cancel"        # data: {intent_id, actor, note?}
"action.execution.start"      # data: {execution_id, intent_id}
"action.execution.succeed"    # data: {execution_id, result?, external_effects?}
"action.execution.fail"       # data: {execution_id, error}
"action.execution.revert"     # data: {execution_id, revert_execution_id, reason?}

# 외부 시스템 연결
"external.binding.upsert"     # data: {owner_kind, owner_id, external_object_id}
"action.effect.record"        # data: {execution_id, external_object_id, effect_type, payload?}
```

### 5.4 엔티티 (파생 상태)

```python
@dataclass
class Entity:
    id: str                    # 예: "user:alice", "project:arrakis"
    type: str                  # 예: "user", "project"
    attributes: dict[str, Any] # 현재 상태
    created_at: datetime
    updated_at: datetime       # 마지막 active event.occurred_at
```

엔티티는 이벤트 로그에서 파생됨. 직접 저장되지 않고 스냅샷 + 증분 replay로 계산.

### 5.5 관계

```python
@dataclass
class Relation:
    source: str                # entity id
    target: str                # entity id
    type: str                  # 예: "owns", "works_at"
    attributes: dict
    created_at: datetime
```

### 5.6 액션 타입

```python
@dataclass
class ActionType:
    name: str                  # 예: "create_ticket", "update_crm_field"
    system: str                # 예: "jira", "salesforce", "google_calendar"
    description: str
    input_schema: dict[str, Any]
    executor: str              # 등록된 handler 이름
    approval_mode: Literal["auto", "human_required", "policy"]
    reversible: bool
    compensator: str | None    # 되돌리기 handler 이름
```

### 5.7 액션 제안 (Intent)

```python
@dataclass
class ActionIntent:
    id: str
    action_type: str
    payload: dict[str, Any]
    proposed_at: datetime
    proposed_by: str           # "agent", "user", "system"
    reason: str | None
    target_entities: list[str]
    approval_mode: Literal["auto", "human_required", "policy"]
    status: Literal["PROPOSED", "APPROVED", "REJECTED", "CANCELED"]
    idempotency_key: str
    approved_by: str | None
    approved_at: datetime | None
    rejected_by: str | None
    rejected_at: datetime | None
    resolution_note: str | None
```

`ActionIntent`는 "무엇을 하려는지"에 대한 의사결정 로그다.
한 intent에는 여러 execution attempt가 매달릴 수 있다.

### 5.8 액션 실행 (Execution)

```python
@dataclass
class ActionExecution:
    id: str
    intent_id: str
    status: Literal["RUNNING", "SUCCEEDED", "FAILED", "REVERTED"]
    attempt: int
    executor: str
    started_at: datetime | None
    finished_at: datetime | None
    result: dict[str, Any] | None
    error: str | None
    reverted_from: str | None  # 이전 execution id
```

### 5.9 외부 객체

```python
@dataclass
class ExternalObject:
    id: str
    system: str                # 예: "salesforce", "jira"
    ref_type: str              # 예: "contact", "issue", "calendar_event"
    external_id: str
    attributes: dict[str, Any]
    updated_at: datetime
```

`ExternalObject`는 외부 시스템의 canonical 객체다.
같은 `(system, ref_type, external_id)`는 하나의 객체만 가진다.

### 5.10 외부 바인딩

```python
@dataclass
class ExternalBinding:
    id: str
    owner_kind: Literal["entity"]
    owner_id: str
    external_object_id: str
    role: str                  # 예: "primary_contact", "ticket_owner"
    bound_at: datetime
```

### 5.11 액션 효과 기록

```python
@dataclass
class ActionEffect:
    id: str
    execution_id: str
    external_object_id: str
    effect_type: str           # 예: "create", "update", "cancel", "comment"
    payload: dict[str, Any]
    recorded_at: datetime
```

이렇게 분리하면:
- 하나의 Salesforce contact를 `user:alice`의 canonical binding으로 유지할 수 있고
- 동시에 여러 action execution의 감사 로그에서도 같은 객체를 참조할 수 있다.

### 5.12 스냅샷

```python
@dataclass
class Snapshot:
    id: str
    timestamp: datetime
    event_id: str              # 이 이벤트까지 반영된 상태
    event_count: int           # 누적 이벤트 수
    state: dict[str, Entity]   # entity_id → Entity
    relations: dict            # 관계 인덱스 덤프
    schema: dict
```

### 5.13 Raw Turn

```python
@dataclass
class RawTurn:
    turn_id: str               # UUID (재추출 idempotency용)
    timestamp: datetime
    user: str
    assistant: str
```

---

## 6. 저장소 스키마

### 6.1 디렉토리 레이아웃

```
~/.engram/
└── users/
    ├── alice/
    │   ├── engram.db              # Tier 2 (SQLite)
    │   ├── raw.jsonl.gz           # Tier 1 (append-only)
    │   └── embed_cache/           # 로컬 임베딩 모델 캐시
    └── bob/
        ├── engram.db
        └── raw.jsonl.gz
```

### 6.2 SQLite 스키마

```sql
-- ============================================
-- Core: Extraction Runs
-- ============================================
CREATE TABLE extraction_runs (
    id                TEXT PRIMARY KEY,
    source_turn_id    TEXT NOT NULL,
    extractor_version TEXT NOT NULL,
    occurred_at       TIMESTAMP NOT NULL,
    processed_at      TIMESTAMP NOT NULL,
    status            TEXT NOT NULL,
    error             TEXT,
    event_count       INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX idx_extraction_runs_source_turn
    ON extraction_runs(source_turn_id, occurred_at);
CREATE UNIQUE INDEX idx_extraction_runs_success_unique
    ON extraction_runs(source_turn_id, extractor_version)
    WHERE status = 'SUCCEEDED';

CREATE TABLE superseded_runs (
    old_run_id      TEXT PRIMARY KEY REFERENCES extraction_runs(id),
    new_run_id      TEXT NOT NULL REFERENCES extraction_runs(id),
    superseded_at   TIMESTAMP NOT NULL
);
CREATE INDEX idx_superseded_runs_new_run_id
    ON superseded_runs(new_run_id);

-- ============================================
-- Core: Event Store
-- ============================================
CREATE TABLE events (
    id                TEXT PRIMARY KEY,
    occurred_at       TIMESTAMP NOT NULL,
    recorded_at       TIMESTAMP NOT NULL,
    type              TEXT NOT NULL,
    data              JSON NOT NULL,
    caused_by         TEXT REFERENCES events(id),
    extraction_run_id TEXT REFERENCES extraction_runs(id),
    seq               INTEGER NOT NULL  -- monotonic 순서 보장
);
CREATE INDEX idx_events_occurred_at ON events(occurred_at);
CREATE INDEX idx_events_recorded_at ON events(recorded_at);
CREATE INDEX idx_events_type ON events(type);
CREATE INDEX idx_events_seq ON events(seq);
CREATE INDEX idx_events_caused_by ON events(caused_by);
CREATE INDEX idx_events_entity_id 
    ON events(json_extract(data, '$.id'));
CREATE INDEX idx_events_extraction_run_id
    ON events(extraction_run_id);

-- ============================================
-- Core: Snapshots
-- ============================================
CREATE TABLE snapshots (
    id           TEXT PRIMARY KEY,
    timestamp    TIMESTAMP NOT NULL,
    event_id     TEXT NOT NULL REFERENCES events(id),
    event_count  INTEGER NOT NULL,
    state        JSON NOT NULL,
    relations    JSON NOT NULL,
    schema       JSON NOT NULL
);
CREATE INDEX idx_snapshots_event_count ON snapshots(event_count);

-- ============================================
-- Operational Writeback
-- ============================================
CREATE TABLE action_types (
    name           TEXT PRIMARY KEY,
    system         TEXT NOT NULL,
    description    TEXT NOT NULL,
    executor       TEXT NOT NULL,
    approval_mode  TEXT NOT NULL,
    reversible     INTEGER NOT NULL,
    compensator    TEXT,
    input_schema   JSON NOT NULL,
    policy         JSON NOT NULL
);

CREATE TABLE action_intents (
    id               TEXT PRIMARY KEY,
    proposed_at      TIMESTAMP NOT NULL,
    action_type      TEXT NOT NULL REFERENCES action_types(name),
    payload          JSON NOT NULL,
    proposed_by      TEXT NOT NULL,
    reason           TEXT,
    target_entities  JSON NOT NULL,
    approval_mode    TEXT NOT NULL,
    status           TEXT NOT NULL,
    idempotency_key  TEXT NOT NULL,
    approved_by      TEXT,
    approved_at      TIMESTAMP,
    rejected_by      TEXT,
    rejected_at      TIMESTAMP,
    resolution_note  TEXT
);
CREATE INDEX idx_action_intents_type ON action_intents(action_type);
CREATE INDEX idx_action_intents_status ON action_intents(status);
CREATE UNIQUE INDEX idx_action_intents_idempotency
    ON action_intents(action_type, idempotency_key)
    WHERE idempotency_key IS NOT NULL;

CREATE TABLE action_executions (
    id              TEXT PRIMARY KEY,
    intent_id       TEXT NOT NULL REFERENCES action_intents(id),
    status          TEXT NOT NULL,
    attempt         INTEGER NOT NULL,
    executor        TEXT NOT NULL,
    started_at      TIMESTAMP,
    finished_at     TIMESTAMP,
    result          JSON,
    error           TEXT,
    reverted_from   TEXT REFERENCES action_executions(id)
);
CREATE INDEX idx_action_executions_intent ON action_executions(intent_id);
CREATE INDEX idx_action_executions_status ON action_executions(status);
CREATE UNIQUE INDEX idx_action_executions_attempt
    ON action_executions(intent_id, attempt);

CREATE TABLE external_objects (
    id           TEXT PRIMARY KEY,
    system       TEXT NOT NULL,
    ref_type     TEXT NOT NULL,
    external_id  TEXT NOT NULL,
    attributes   JSON NOT NULL,
    updated_at   TIMESTAMP NOT NULL
);
CREATE UNIQUE INDEX idx_external_objects_unique
    ON external_objects(system, ref_type, external_id);

CREATE TABLE external_bindings (
    id                  TEXT PRIMARY KEY,
    owner_kind          TEXT NOT NULL,
    owner_id            TEXT NOT NULL,
    external_object_id  TEXT NOT NULL REFERENCES external_objects(id),
    role                TEXT NOT NULL,
    bound_at            TIMESTAMP NOT NULL
);
CREATE UNIQUE INDEX idx_external_bindings_owner_role
    ON external_bindings(owner_kind, owner_id, role);
CREATE INDEX idx_external_bindings_object
    ON external_bindings(external_object_id);

CREATE TABLE action_effects (
    id                  TEXT PRIMARY KEY,
    execution_id        TEXT NOT NULL REFERENCES action_executions(id),
    external_object_id  TEXT NOT NULL REFERENCES external_objects(id),
    effect_type         TEXT NOT NULL,
    payload             JSON NOT NULL,
    recorded_at         TIMESTAMP NOT NULL
);
CREATE INDEX idx_action_effects_execution
    ON action_effects(execution_id);
CREATE INDEX idx_action_effects_external_object
    ON action_effects(external_object_id);

-- ============================================
-- Retrieval: Semantic Index (sqlite-vec)
-- ============================================
-- 엔티티별 임베딩
-- 실제 차원은 초기화 시 선택된 embedder.dim 으로 DDL 템플릿 치환
-- 예: MiniLM = 384, text-embedding-3-small = 1536
CREATE VIRTUAL TABLE vec_entities USING vec0(
    entity_id    TEXT PRIMARY KEY,
    embedding    FLOAT[{{EMBEDDING_DIM}}]
);

-- 이벤트별 임베딩 (변경 이유 등 의미 있는 이벤트만)
CREATE VIRTUAL TABLE vec_events USING vec0(
    event_id     TEXT PRIMARY KEY,
    embedding    FLOAT[{{EMBEDDING_DIM}}],
    occurred_at  INTEGER       -- unix ts (temporal filter용)
);

-- ============================================
-- Retrieval: Entity Text Index (FTS5)
-- ============================================
CREATE VIRTUAL TABLE entity_fts USING fts5(
    entity_id UNINDEXED,
    text,                      -- 엔티티 텍스트 표현 (이름, 별칭, 속성 값)
    tokenize = 'unicode61'
);

-- ============================================
-- Metadata
-- ============================================
CREATE TABLE meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
-- 예: schema_version, embedding_model, embedding_dim, created_at 등

-- ============================================
-- PRAGMA 설정
-- ============================================
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA foreign_keys = ON;
PRAGMA temp_store = MEMORY;
PRAGMA mmap_size = 268435456;  -- 256MB
```

### 6.3 Raw Log 포맷

```jsonl
{"turn_id":"uuid1","timestamp":"2026-04-15T10:00:00Z","user":"...","assistant":"..."}
{"turn_id":"uuid2","timestamp":"2026-04-15T10:01:00Z","user":"...","assistant":"..."}
```

- gzip 압축 (`raw.jsonl.gz`)
- append-only
- `turn_id`는 재추출 시 idempotency 보장

---

## 7. Public API

### 7.1 Engram 클래스

```python
from typing import Literal, Callable, Any
from datetime import datetime

class Engram:
    def __init__(
        self,
        user_id: str = "default",
        path: str | None = None,
        raw_log: bool = True,
        raw_mirror: Literal["none", "s3"] = "none",
        s3_bucket: str | None = None,
        extraction: Literal["local", "llm", "hybrid"] = "hybrid",
        llm_client: Any = None,               # openai.Client | anthropic.Client
        llm_model: str = "claude-haiku-4-5",
        extractor_version: str = "v1",
        embedding: Literal["local", "openai", "custom"] = "local",
        embedding_model: str = "all-MiniLM-L6-v2",
        embedding_fn: Callable[[str], list[float]] | None = None,
        action_handlers: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] | None = None,
        snapshot_interval: int = 1000,
        queue_max_size: int = 10000,
        queue_overflow: Literal["block", "error"] = "block",
    ) -> None: ...
```

### 7.2 Write API

```python
def turn(self, user: str, assistant: str) -> None:
    """
    대화 턴 기록. Fire-and-Forget.
    
    1. 로컬 Raw Log에 동기 append + fsync (durability)
    2. `turn_id`와 `turn_timestamp`를 큐 아이템에 싣고 추출 큐에 넣음
    3. 선택 시 S3 mirror는 비동기 백업
    
    레이턴시: ~1-5ms (local SSD + fsync 기준)
    
    큐 가득 참 시 queue_overflow 설정에 따라 동작:
      - "block": 블로킹 (기본)
      - "error": QueueFullError 발생
    """

def turn_sync(self, user: str, assistant: str) -> list[Event]:
    """
    동기 처리. 추출 완료까지 블로킹. 
    레이턴시: ~1-3초 (LLM 호출 포함). 테스트/디버깅용.
    """

async def turn_async(self, user: str, assistant: str) -> list[Event]:
    """비동기 처리. async 앱용."""

def append(
    self, 
    event_type: str, 
    data: dict, 
    caused_by: str | None = None,
    occurred_at: datetime | None = None,
) -> str:
    """
    수동 이벤트 추가. LLM 추출 우회.
    `occurred_at`를 주지 않으면 now를 사용.
    
    Returns: event_id
    
    Example:
        mem.append("entity.create", {
            "id": "user:alice",
            "type": "user",
            "attrs": {"name": "Alice", "timezone": "Asia/Seoul"}
        })
    """
```

### 7.3 Context API

```python
def context(
    self,
    query: str | None = None,
    max_tokens: int = 4000,
    top_k: int = 20,
    weights: dict[str, float] | None = None,
    include_raw: bool = False,
    raw_turns: int = 10,
    include_history: bool = True,
    temporal_window: tuple[datetime, datetime] | None = None,
) -> str:
    """
    LLM 주입용 컨텍스트 생성.
    
    Args:
        query: 현재 사용자 입력. None이면 최근성 중심 덤프.
        max_tokens: 엄격 준수되는 상한.
        top_k: retrieval 후보 수.
        weights: 4축 가중치 커스텀. 
                 기본: {"entity":0.35, "semantic":0.30, 
                        "recency":0.20, "causal":0.15}
        include_raw: 최근 원본 포함 여부.
        raw_turns: 포함할 원본 턴 수.
        include_history: supporting_event_ids 기반 변경 히스토리 포함.
        temporal_window: (start, end) 기간 필터. Event.occurred_at 기준.
    
    Returns:
        구조화된 텍스트 (시스템 프롬프트 삽입용).
    """
```

### 7.4 Retrieval API

```python
def search(
    self,
    query: str,
    top_k: int = 10,
    axis: Literal["all", "entity", "semantic", "temporal", "causal"] = "all",
) -> list[SearchResult]:
    """
    Low-level retrieval. entity hit와 event hit를 모두 반환할 수 있다.
    디버깅·리랭킹·커스텀 컨텍스트 구성에 사용.
    """

def recall(
    self,
    entity_id: str,
    depth: int = 2,
    include_events: bool = True,
    include_raw: bool = False,
) -> RecallBundle:
    """
    단일 엔티티 중심 풀 번들.
    상태 + 관계 (depth-hop) + 이벤트 + (선택) raw 언급.
    """
```

### 7.5 Query API

```python
def get(self, entity_id: str) -> Entity | None:
    """현재 상태 (스냅샷 캐시)."""

def get_at(self, entity_id: str, at: datetime) -> Entity | None:
    """
    특정 시점 상태. 
    가장 가까운 이전 스냅샷 + `occurred_at <= at` 이벤트 replay.
    """

def history(
    self, 
    entity_id: str, 
    attr: str | None = None,
) -> list[HistoryEntry]:
    """변경 히스토리. attr 지정 시 해당 속성만."""

def related(
    self,
    entity_id: str,
    rel_type: str | None = None,
    direction: Literal["in", "out", "both"] = "out",
    depth: int = 1,
) -> list[Entity]:
    """관계 그래프 탐색."""

def causal_chain(
    self, 
    event_id: str, 
    direction: Literal["upstream", "downstream"] = "upstream",
) -> list[Event]:
    """caused_by 체인 추적."""
```

### 7.6 Action API

```python
def register_action_type(self, action_type: ActionType) -> None:
    """실행 가능한 typed action 등록."""

def propose_action(
    self,
    action_type: str,
    payload: dict[str, Any],
    *,
    reason: str | None = None,
    proposed_by: str = "agent",
    target_entities: list[str] | None = None,
    idempotency_key: str,
) -> ActionIntent:
    """
    에이전트/사용자가 액션을 '제안'한다.
    여기서는 아직 외부 시스템 writeback이 일어나지 않는다.

    같은 `(action_type, idempotency_key)` + 같은 canonical payload면 기존 intent를 반환하고,
    같은 키에 다른 payload면 `IdempotencyConflictError`를 낸다.
    """

def approve_action(
    self,
    intent_id: str,
    approver: str,
    note: str | None = None,
) -> ActionIntent:
    """승인 필요 액션을 APPROVED 상태로 전이."""

def reject_action(
    self,
    intent_id: str,
    approver: str,
    note: str | None = None,
) -> ActionIntent:
    """제안된 액션을 거절하고 실행 불가로 표시."""

def cancel_action(
    self,
    intent_id: str,
    actor: str,
    note: str | None = None,
) -> ActionIntent:
    """아직 실행되지 않은 intent를 취소."""

def execute_action(
    self,
    intent_id: str,
    *,
    dry_run: bool = False,
) -> ActionExecution:
    """
    등록된 handler를 호출해 실제 외부 시스템에 writeback.
    정책상 승인 필요인데 미승인 상태면 예외.

    idempotency 규칙:
      - latest attempt가 RUNNING이면 그 execution을 반환
      - latest attempt가 SUCCEEDED이면 같은 결과를 반환
      - latest attempt가 FAILED이면 새 attempt를 생성
    """

def revert_action(
    self,
    execution_id: str,
    *,
    reason: str | None = None,
) -> ActionExecution:
    """
    reversible action이면 compensator를 호출해 되돌림.
    불가능한 액션은 명시적으로 실패를 기록.
    """

def bind_external_object(
    self,
    owner_kind: Literal["entity"],
    owner_id: str,
    system: str,
    ref_type: str,
    external_id: str,
    role: str = "primary",
    attributes: dict[str, Any] | None = None,
) -> str:
    """내부 owner와 canonical external object를 바인딩."""

def record_action_effect(
    self,
    execution_id: str,
    system: str,
    ref_type: str,
    external_id: str,
    effect_type: str,
    payload: dict[str, Any] | None = None,
) -> str:
    """action execution이 외부 객체에 남긴 효과를 감사 로그로 기록."""

def pending_actions(
    self,
    status: Literal["PROPOSED", "APPROVED"] | None = None,
) -> list[ActionIntent]:
    """승인 대기 / 실행 대기 액션 조회."""

def action_executions(self, intent_id: str) -> list[ActionExecution]:
    """단일 intent에 대한 execution attempt 이력."""
```

### 7.7 Raw Log API

```python
def raw_recent(self, n: int = 10) -> list[RawTurn]:
    """최근 n개 턴."""

def raw_search(self, query: str, limit: int = 10) -> list[RawTurn]:
    """키워드 전문 검색 (fallback)."""

def raw_iter(
    self, 
    from_time: datetime | None = None,
    to_time: datetime | None = None,
) -> Iterator[RawTurn]:
    """전체 순회 (분석용)."""
```

### 7.8 Maintenance API

```python
def reprocess(
    self,
    from_time: datetime | None = None,
    to_time: datetime | None = None,
    extractor_version: str | None = None,
) -> ReprocessResult:
    """
    원본에서 재추출. 
    
    같은 `(turn_id, extractor_version)`의 성공 run이 이미 있으면 skip.
    새 extractor_version으로 재처리하면 새 extraction_run + events를 append하고,
    이전 active run은 `superseded_runs`에 기록되어 active query에서 제외.
    
    Returns:
        ReprocessResult(runs_created, runs_superseded, events_created, duration)
    """

def compact(self) -> None:
    """강제 스냅샷 생성."""

def prune(self, keep_days: int = 90) -> int:
    """오래된 파생 아티팩트만 정리. events/raw log는 삭제하지 않음."""

def stats(self) -> Stats:
    """통계: extraction run/이벤트/액션 수, 엔티티 수, 큐 크기, 저장소 크기 등."""

def flush(self, timeout: float = 5.0) -> int:
    """현재 큐에 accepted된 항목이 skip 또는 Tier 2 commit까지 끝날 때 대기."""

def close(self) -> None:
    """모든 리소스 정리."""

def __enter__(self): return self
def __exit__(self, *args): self.close()
```

---

## 8. 타입 시스템

```python
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal


@dataclass
class Event:
    id: str
    occurred_at: datetime
    recorded_at: datetime
    type: str
    data: dict[str, Any]
    caused_by: str | None = None
    extraction_run_id: str | None = None
    seq: int = 0


@dataclass
class ExtractionRun:
    id: str
    source_turn_id: str
    extractor_version: str
    occurred_at: datetime
    processed_at: datetime
    status: Literal["SUCCEEDED", "FAILED", "SKIPPED"]
    error: str | None = None
    event_count: int = 0


@dataclass
class Entity:
    id: str
    type: str
    attributes: dict[str, Any]
    created_at: datetime
    updated_at: datetime


@dataclass
class Relation:
    source: str
    target: str
    type: str
    attributes: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ActionType:
    name: str
    system: str
    description: str
    input_schema: dict[str, Any]
    executor: str
    approval_mode: Literal["auto", "human_required", "policy"]
    reversible: bool
    compensator: str | None = None


@dataclass
class ActionIntent:
    id: str
    action_type: str
    payload: dict[str, Any]
    proposed_at: datetime
    proposed_by: str
    reason: str | None
    target_entities: list[str]
    approval_mode: Literal["auto", "human_required", "policy"]
    status: Literal["PROPOSED", "APPROVED", "REJECTED", "CANCELED"]
    idempotency_key: str
    approved_by: str | None = None
    approved_at: datetime | None = None
    rejected_by: str | None = None
    rejected_at: datetime | None = None
    resolution_note: str | None = None


@dataclass
class ActionExecution:
    id: str
    intent_id: str
    status: Literal["RUNNING", "SUCCEEDED", "FAILED", "REVERTED"]
    attempt: int
    executor: str
    started_at: datetime | None = None
    finished_at: datetime | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    reverted_from: str | None = None


@dataclass
class ExternalObject:
    id: str
    system: str
    ref_type: str
    external_id: str
    attributes: dict[str, Any]
    updated_at: datetime


@dataclass
class ExternalBinding:
    id: str
    owner_kind: Literal["entity"]
    owner_id: str
    external_object_id: str
    role: str
    bound_at: datetime


@dataclass
class ActionEffect:
    id: str
    execution_id: str
    external_object_id: str
    effect_type: str
    payload: dict[str, Any]
    recorded_at: datetime


@dataclass
class Snapshot:
    id: str
    timestamp: datetime
    event_id: str
    event_count: int
    state: dict[str, Any]
    relations: dict[str, Any]
    schema: dict[str, Any]


@dataclass
class RawTurn:
    turn_id: str
    timestamp: datetime
    user: str
    assistant: str


@dataclass
class QueueItem:
    turn_id: str
    turn_timestamp: datetime
    user: str
    assistant: str
    enqueued_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class HistoryEntry:
    occurred_at: datetime
    attribute: str
    new_value: Any
    old_value: Any | None
    event_id: str
    reason: str | None = None


@dataclass
class SearchResult:
    kind: Literal["entity", "event"]
    entity_id: str | None         # 엔티티 매칭 시
    event_id: str | None          # 이벤트 매칭 시
    score: float
    matched_axis: Literal["entity", "semantic", "temporal", "causal"]
    snippet: str | None = None


@dataclass
class RetrievalCandidate:
    """내부 용도. HybridRanker 입력."""
    entity_id: str
    entity_match: float     # 0-1
    semantic_score: float   # 0-1
    recency_score: float    # 0-1
    causal_score: float     # 0-1
    supporting_event_ids: list[str] = field(default_factory=list)
    importance: float = 1.0 # 변경 빈도 / 참조 횟수


@dataclass
class RecallBundle:
    entity: Entity
    related: list[Entity]
    events: list[Event]
    raw_mentions: list[RawTurn]


@dataclass
class ReprocessResult:
    runs_created: int
    runs_superseded: int
    events_created: int
    duration_seconds: float
    errors: list[str]


@dataclass
class Stats:
    event_count: int
    extraction_run_count: int
    entity_count: int
    relation_count: int
    action_intent_count: int
    action_execution_count: int
    pending_action_count: int
    queue_size: int
    queue_rejected: int
    raw_log_bytes: int
    db_bytes: int
    last_snapshot_at: datetime | None


class EngramError(Exception): ...
class QueueFullError(EngramError): ...
class ExtractionError(EngramError): ...
class StorageError(EngramError): ...
class IdempotencyConflictError(EngramError): ...
```

---

## 9. Write Path: 추출 파이프라인

### 9.1 2단계 필터링

**목적:** LLM 호출 비용 절감. 로컬에서 70%를 선필터.

```python
class LocalFilter:
    """규칙 기반 로컬 필터. ~1ms."""
    
    # 무시해도 되는 패턴
    IGNORE_PATTERNS = [
        r"^(안녕|하이|헬로|반가워)",
        r"^(날씨|오늘|내일).*(좋|덥|춥|비)",
        r"^(ㅋ|ㅎ|ㅠ|ㅜ)+$",
        r"^(네|응|그래|알겠어|오키|ok)$",
        r"^(감사|고마워|땡큐)",
        r"^[?!.,\s]*$",
    ]
    
    # 추출 가치 높은 패턴
    EXTRACT_PATTERNS = [
        r"(나는|난|저는|내가).{0,20}(이야|예요|입니다|야)",
        r"(좋아|싫어|선호|원해|하고\s?싶)",
        r"(하기로|결정|선택|정했)",
        r"(변경|바꿔|수정|이제는)",
        r"(기억|잊지|메모|저장)",
        r"(사는|살고|이사|위치|거주)",
        r"(프로젝트|회사|일|업무)",
    ]
    
    def classify(
        self, 
        user: str, 
        assistant: str,
    ) -> Literal["skip", "extract", "uncertain"]:
        combined = user + " " + assistant
        
        if any(re.search(p, combined) for p in self.IGNORE_PATTERNS):
            return "skip"
        if any(re.search(p, combined) for p in self.EXTRACT_PATTERNS):
            return "extract"
        # 길이 휴리스틱
        if len(user) < 10 and len(assistant) < 30:
            return "skip"
        return "uncertain"  # LLM으로 넘김
```

### 9.2 LLM 추출

```python
EXTRACTION_PROMPT = """당신은 대화에서 장기 기억할 만한 사실을 추출하는 시스템입니다.

다음 대화 턴에서 추출할 것:
- 사용자/엔티티 속성 변경
- 선호·결정·취향
- 관계 (누가 누구와, 무엇을 소유/담당 등)
- 시간 관련 사실 (계획, 약속)

추출하지 말 것:
- 일시적 감정/날씨/인사
- 가정법·농담·추측
- 중복 정보 (같은 세션 내)

기존 엔티티 (매칭 우선 고려):
{known_entities}

현재 시각: {now}

대화:
User: {user}
Assistant: {assistant}

다음 JSON으로만 응답:
{{
  "events": [
    {{
      "type": "entity.create" | "entity.update" | "relation.create" | ...,
      "data": {{...}},
      "confidence": 0.0-1.0,
      "reason": "왜 이 이벤트를 추출했는지"
    }}
  ]
}}

아무것도 추출할 게 없으면 events는 빈 배열."""


class LLMExtractor:
    def __init__(self, llm: Any, version: str = "v1"):
        self.llm = llm
        self.version = version
    
    def extract(
        self, 
        run_id: str,
        turn_id: str,
        turn_timestamp: datetime,
        user: str, 
        assistant: str, 
        known_entities: list[Entity],
    ) -> list[Event]:
        prompt = EXTRACTION_PROMPT.format(
            known_entities=self._format_entities(known_entities),
            now=datetime.utcnow().isoformat(),
            user=user,
            assistant=assistant,
        )
        
        resp = self.llm.complete(prompt, response_format="json")
        parsed = json.loads(resp)
        
        events = []
        for item in parsed["events"]:
            if item.get("confidence", 0) < self.confidence_threshold:
                continue
            events.append(Event(
                id=str(uuid4()),
                occurred_at=turn_timestamp,
                recorded_at=datetime.utcnow(),
                type=item["type"],
                data=item["data"],
                caused_by=None,
                extraction_run_id=run_id,
            ))
        return events
```

핵심 규칙:
- `turn_timestamp`는 raw log에 적힌 원본 대화 시각이다.
- 재처리든 실시간 처리든 추출 이벤트의 `occurred_at`은 항상 이 값을 사용한다.
- 그래서 나중에 추출기가 바뀌어도 시간여행 결과는 바뀌지 않고, 바뀌는 것은 `recorded_at`과 run lineage뿐이다.

### 9.3 Worker

```python
class BackgroundWorker:
    """큐에서 꺼내 파이프라인 실행. 단일 스레드 / user_id."""
    
    def __init__(self, queue: WriteQueue, pipeline: ExtractionPipeline):
        self.queue = queue
        self.pipeline = pipeline
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
    
    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
    
    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                item = self.queue.get(timeout=0.5)
            except Empty:
                continue
            if item is SENTINEL:
                self.queue.task_done()
                break
            
            try:
                self.pipeline.process(item)
            except Exception as e:
                logger.exception("extraction failed: %s", e)
                # Raw log엔 이미 저장됨 → reprocess()로 복구 가능
            finally:
                self.queue.task_done()
    
    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout)
```

### 9.4 파이프라인 조립

```python
class ExtractionPipeline:
    def __init__(
        self,
        filter_: LocalFilter,
        extractor: LLMExtractor,
        conflict_detector: ConflictDetector,
        store: EventStore,
        state_cache: StateCache,
        relations: RelationIndex,
        semantic_idx: SemanticIndex,
        entity_idx: EntityIndex,
        snapshot_mgr: SnapshotManager,
    ) -> None:
        ...
    
    def process(self, item: QueueItem) -> None:
        user, assistant = item.user, item.assistant
        
        if self.store.already_processed(
            item.turn_id,
            self.extractor.version,
        ):
            return
        
        # 1. 로컬 필터
        decision = self.filter_.classify(user, assistant)
        if decision == "skip":
            self.store.record_run(ExtractionRun(
                id=str(uuid4()),
                source_turn_id=item.turn_id,
                extractor_version=self.extractor.version,
                occurred_at=item.turn_timestamp,
                processed_at=datetime.utcnow(),
                status="SKIPPED",
                event_count=0,
            ))
            return
        
        # 2. extraction run 생성
        run = ExtractionRun(
            id=str(uuid4()),
            source_turn_id=item.turn_id,
            extractor_version=self.extractor.version,
            occurred_at=item.turn_timestamp,
            processed_at=datetime.utcnow(),
            status="FAILED",  # 성공 시 아래에서 뒤집음
            error=None,
            event_count=0,
        )

        # 3. LLM 추출
        known = self.state_cache.top_entities(50)
        try:
            events = self.extractor.extract(
                run.id,
                item.turn_id,
                item.turn_timestamp,
                user,
                assistant,
                known,
            )
        except Exception as e:
            run.processed_at = datetime.utcnow()
            run.status = "FAILED"
            run.error = str(e)
            self.store.record_run(run)
            raise
        
        # 4. 충돌 감지
        events = self.conflict_detector.annotate(events, self.state_cache)
        old_run_id = self.store.active_run_id_for_turn(item.turn_id)
        run.processed_at = datetime.utcnow()
        run.status = "SUCCEEDED"
        run.event_count = len(events)
        
        # 5. 커밋 (트랜잭션)
        with self.store.transaction() as tx:
            self.store.record_run(run, tx=tx)
            for event in events:
                self.store.append(tx, event)
                self.state_cache.apply(event)
                self.relations.apply(event)
                self.entity_idx.apply(event)
                # 임베딩은 트랜잭션 밖에서 (비용 큼)
            if old_run_id:
                self.store.mark_run_superseded(tx, old_run_id, run.id)
        
        # 6. 임베딩 (best-effort)
        for event in events:
            try:
                self.semantic_idx.index_event(event)
            except Exception:
                logger.warning("semantic index failed for %s", event.id)
        
        # 7. 엔티티 임베딩 + 스냅샷 체크
        for event in events:
            self.semantic_idx.refresh_entities_for_event(event, self.state_cache)
        self.snapshot_mgr.maybe_create(self.state_cache, self.relations)
```

설명:
- `already_processed()`는 event 존재가 아니라 **성공한 extraction run 존재 여부**를 본다.
- `mark_run_superseded()`는 이전 run 전체를 새 run 전체로 대체했다는 lineage를 남긴다.
- turn 하나에서 여러 이벤트가 나와도 lineage가 run 단위라 감사와 디버깅이 단순해진다.

---

## 10. Read Path: Retrieval 파이프라인

### 10.1 4축 개요

| 축 | 신호 | 구현 | 비용 |
|----|------|------|------|
| Entity | 쿼리 내 엔티티 언급이 연결된 이벤트 | FTS5 + entity→event 확장 | 저 |
| Semantic | 이벤트 문맥 유사도 | sqlite-vec (`vec_events`) | 중 |
| Temporal | `occurred_at` 시간 윈도우 | SQL range scan | 저 |
| Causal | caused_by 기반 event graph | 그래프 탐색 | 중 |

v2.2의 핵심 변화는 **엔티티를 직접 찾는 대신, 먼저 관련 이벤트를 찾고 그 결과를 엔티티로 투영한다**는 점이다.
이 방식이 필요한 이유는 "무슨 일이 있었나"라는 질문이 종종 엔티티 이름보다 이벤트 문맥에 더 강하게 묶여 있기 때문이다.

### 10.2 EntityMatcher

```python
class EntityMatcher:
    """쿼리에서 엔티티 언급을 찾고, 그 엔티티와 관련된 seed event를 찾는다."""
    
    def __init__(self, entity_idx: EntityIndex):
        self.entity_idx = entity_idx
    
    def match_entities(self, query: str) -> list[tuple[str, float]]:
        """
        Returns: [(entity_id, match_score), ...]
        
        전략:
        1. FTS5로 쿼리 토큰과 엔티티 텍스트 매칭
        2. BM25 스코어 정규화 → [0, 1]
        3. 완전 일치 > 부분 일치 > 퍼지 매치
        """
        results = self.entity_idx.search_fts(query, limit=20)
        if not results:
            return []
        
        max_score = max(r.bm25 for r in results)
        return [
            (r.entity_id, r.bm25 / max_score) 
            for r in results
        ]

    def match_events(self, query: str, store: EventStore) -> dict[str, float]:
        entity_hits = self.match_entities(query)
        scores: dict[str, float] = {}
        for entity_id, score in entity_hits:
            for event in store.events_for_entity(entity_id, limit=20):
                scores[event.id] = max(scores.get(event.id, 0.0), score)
        return scores
```

### 10.3 SemanticIndex

```python
class SemanticIndex:
    """sqlite-vec 기반 벡터 검색."""
    
    def __init__(self, conn: sqlite3.Connection, embedder: Embedder):
        self.conn = conn
        self.embedder = embedder
        self._validate_dimension()
    
    def index_entity(self, entity: Entity) -> None:
        text = self._entity_to_text(entity)
        emb = self.embedder.embed(text)
        self.conn.execute(
            "INSERT OR REPLACE INTO vec_entities VALUES (?, ?)",
            (entity.id, serialize_vec(emb))
        )
    
    def index_event(self, event: Event) -> None:
        # 의미 있는 이벤트만 (create, update에 reason 있는 것)
        text = self._event_to_text(event)
        if not text:
            return
        emb = self.embedder.embed(text)
        ts = int(event.occurred_at.timestamp())
        self.conn.execute(
            "INSERT OR REPLACE INTO vec_events VALUES (?, ?, ?)",
            (event.id, serialize_vec(emb), ts)
        )
    
    def search_entities(
        self, 
        query: str, 
        k: int = 20,
    ) -> list[tuple[str, float]]:
        emb = self.embedder.embed(query)
        rows = self.conn.execute("""
            SELECT entity_id, distance 
            FROM vec_entities 
            WHERE embedding MATCH ? 
            ORDER BY distance 
            LIMIT ?
        """, (serialize_vec(emb), k)).fetchall()
        # distance → similarity score [0, 1]
        return [(eid, 1.0 - min(d, 1.0)) for eid, d in rows]
    
    def search_events(
        self, 
        query: str, 
        k: int = 20,
        window: tuple[datetime, datetime] | None = None,
    ) -> list[tuple[str, float]]:
        emb = self.embedder.embed(query)
        sql = """
            SELECT event_id, distance, occurred_at
            FROM vec_events 
            WHERE embedding MATCH ?
        """
        params = [serialize_vec(emb)]
        if window:
            start, end = window
            sql += " AND occurred_at BETWEEN ? AND ?"
            params.extend([int(start.timestamp()), int(end.timestamp())])
        sql += " ORDER BY distance LIMIT ?"
        params.append(k)
        
        rows = self.conn.execute(sql, params).fetchall()
        return [(eid, 1.0 - min(d, 1.0)) for eid, d, _ in rows]
    
    def _validate_dimension(self) -> None:
        stored_dim = int(self.conn.execute("""
            SELECT value FROM meta WHERE key = 'embedding_dim'
        """).fetchone()[0])
        if stored_dim != self.embedder.dim:
            raise StorageError(
                f"embedding dim mismatch: db={stored_dim}, embedder={self.embedder.dim}"
            )
```

### 10.4 Embedder 추상화

```python
class Embedder(Protocol):
    dim: int
    def embed(self, text: str) -> list[float]: ...


class LocalEmbedder:
    """sentence-transformers 기반. 첫 호출 시 모델 다운로드."""
    
    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model)
        self.dim = self.model.get_sentence_embedding_dimension()
    
    def embed(self, text: str) -> list[float]:
        return self.model.encode(text, normalize_embeddings=True).tolist()


class OpenAIEmbedder:
    def __init__(self, model: str = "text-embedding-3-small"):
        self.model = model
        self.dim = 1536
    
    def embed(self, text: str) -> list[float]:
        resp = openai.embeddings.create(input=text, model=self.model)
        return resp.data[0].embedding


class CustomEmbedder:
    def __init__(self, fn: Callable[[str], list[float]], dim: int):
        self.fn = fn
        self.dim = dim
    
    def embed(self, text: str) -> list[float]:
        return self.fn(text)
```

### 10.5 TemporalFilter

```python
import math

class TemporalFilter:
    """이벤트 발생 시각 기준 최근성 가중치."""
    
    HALF_LIFE_DAYS = 30  # 튜너블
    
    def score(
        self, 
        latest_occurred_at: datetime,
        now: datetime,
    ) -> float:
        age_days = (now - latest_occurred_at).total_seconds() / 86400
        # exponential decay
        return math.exp(-math.log(2) * age_days / self.HALF_LIFE_DAYS)
```

### 10.6 CausalExpander

```python
class CausalExpander:
    """시드 이벤트로부터 caused_by 체인 전파."""
    
    def expand_events(
        self,
        seed_event_ids: list[str],
        store: EventStore,
        depth: int = 2,
    ) -> dict[str, float]:
        """
        Returns: event_id → causal_score
        
        전파 공식: score = 0.5^distance
        """
        scores: dict[str, float] = {}

        # BFS
        visited: set[str] = set()
        queue: list[tuple[str, int]] = [(event_id, 0) for event_id in seed_event_ids]
        
        while queue:
            event_id, d = queue.pop(0)
            if event_id in visited or d > depth:
                continue
            visited.add(event_id)
            
            event = store.get_event(event_id)
            if not event:
                continue
            
            scores[event_id] = max(scores.get(event_id, 0.0), 0.5 ** d)
            
            # caused_by 체인 확장
            if event.caused_by:
                queue.append((event.caused_by, d + 1))
            # downstream 이벤트도
            for child in store.events_caused_by(event_id):
                queue.append((child.id, d + 1))
        
        return scores
```

### 10.7 HybridRanker

```python
class HybridRanker:
    DEFAULT_WEIGHTS = {
        "entity":   0.35,
        "semantic": 0.30,
        "recency":  0.20,
        "causal":   0.15,
    }
    
    def rank(
        self,
        candidates: list[RetrievalCandidate],
        weights: dict[str, float] | None = None,
    ) -> list[tuple[RetrievalCandidate, float]]:
        w = {**self.DEFAULT_WEIGHTS, **(weights or {})}
        
        scored = []
        for c in candidates:
            score = (
                w["entity"]   * c.entity_match +
                w["semantic"] * c.semantic_score +
                w["recency"]  * c.recency_score +
                w["causal"]   * c.causal_score
            ) * c.importance
            scored.append((c, score))
        
        return sorted(scored, key=lambda x: -x[1])
```

### 10.8 Pipeline 조립

```python
class RetrievalPipeline:
    def __init__(
        self,
        entity_matcher: EntityMatcher,
        semantic_idx: SemanticIndex,
        temporal_filter: TemporalFilter,
        causal_expander: CausalExpander,
        ranker: HybridRanker,
        state_cache: StateCache,
        store: EventStore,
    ) -> None: ...
    
    def run(
        self,
        query: str,
        top_k: int = 20,
        weights: dict | None = None,
        temporal_window: tuple[datetime, datetime] | None = None,
    ) -> list[RetrievalCandidate]:
        now = datetime.utcnow()
        
        # 1. seed event 수집
        entity_seed_scores = self.entity_matcher.match_events(query, self.store)
        semantic_event_scores = dict(
            self.semantic_idx.search_events(query, k=top_k * 4, window=temporal_window)
        )
        temporal_seed_scores = {
            event.id: 1.0
            for event in self.store.events_in_range(*temporal_window)
        } if temporal_window else {}

        seed_event_ids: set[str] = set()
        seed_event_ids.update(entity_seed_scores.keys())
        seed_event_ids.update(semantic_event_scores.keys())
        seed_event_ids.update(temporal_seed_scores.keys())

        causal_event_scores = self.causal_expander.expand_events(
            list(seed_event_ids),
            self.store,
            depth=2,
        ) if seed_event_ids else {}
        seed_event_ids.update(causal_event_scores.keys())

        # 2. seed event를 엔티티 후보로 투영
        entity_aggregate: dict[str, RetrievalCandidate] = {}
        for event_id in seed_event_ids:
            event = self.store.get_event(event_id)
            if not event:
                continue
            for entity_id in self._entities_affected(event):
                entity = self.state_cache.get(entity_id)
                if not entity:
                    continue

                cand = entity_aggregate.setdefault(
                    entity_id,
                    RetrievalCandidate(
                        entity_id=entity_id,
                        entity_match=0.0,
                        semantic_score=0.0,
                        recency_score=0.0,
                        causal_score=0.0,
                        supporting_event_ids=[],
                        importance=self._importance(entity),
                    ),
                )
                cand.entity_match = max(cand.entity_match, entity_seed_scores.get(event_id, 0.0))
                cand.semantic_score = max(cand.semantic_score, semantic_event_scores.get(event_id, 0.0))
                cand.causal_score = max(cand.causal_score, causal_event_scores.get(event_id, 0.0))
                cand.recency_score = max(
                    cand.recency_score,
                    self.temporal_filter.score(event.occurred_at, now),
                )
                if event_id not in cand.supporting_event_ids:
                    cand.supporting_event_ids.append(event_id)

        candidates = list(entity_aggregate.values())
        
        # 4. 랭킹
        ranked = self.ranker.rank(candidates, weights)
        return [c for c, _ in ranked[:top_k]]
    
    def _importance(self, entity: Entity) -> float:
        """변경 횟수 / 참조 횟수 기반 중요도."""
        # 간단 구현: updated_at이 자주 바뀌면 중요
        change_count = self.store.count_events_for_entity(entity.id)
        return min(1.0, 0.5 + change_count * 0.05)
```

---

## 11. Context Builder

### 11.1 Token Budget 전략

4개 섹션에 비율로 분배. 각 섹션이 예산 초과 시 잘림.

| 섹션 | 기본 비율 | 내용 |
|------|-----------|------|
| Current State | 40% | Top-ranked 엔티티 속성 |
| Relations | 20% | 1-hop 관계 |
| History | 25% | supporting_event_ids 기반 변경 요약 |
| Raw (선택) | 15% | 최근 원본 턴 |

### 11.2 구현

```python
class ContextBuilder:
    # tiktoken 또는 anthropic token counter
    def __init__(self, token_counter: Callable[[str], int]):
        self.count_tokens = token_counter
    
    def build(
        self,
        ranked: list[RetrievalCandidate],
        state_cache: StateCache,
        relations: RelationIndex,
        store: EventStore,
        raw_log: RawLog,
        max_tokens: int,
        include_raw: bool = False,
        raw_turns: int = 10,
        include_history: bool = True,
    ) -> str:
        sections: list[str] = []
        
        # Priority 1: Current State (40%)
        budget = int(max_tokens * 0.4)
        sections.append(self._render_state(ranked, state_cache, budget))
        
        # Priority 2: Relations (20%)
        budget = int(max_tokens * 0.2)
        sections.append(self._render_relations(ranked, relations, state_cache, budget))
        
        # Priority 3: History (25%)
        if include_history:
            budget = int(max_tokens * 0.25)
            sections.append(self._render_history(ranked, store, budget))
        
        # Priority 4: Raw (15%)
        if include_raw:
            budget = int(max_tokens * 0.15)
            sections.append(self._render_raw(raw_log, raw_turns, budget))
        
        result = "\n\n".join(s for s in sections if s)
        
        # 최종 가드: 토큰 예산 초과 시 하드 잘라냄
        if self.count_tokens(result) > max_tokens:
            result = self._truncate(result, max_tokens)
        
        return result
    
    def _render_state(
        self,
        ranked: list[RetrievalCandidate],
        state: StateCache,
        budget: int,
    ) -> str:
        lines = ["## Current State"]
        used = self.count_tokens(lines[0])
        
        for c in ranked:
            entity = state.get(c.entity_id)
            if not entity:
                continue
            attrs = {k: v for k, v in entity.attributes.items() 
                     if not k.startswith("_")}
            line = f"- {entity.id}: {json.dumps(attrs, ensure_ascii=False)}"
            line_tokens = self.count_tokens(line)
            if used + line_tokens > budget:
                break
            lines.append(line)
            used += line_tokens
        
        return "\n".join(lines)
    
    def _render_history(
        self,
        ranked: list[RetrievalCandidate],
        store: EventStore,
        budget: int,
    ) -> str:
        lines = ["## Recent Changes"]
        used = self.count_tokens(lines[0])

        seen: set[str] = set()
        for c in ranked:
            for event_id in c.supporting_event_ids:
                if event_id in seen:
                    continue
                seen.add(event_id)
                event = store.get_event(event_id)
                if not event or event.type != "entity.update":
                    continue
                line = self._history_line(event)
                line_tokens = self.count_tokens(line)
                if used + line_tokens > budget:
                    return "\n".join(lines)
                lines.append(line)
                used += line_tokens

        return "\n".join(lines)
```

### 11.3 출력 예시

```
## Current State
- user:alice: {"name": "Alice", "location": "Busan", "diet": "vegetarian"}
- project:engram: {"status": "design", "target_release": "2026-Q3"}
- trip:tokyo: {"dates": "2026-04-20~24", "purpose": "vacation"}

## Relations
- user:alice --[owns]--> project:engram
- user:alice --[traveling_to]--> trip:tokyo

## Recent Changes
- user:alice.location: Seoul → Busan (occurred_at=2026-06-20, "발령으로 이사")
- project:engram.status: ideation → design (occurred_at=2026-03-15)

## Recent Conversation
[2026-04-15 10:00] user: 난 채식주의자야
[2026-04-15 10:00] assistant: 알겠습니다...
```

---

## 12. 이벤트 소싱 + 스냅샷

### 12.1 EventStore

```python
class EventStore:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self._seq_counter = self._load_max_seq()
    
    def append(self, tx: Transaction, event: Event) -> None:
        self._seq_counter += 1
        event.seq = self._seq_counter
        self.conn.execute("""
            INSERT INTO events (
                id, occurred_at, recorded_at, type, data,
                caused_by, extraction_run_id, seq
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            event.id,
            event.occurred_at,
            event.recorded_at,
            event.type,
            json.dumps(event.data, ensure_ascii=False),
            event.caused_by,
            event.extraction_run_id,
            event.seq,
        ))
    
    def record_run(self, run: ExtractionRun, tx: Transaction | None = None) -> None: ...
    def already_processed(self, turn_id: str, extractor_version: str) -> bool: ...
    def active_run_id_for_turn(self, turn_id: str) -> str | None: ...
    def mark_run_superseded(
        self,
        tx: Transaction,
        old_run_id: str,
        new_run_id: str,
    ) -> None: ...
    def get_event(self, event_id: str) -> Event | None: ...
    def events_for_entity(
        self,
        entity_id: str,
        limit: int = 100,
        include_superseded: bool = False,
    ) -> list[Event]: ...
    def events_since(
        self,
        seq: int,
        include_superseded: bool = False,
    ) -> Iterator[Event]: ...
    def events_caused_by(self, event_id: str) -> list[Event]: ...
    def events_in_range(self, start: datetime, end: datetime) -> list[Event]: ...
```

Active query 규칙:
- 같은 turn에 여러 extraction run이 있을 수 있다.
- `superseded_runs`에 들어간 run은 active query에서 제외한다.
- 따라서 현재 상태, 히스토리, retrieval은 모두 "active run의 이벤트만 본다"는 계약을 따른다.

### 12.2 SnapshotManager

```python
class SnapshotManager:
    EVENT_THRESHOLD = 1000
    TIME_THRESHOLD = timedelta(hours=24)
    MAX_KEPT = 10
    
    def should_create(
        self, 
        last: Snapshot | None, 
        current_count: int,
    ) -> bool:
        if last is None:
            return current_count >= self.EVENT_THRESHOLD
        return (
            current_count - last.event_count >= self.EVENT_THRESHOLD or
            datetime.utcnow() - last.timestamp >= self.TIME_THRESHOLD
        )
    
    def create(
        self, 
        state: StateCache, 
        relations: RelationIndex,
        last_event_id: str,
        event_count: int,
    ) -> Snapshot:
        snap = Snapshot(
            id=str(uuid4()),
            timestamp=datetime.utcnow(),
            event_id=last_event_id,
            event_count=event_count,
            state=state.dump(),
            relations=relations.dump(),
            schema=self._schema_snapshot(),
        )
        self._persist(snap)
        self._prune_old()
        return snap
    
    def restore_to(self, at: datetime) -> tuple[StateCache, RelationIndex]:
        """
        시간여행 복원.
        1. at 이전의 가장 최근 스냅샷 찾기
        2. 그 스냅샷 이후 ~ at 까지 active 이벤트 replay
        """
        snap = self._find_snapshot_before(at)
        state = StateCache.from_dump(snap.state) if snap else StateCache()
        relations = RelationIndex.from_dump(snap.relations) if snap else RelationIndex()
        
        start_seq = snap.event_count if snap else 0
        for event in self.store.events_since(start_seq):
            if event.occurred_at > at:
                break
            state.apply(event)
            relations.apply(event)
        
        return state, relations
```

### 12.3 StateCache

```python
class StateCache:
    """메모리 기반 현재 상태. 이벤트 replay 결과."""
    
    def __init__(self):
        self._entities: dict[str, Entity] = {}
    
    def apply(self, event: Event) -> None:
        if event.type == "entity.create":
            data = event.data
            self._entities[data["id"]] = Entity(
                id=data["id"],
                type=data["type"],
                attributes=data.get("attrs", {}),
                created_at=event.occurred_at,
                updated_at=event.occurred_at,
            )
        elif event.type == "entity.update":
            eid = event.data["id"]
            if eid in self._entities:
                attrs = {k: v for k, v in event.data.items() 
                         if k not in ("id", "_changed_from")}
                self._entities[eid].attributes.update(attrs)
                self._entities[eid].updated_at = event.occurred_at
        elif event.type == "entity.delete":
            self._entities.pop(event.data["id"], None)
    
    def get(self, entity_id: str) -> Entity | None:
        return self._entities.get(entity_id)
    
    def top_entities(self, n: int) -> list[Entity]:
        return sorted(
            self._entities.values(), 
            key=lambda e: e.updated_at, 
            reverse=True,
        )[:n]
    
    def dump(self) -> dict: ...
    @classmethod
    def from_dump(cls, data: dict) -> StateCache: ...
```

### 12.4 RelationIndex

```python
class RelationIndex:
    """인메모리 양방향 그래프. O(1) 인접 조회."""
    
    def __init__(self):
        self.outgoing: dict[str, list[tuple[str, str]]] = defaultdict(list)
        # source → [(target, rel_type), ...]
        self.incoming: dict[str, list[tuple[str, str]]] = defaultdict(list)
        self.by_type: dict[str, list[tuple[str, str]]] = defaultdict(list)
    
    def apply(self, event: Event) -> None:
        if event.type == "relation.create":
            d = event.data
            self.outgoing[d["source"]].append((d["target"], d["type"]))
            self.incoming[d["target"]].append((d["source"], d["type"]))
            self.by_type[d["type"]].append((d["source"], d["target"]))
        elif event.type == "relation.delete":
            # 제거 로직
            ...
    
    def adjacent(
        self, 
        entity_id: str, 
        direction: Literal["in", "out", "both"] = "out",
    ) -> list[tuple[str, str]]: ...
    
    def traverse(
        self, 
        entity_id: str, 
        depth: int = 2,
    ) -> set[str]: ...
```

---

## 13. Raw Log (Tier 1)

### 13.1 로컬 구현

```python
class LocalRawLog:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
    
    def append(self, turn_id: str, user: str, assistant: str) -> RawTurn:
        entry = RawTurn(
            turn_id=turn_id,
            timestamp=datetime.utcnow(),
            user=user,
            assistant=assistant,
        )
        line = json.dumps({
            "turn_id": entry.turn_id,
            "timestamp": entry.timestamp.isoformat(),
            "user": entry.user,
            "assistant": entry.assistant,
        }, ensure_ascii=False) + "\n"
        
        with self._lock:
            with gzip.open(self.path, "at", encoding="utf-8") as f:
                f.write(line)
                f.flush()
                os.fsync(f.fileno())  # durability ACK 경로
        
        return entry
    
    def iter(
        self, 
        from_time: datetime | None = None,
        to_time: datetime | None = None,
    ) -> Iterator[RawTurn]:
        if not self.path.exists():
            return
        with gzip.open(self.path, "rt", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                ts = datetime.fromisoformat(entry["timestamp"])
                if from_time and ts < from_time:
                    continue
                if to_time and ts > to_time:
                    break
                yield RawTurn(**entry)
    
    def recent(self, n: int) -> list[RawTurn]:
        # 효율화: 파일 끝에서부터 읽기. 간단 구현은 전체 스캔 후 끝.
        turns = list(self.iter())
        return turns[-n:]
    
    def search(self, query: str, limit: int = 10) -> list[RawTurn]:
        q = query.lower()
        results = []
        for turn in self.iter():
            if q in turn.user.lower() or q in turn.assistant.lower():
                results.append(turn)
                if len(results) >= limit:
                    break
        return results
```

`RawTurn.timestamp`는 v2.2에서 단순 기록 시간이 아니라 **이후 모든 추출 이벤트의 occurred_at 기준점**이다.

### 13.2 S3 mirror (옵션)

```python
class S3RawMirror:
    """LocalRawLog 이후 비동기로 업로드하는 백업 mirror."""
    
    BUFFER_SIZE = 100
    FLUSH_INTERVAL = 60  # 초
    
    def __init__(self, bucket: str, user_id: str):
        self.bucket = bucket
        self.prefix = f"raw/{user_id}/"
        self.s3 = boto3.client("s3")
        self._buffer: list[RawTurn] = []
        self._lock = threading.Lock()
        self._last_flush = time.time()
    
    def enqueue(self, entry: RawTurn) -> None:
        with self._lock:
            self._buffer.append(entry)
            should_flush = (
                len(self._buffer) >= self.BUFFER_SIZE or
                time.time() - self._last_flush > self.FLUSH_INTERVAL
            )
        if should_flush:
            self._flush()
    
    def _flush(self) -> None:
        with self._lock:
            if not self._buffer:
                return
            buffer, self._buffer = self._buffer, []
            self._last_flush = time.time()
        
        date = datetime.utcnow().strftime("%Y/%m/%d")
        key = f"{self.prefix}{date}/{uuid4()}.jsonl.gz"
        
        bio = io.BytesIO()
        with gzip.GzipFile(fileobj=bio, mode="wb") as gz:
            for item in buffer:
                gz.write((json.dumps({
                    "turn_id": item.turn_id,
                    "timestamp": item.timestamp.isoformat(),
                    "user": item.user,
                    "assistant": item.assistant,
                }, ensure_ascii=False) + "\n").encode())
        bio.seek(0)
        self.s3.upload_fileobj(bio, self.bucket, key)
```

주의:
- S3 mirror는 **백업/아카이브용**이다.
- `turn()` durability ACK는 항상 `LocalRawLog.append()`가 담당한다.
- 따라서 v2.2에서 writable Engram은 로컬 영속 디스크를 전제로 한다.

---

## 14. 충돌 감지 및 히스토리

### 14.1 ConflictDetector

```python
class ConflictDetector:
    """추출된 이벤트에 _changed_from 주석 추가."""
    
    def annotate(
        self, 
        events: list[Event], 
        state: StateCache,
    ) -> list[Event]:
        for event in events:
            if event.type != "entity.update":
                continue
            
            existing = state.get(event.data["id"])
            if not existing:
                continue
            
            changed_from: dict[str, Any] = {}
            for key, new_val in event.data.items():
                if key.startswith("_") or key == "id":
                    continue
                old_val = existing.attributes.get(key)
                if old_val is not None and old_val != new_val:
                    changed_from[key] = old_val
            
            if changed_from:
                event.data["_changed_from"] = changed_from
        
        return events
```

### 14.2 History 조회

```python
def history(
    self, 
    entity_id: str, 
    attr: str | None = None,
) -> list[HistoryEntry]:
    entries = []
    for event in self.store.events_for_entity(entity_id):
        if event.type != "entity.update":
            continue
        changed = event.data.get("_changed_from", {})
        for key, old_val in changed.items():
            if attr and key != attr:
                continue
            new_val = event.data.get(key)
            entries.append(HistoryEntry(
                occurred_at=event.occurred_at,
                attribute=key,
                old_value=old_val,
                new_value=new_val,
                event_id=event.id,
                reason=event.data.get("_reason"),
            ))
    return entries
```

---

## 15. 액션 Writeback

### 15.1 핵심 원칙

1. **Typed action only**: `create_ticket`, `update_crm_field`, `schedule_meeting`처럼 이름과 입력 스키마가 고정된 액션만 1급 객체로 다룬다.
2. **Intent first**: 에이전트가 바로 외부 시스템을 수정하지 않고, 먼저 `ActionIntent`를 남긴다.
3. **Policy gated**: 액션 타입별로 `auto` 또는 `human_required` 정책을 둔다.
4. **Intent state와 execution state를 분리한다**: 의사결정과 실행 시도를 같은 상태 머신에 섞지 않는다.
5. **Idempotency is mandatory**: writeback 액션은 항상 `idempotency_key`를 가져야 한다.
6. **Every external effect gets a handle**: canonical external object와 audit effect를 분리해 기록한다.
7. **Revert if you can, record if you can't**: 되돌릴 수 있는 액션은 compensator를 제공하고, 불가능한 액션은 비가역임을 명시한다.

### 15.2 상태 머신

```text
Intent state:
PROPOSED ──approve_action()──> APPROVED
    │
    ├─ reject_action() ──────> REJECTED
    └─ cancel_action() ──────> CANCELED

Execution attempt state:
APPROVED intent ──execute_action()──> RUNNING
RUNNING ├─ success ──> SUCCEEDED
        ├─ failure ──> FAILED
        └─ revert_action() after success ──> REVERTED
```

설명:
- `ActionIntent`는 "무엇을 하려는지"와 "사람이 승인/거절했는지"를 담는 의사결정 로그다.
- `ActionExecution`은 "실제로 몇 번째 시도가 어떻게 끝났는지"를 담는 attempt 로그다.
- 따라서 하나의 intent에 여러 execution이 달릴 수 있고, retry는 execution만 늘린다.

### 15.3 Policy / Approval

```python
class PolicyEngine:
    def resolve(self, action_type: ActionType, payload: dict[str, Any]) -> str:
        """
        Returns:
            "auto"            -> 즉시 APPROVED
            "human_required"  -> 사람 승인 필요
        """
```

예시 정책:
- `create_ticket`: auto
- `schedule_meeting`: auto 또는 참석자 수/외부 참석자 여부에 따라 human_required
- `update_crm_field`: 중요한 필드(계약 상태, 신용 한도 등)는 human_required

권장 규칙:
- 읽기/추가 성격의 저위험 액션은 auto
- 금액, 고객 상태, 권한, 삭제, 대외 발송은 human_required

### 15.4 Idempotency 계약

모든 writeback intent는 `idempotency_key`를 가진다.

규칙:
- 같은 `(action_type, idempotency_key)`와 같은 canonical payload면 기존 intent를 반환한다.
- 같은 키인데 payload가 다르면 `IdempotencyConflictError`를 낸다.
- `execute_action()`는 latest attempt가 `RUNNING`이면 그 attempt를 그대로 반환한다.
- latest attempt가 `SUCCEEDED`이면 같은 결과를 반환하고 외부 writeback을 다시 하지 않는다.
- latest attempt가 `FAILED`일 때만 새 attempt를 만든다.
- 가능하면 외부 커넥터 호출에도 같은 key를 넘긴다.

### 15.5 External Object 모델

v2.2에서는 외부 참조를 3계층으로 분리한다.

1. `external_objects`
같은 `(system, ref_type, external_id)`의 canonical 외부 객체

2. `external_bindings`
내부 owner, 보통 entity와 canonical object의 연결

3. `action_effects`
특정 execution이 어떤 외부 객체에 어떤 효과를 냈는지 남기는 감사 로그

예:
- `external_object:salesforce/contact/003...`
- `external_binding(entity:user:alice -> external_object:003...)`
- `action_effect(exec_123 -> external_object:003..., effect_type="update")`

이 매핑이 있어야:
- 같은 대상을 다음 액션에서 다시 찾을 수 있고
- 실행 결과를 히스토리로 묶을 수 있고
- 되돌리기 시 외부 대상 ID를 정확히 잡을 수 있다

### 15.6 Compensation / Revert

reversible action은 `compensator`를 등록한다.

예:
- `create_ticket` → 되돌리기: issue close / delete (조직 정책에 따라 다름)
- `schedule_meeting` → 되돌리기: calendar event cancel
- `update_crm_field` → 되돌리기: 이전 값으로 patch

주의:
- 모든 액션이 완전히 되돌려지진 않는다.
- 외부 시스템 정책상 삭제 불가일 수 있다.
- 이런 경우에도 `REVERTED` 대신 `FAILED`와 이유를 남겨 감사 가능해야 한다.

### 15.7 예시 플로우

```python
intent = mem.propose_action(
    "update_crm_field",
    payload={
        "system": "salesforce",
        "record_type": "contact",
        "record_id": "003xx",
        "field": "customer_tier",
        "value": "gold",
    },
    reason="최근 구매 금액 기준 승급",
    target_entities=["user:alice"],
    idempotency_key="crm:alice:tier:gold:v1",
)

intent = mem.approve_action(intent.id, approver="ops:mina")
execution = mem.execute_action(intent.id)

mem.bind_external_object(
    owner_kind="entity",
    owner_id="user:alice",
    system="salesforce",
    ref_type="contact",
    external_id="003xx",
    role="primary_contact",
)

mem.record_action_effect(
    execution_id=execution.id,
    system="salesforce",
    ref_type="contact",
    external_id="003xx",
    effect_type="update",
    payload={"field": "customer_tier", "value": "gold"},
)
```

이 흐름에서 Engram이 보장하는 것:
- 누가 어떤 액션을 제안했는지
- 누가 승인했는지
- 같은 요청이 재시도되어도 중복 writeback이 막히는지
- 언제 어떤 execution attempt가 실행됐는지
- 결과가 성공/실패/되돌림 중 무엇인지
- 어떤 외부 객체에 영향을 줬는지

---

## 16. 멀티테넌시

### 16.1 기본: 파일 격리

```python
class Engram:
    def __init__(self, user_id: str = "default", path: str | None = None, ...):
        if path is None:
            base = Path.home() / ".engram" / "users" / user_id
        else:
            base = Path(path) / user_id
        base.mkdir(parents=True, exist_ok=True)
        
        self.db_path = base / "engram.db"
        self.raw_path = base / "raw.jsonl.gz"
        ...
```

### 16.2 서버 환경: Pool

```python
class EngramPool:
    """서버 프로세스 내 user_id별 Engram 캐시."""
    
    def __init__(self, max_instances: int = 1000, ttl_seconds: int = 3600):
        self._instances: OrderedDict[str, tuple[Engram, float]] = OrderedDict()
        self.max = max_instances
        self.ttl = ttl_seconds
        self._lock = threading.Lock()
    
    def get(self, user_id: str) -> Engram:
        with self._lock:
            now = time.time()
            # TTL eviction
            for uid, (inst, last) in list(self._instances.items()):
                if now - last > self.ttl:
                    inst.close()
                    del self._instances[uid]
            
            if user_id in self._instances:
                inst, _ = self._instances.pop(user_id)
                self._instances[user_id] = (inst, now)  # LRU 재삽입
                return inst
            
            # 새 인스턴스
            if len(self._instances) >= self.max:
                # LRU eviction
                old_uid, (old_inst, _) = self._instances.popitem(last=False)
                old_inst.close()
            
            inst = Engram(user_id=user_id)
            self._instances[user_id] = (inst, now)
            return inst
    
    def close_all(self) -> None:
        with self._lock:
            for inst, _ in self._instances.values():
                inst.close()
            self._instances.clear()
```

### 16.3 FastAPI 예시

```python
from fastapi import FastAPI, Header
from contextlib import asynccontextmanager

pool = EngramPool()

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    pool.close_all()

app = FastAPI(lifespan=lifespan)

@app.post("/chat")
async def chat(message: str, user_id: str = Header(...)):
    mem = pool.get(user_id)
    context = mem.context(query=message, max_tokens=2000)
    response = await llm.generate(system=context, user=message)
    mem.turn(message, response)
    return {"response": response}
```

---

## 17. 동시성 및 라이프사이클

### 17.1 큐 정책

```python
from queue import Queue, Full, Empty

class WriteQueue:
    def __init__(
        self, 
        max_size: int = 10000,
        overflow: Literal["block", "error"] = "block",
    ):
        self._q = Queue(maxsize=max_size)
        self.overflow = overflow
        self._closed = False
        self.rejected = 0
    
    def put(self, item: QueueItem) -> None:
        if self._closed:
            raise EngramError("queue is closing")
        try:
            self._q.put(item, block=(self.overflow == "block"))
        except Full:
            self.rejected += 1
            raise QueueFullError()
    
    def close_for_writes(self) -> None:
        self._closed = True
    
    def stop_worker(self) -> None:
        self._q.put(SENTINEL)
    
    def get(self, timeout: float | None = None):
        return self._q.get(timeout=timeout)
    
    def task_done(self) -> None:
        self._q.task_done()
```

**보장:**
- **같은 user_id 내 순서 보장**: 단일 워커 스레드.
- **프로세스 크래시 시 큐 유실**: Raw Log는 살아있으므로 `reprocess()`로 복구.
- **백프레셔**: `block` 또는 명시적 `QueueFullError`.
- **flush 의미**: 이미 accepted된 항목이 skip 또는 commit까지 끝날 때까지 대기.

### 17.2 graceful shutdown

```python
def close(self) -> None:
    """
    1. 새 write 차단
    2. accepted queue drain 대기
    3. sentinel로 워커 정상 종료
    4. optional S3 mirror flush
    5. DB 커밋 & 클로즈
    """
    self.queue.close_for_writes()
    self.flush(timeout=10.0)
    self.queue.stop_worker()
    self.worker.stop(timeout=5.0)
    if self.raw_mirror:
        self.raw_mirror._flush()
    self.conn.commit()
    self.conn.close()
```

### 17.3 Serverless 주의

**Lambda / Cloud Functions 환경에서는 EngramPool 사용 금지.**
- 각 요청마다 `Engram()` 생성·`close()` (콜드스타트 ~100ms 허용 시)
- writable mode는 로컬 영속 디스크가 있을 때만 권장
- pure serverless write path는 v2.2의 1차 대상이 아님
- Tier 2 파일을 EFS 등 공유 스토리지에 두는 패턴은 **지원 안 함** (SQLite WAL 동시성 문제)

---

## 18. 성능 목표

### 18.1 레이턴시

| 작업 | p50 | p99 | 복잡도 |
|------|-----|-----|--------|
| `turn()` | 1ms | 5ms | O(1) |
| `propose_action()` | 1ms | 5ms | O(1) |
| `approve_action()` | 1ms | 5ms | O(1) |
| `execute_action()` | external I/O bound | depends | O(connector) |
| `context(query=...)` | 20ms | 50ms | O(k log n) |
| `get(entity_id)` | 0.1ms | 1ms | O(1) |
| `get_at(entity_id, t)` | 5ms | 20ms | O(스냅샷이후이벤트) |
| `history()` | 5ms | 20ms | O(엔티티이벤트수) |
| `related(depth=1)` | 0.5ms | 2ms | O(차수) |
| `related(depth=3)` | 5ms | 30ms | O(차수³) |
| `search()` | 20ms | 80ms | O(k log n) |
| `recall()` | 10ms | 40ms | O(depth·차수) |
| `raw_search()` | 50ms | 300ms | O(n) linear scan |
| 콜드 스타트 | 100ms | 500ms | — |

### 18.2 처리량 (단일 인스턴스)

| 시나리오 | 목표 |
|----------|------|
| `turn()` throughput | 1k/sec (local SSD + fsync) |
| 추출 처리 (hybrid) | ~30턴/분 (LLM bound) |
| `context()` throughput | 500/sec |

### 18.3 저장 비용

| 항목 | 크기 |
|------|------|
| Raw Log | ~5MB/유저/년 |
| Events | ~10MB/유저/년 |
| Action log | ~2MB/유저/년 |
| Snapshots | ~1MB/유저 (압축) |
| Vector index | ~2MB/1000 엔티티 (MiniLM) |
| **총** | **~22MB/유저/년** |

---

## 19. 파일 구조

```
engram/
├── pyproject.toml
├── README.md
├── LICENSE                          # MIT
├── CHANGELOG.md
│
├── src/engram/
│   ├── __init__.py                  # from engram import Engram
│   ├── engram.py                    # Engram 클래스 (orchestration)
│   ├── types.py                     # 모든 dataclass
│   ├── errors.py                    # 예외 계층
│   │
│   ├── storage/
│   │   ├── __init__.py
    │   │   ├── store.py                 # EventStore + extraction run lineage
│   │   ├── snapshot.py              # SnapshotManager
│   │   ├── state_cache.py           # StateCache
│   │   ├── relations.py             # RelationIndex
    │   │   ├── raw_log.py               # LocalRawLog + S3RawMirror
    │   │   └── schema.sql               # DDL
│   │
│   ├── extraction/
│   │   ├── __init__.py
│   │   ├── pipeline.py              # ExtractionPipeline
│   │   ├── filter.py                # LocalFilter
│   │   ├── llm_extractor.py         # LLMExtractor + 프롬프트
│   │   ├── conflict.py              # ConflictDetector
│   │   └── worker.py                # BackgroundWorker + WriteQueue
│   │
│   ├── retrieval/
│   │   ├── __init__.py
    │   │   ├── pipeline.py              # event-seeded RetrievalPipeline
│   │   ├── entity_matcher.py
│   │   ├── semantic_index.py        # sqlite-vec wrapper
│   │   ├── entity_index.py          # FTS5 wrapper
│   │   ├── temporal.py              # TemporalFilter
│   │   ├── causal.py                # CausalExpander
│   │   ├── ranker.py                # HybridRanker
│   │   └── embedder.py              # Embedder protocol + 구현체들
│   │
│   ├── context/
│   │   ├── __init__.py
│   │   ├── builder.py               # ContextBuilder
│   │   └── token_counter.py         # tiktoken 래퍼
│   │
│   ├── actions/
│   │   ├── __init__.py
│   │   ├── registry.py              # ActionType registry
    │   │   ├── service.py               # propose/approve/execute/retry/revert
    │   │   ├── policy.py                # PolicyEngine
    │   │   ├── executors.py             # handler adapter
    │   │   └── external.py              # external_objects/bindings/effects
│   │
│   └── pool.py                      # EngramPool
│
├── tests/
│   ├── unit/
│   │   ├── test_store.py
│   │   ├── test_snapshot.py
│   │   ├── test_extractor.py
│   │   ├── test_conflict.py
│   │   ├── test_actions.py
│   │   ├── test_semantic_index.py
│   │   ├── test_ranker.py
│   │   └── test_context_builder.py
│   ├── integration/
│   │   ├── test_write_path.py
│   │   ├── test_writeback.py
│   │   ├── test_read_path.py
│   │   ├── test_reprocess.py
│   │   └── test_multitenant.py
│   └── fixtures/
│       ├── conversations.jsonl
│       └── expected_events.json
│
├── benchmarks/
│   ├── locomo/                      # LOCOMO benchmark harness
│   └── latency.py
│
└── examples/
    ├── basic.py
    ├── chatbot.py
    ├── fastapi_server.py
    ├── custom_weights.py
    └── reprocess_migration.py
```

**코드 규모 추정: ~1,800줄**

| 영역 | 줄 |
|------|-----|
| storage/ | 430 |
| extraction/ | 300 |
| retrieval/ | 430 |
| context/ | 120 |
| actions/ | 260 |
| engram.py + types.py + errors.py + pool.py | 260 |
| **총** | **~1,800** |

---

## 20. 구현 순서

### Week 1: Storage Foundation

**목표:** extraction run + event append → 상태 복원이 되는 최소 구현.

- [ ] `types.py`: 전 dataclass 정의
- [ ] `storage/schema.sql`: DDL 완성
- [ ] `storage/store.py`: EventStore + `extraction_runs` + `superseded_runs`
- [ ] `storage/state_cache.py`: StateCache (apply, dump/restore)
- [ ] `storage/relations.py`: RelationIndex
- [ ] `storage/snapshot.py`: SnapshotManager
- [ ] `storage/raw_log.py`: LocalRawLog
- [ ] Unit tests: append → restore 왕복

**Deliverable:** `mem.append(...)` → `mem.get(...)` 왕복 동작.

### Week 2: Extraction Pipeline

**목표:** RawTurn timestamp를 보존한 채 LLM 추출로 이벤트 생성.

- [ ] `extraction/filter.py`: LocalFilter + 패턴 튜닝
- [ ] `extraction/llm_extractor.py`: 프롬프트 + JSON 파싱 + `occurred_at/recorded_at` 처리
- [ ] `extraction/conflict.py`: ConflictDetector
- [ ] `extraction/worker.py`: BackgroundWorker + WriteQueue (`block` / `error`)
- [ ] `extraction/pipeline.py`: 전체 조립
- [ ] Integration test: `turn()` → extraction_run + 이벤트 생성 검증

**Deliverable:** `mem.turn(...)` → 백그라운드에서 이벤트 생성.

### Week 3: Retrieval (Entity + Semantic)

**목표:** event-seeded retrieval의 최소 패리티.

- [ ] `retrieval/embedder.py`: Local + OpenAI + Custom
- [ ] `retrieval/semantic_index.py`: entity/event sqlite-vec 통합
- [ ] `retrieval/entity_index.py`: FTS5 통합
- [ ] `retrieval/entity_matcher.py`
- [ ] 쓰기 파이프라인 연계: 이벤트 커밋 후 임베딩 인덱싱
- [ ] seed event → entity projection
- [ ] 간단 ranker (event-derived entity + semantic 2축)
- [ ] `context/builder.py`: 토큰 예산 기본 구현
- [ ] Integration test: query → 관련 엔티티

**Deliverable:** `mem.context(query=...)` 기본 동작.

### Week 4: Retrieval (Temporal + Causal) + Context

**목표:** `occurred_at` 기준 4축 완성.

- [ ] `retrieval/temporal.py`: Event.occurred_at 기반 window 필터
- [ ] `retrieval/causal.py`: CausalExpander (BFS)
- [ ] `retrieval/ranker.py`: HybridRanker (4축 + 가중치)
- [ ] `retrieval/pipeline.py`: 전체 조립
- [ ] `context/builder.py`: supporting_event_ids 기반 History 렌더링
- [ ] `search()`, `recall()` API
- [ ] Integration test: 4축 각각 + 합성

**Deliverable:** API 섹션 7의 모든 메서드 동작.

### Week 5: Action Writeback

**목표:** 안전한 operational writeback의 최소 세트 확보.

- [ ] `actions/registry.py`: `ActionType` 등록/조회
- [ ] `actions/policy.py`: auto vs human_required 정책
- [ ] `actions/service.py`: `propose_action()`, `approve_action()`, `reject_action()`, `execute_action()`, `revert_action()`
- [ ] `actions/external.py`: `external_objects`, `external_bindings`, `action_effects`
- [ ] idempotency key canonicalization + dedupe 규칙
- [ ] mock executor 3종: `create_ticket`, `update_crm_field`, `schedule_meeting`
- [ ] Integration test: propose → approve → execute → retry/revert

**Deliverable:** typed action 3종이 intent/execution/external object까지 왕복.

### Week 6: Benchmark + Tuning

**목표:** 숫자 확보.

- [ ] LOCOMO benchmark harness 이식
- [ ] mem0 baseline 측정
- [ ] Engram 측정
- [ ] HybridRanker 가중치 grid search
- [ ] Temporal/Causal 축이 실제로 효과 있는지 검증
- [ ] 레이턴시 프로파일링 → 병목 제거
- [ ] 결과 문서화

**Deliverable:** "Engram vs mem0 on LOCOMO" 표.

### Week 7: 문서 + 배포

- [ ] README (영문 + 한글)
- [ ] API 레퍼런스 (mkdocs)
- [ ] 예제 5개
- [ ] `reprocess()` 마이그레이션 가이드
- [ ] `pyproject.toml` + PyPI 배포
- [ ] GitHub Actions CI (lint, type check, test, benchmark)

---

## 21. 테스트 전략

### 21.1 계층

| 계층 | 도구 | 범위 |
|------|------|------|
| Unit | pytest | 각 클래스 메서드 |
| Integration | pytest + tmp_path | write→read 왕복 |
| Property | hypothesis | EventStore 불변식 |
| Benchmark | pytest-benchmark | 레이턴시 회귀 |
| End-to-End | pytest + real LLM | 실제 LLM 호출 (기본 skip) |

### 21.2 중요 불변식 (property test)

- **Event Store:** `append(e1); append(e2)` 후 `iter()` 결과는 삽입 순서 보존.
- **Snapshot:** `snapshot + replay(이후이벤트) == 현재 상태`.
- **get_at:** `get_at(id, now)` == `get(id)` (모든 id에 대해).
- **Canonical time invariant:** reprocess 전후로 같은 raw turn에서 나온 이벤트의 `occurred_at`은 변하지 않는다.
- **Run lineage invariant:** 같은 raw log를 같은 extractor_version으로 두 번 reprocess해도 active run은 하나만 남는다.
- **Temporal query invariant:** `temporal_window`는 `Event.occurred_at` 기준으로만 결과를 바꾼다.
- **Approval invariant:** `human_required` 액션은 intent가 APPROVED 이전에 execution이 생길 수 없다.
- **Idempotency invariant:** 같은 `(action_type, idempotency_key)` + 같은 payload는 기존 intent를 반환한다.
- **Execution dedupe invariant:** latest execution이 `RUNNING` 또는 `SUCCEEDED`면 `execute_action()`는 새 attempt를 만들지 않는다.
- **Revert invariant:** reversible action은 성공 후 revert 시 `REVERTED` 또는 명시적 `FAILED(reason)` 둘 중 하나를 남긴다.
- **External object invariant:** 같은 `(system, ref_type, external_id)`는 하나의 canonical `external_object`만 가진다.
- **Binding/effect coexistence:** 같은 외부 객체를 entity binding과 여러 action effect가 동시에 참조할 수 있다.
- **Token budget:** `len(context(max_tokens=N)) ≤ N` (tokenizer 기준).

### 21.3 Fixture

```python
# tests/fixtures/conversations.jsonl
# 각 줄: {user, assistant, expected_events}
# LOCOMO에서 추출한 대화 20개 + 합성 엣지케이스 20개
```

### 21.4 LLM 모킹

```python
class MockLLMExtractor(LLMExtractor):
    """정해진 입력 → 정해진 출력. 결정론적 테스트용."""
    def __init__(self, fixtures: dict[tuple[str, str], list[dict[str, Any]]]):
        self.fixtures = fixtures
    
    def extract(self, run_id, turn_id, turn_timestamp, user, assistant, known_entities):
        events = []
        for item in self.fixtures.get((user, assistant), []):
            events.append(Event(
                id=str(uuid4()),
                occurred_at=turn_timestamp,
                recorded_at=datetime.utcnow(),
                type=item["type"],
                data=item["data"],
                extraction_run_id=run_id,
            ))
        return events
```

---

## 22. 벤치마크 계획

### 22.1 LOCOMO 지표

| 메트릭 | OpenAI 메모리 | mem0 | Engram 목표 |
|--------|---------------|------|-------------|
| Single-hop F1 | baseline | +26% | mem0 동등 이상 |
| Multi-hop F1 | baseline | 우위 | mem0 동등 |
| Temporal F1 | ~15% | ~51.5 | **55+ (구조적 우위)** |
| Open-domain J | baseline | 58.1 | 60+ |

### 22.2 레이턴시 벤치

- `turn()` throughput: local SSD + fsync 기준 p99 5ms 이내
- `context()` p99: 50ms 이내 @ 10k 엔티티
- 콜드 스타트: 500ms 이내 @ 1M 이벤트 DB

### 22.3 비용 벤치

- 100턴 대화당 LLM 추출 비용
- 2단계 필터 skip률
- 임베딩 비용 (local vs API)

---

## 23. 알려진 한계

### 23.1 v2.2 범위 외

| 항목 | 상태 | 해결 계획 |
|------|------|-----------|
| Entity resolution (중복 엔티티 통합) | 약함 | v2.3: 임베딩 기반 엔티티 링킹 |
| LLM 추출 비결정성 | 내재적 | reprocess + confidence filter로 완화 |
| 임베딩 모델 교체 시 재계산 | 비용 큼 | 백그라운드 재인덱싱 도구 |
| HybridRanker 가중치 자동 튜닝 | 수동 | v2.3: 온라인 학습 |
| 커넥터별 revert 품질 차이 | 큼 | 액션 타입별 compensator 명시 + 조직 정책으로 제한 |
| 분산 / 멀티 프로세스 쓰기 | 미지원 | 영구 미지원 (임베디드 정체성) |
| 실시간 동기화 | 미지원 | v3: 선택적 CRDT 레이어 |

### 23.2 Moat 관련 정직한 평가

**Engram의 구조적 우위:**
1. 임베디드 (외부 DB 없음) — mem0는 이걸 쉽게 따라오기 어려움 (cloud managed 비즈니스 모델과 상충)
2. Event sourcing → 시간여행 — mem0는 상태 중심 설계라 retrofit 비용 큼
3. reprocess() — 추출 개선을 제품에 내장한 유일한 설계

**Engram의 약점:**
1. 생태계 (통합 수) — mem0 대비 절대 열위
2. 관리형 서비스 없음 — enterprise 고객은 선호 안 함
3. 벡터 품질 — sqlite-vec는 Qdrant 대비 대규모에서 열위 (단일 유저 수준은 OK)

---

## 24. Appendix: 주요 코드 스켈레톤

### 24.1 Engram 조립

```python
# src/engram/engram.py

class Engram:
    def __init__(
        self,
        user_id: str = "default",
        path: str | None = None,
        raw_log: bool = True,
        raw_mirror: Literal["none", "s3"] = "none",
        s3_bucket: str | None = None,
        extraction: Literal["local", "llm", "hybrid"] = "hybrid",
        llm_client: Any = None,
        llm_model: str = "claude-haiku-4-5",
        extractor_version: str = "v1",
        embedding: Literal["local", "openai", "custom"] = "local",
        embedding_model: str = "all-MiniLM-L6-v2",
        embedding_fn: Callable | None = None,
        action_handlers: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] | None = None,
        snapshot_interval: int = 1000,
        queue_max_size: int = 10000,
        queue_overflow: Literal["block", "error"] = "block",
    ):
        # 1. 경로 준비
        base = self._resolve_path(user_id, path)
        
        # 2. DB 연결 (sqlite-vec 로드)
        self.conn = self._open_db(base / "engram.db")
        
        # 3. 스토리지 레이어
        self.store = EventStore(self.conn)
        self.state = StateCache()
        self.relations = RelationIndex()
        self.snapshot_mgr = SnapshotManager(
            self.conn, self.store, 
            event_threshold=snapshot_interval,
        )
        
        # 4. 상태 복원 (최신 스냅샷 + 이후 이벤트 replay)
        self._restore_state()
        
        # 5. Raw log (durable path) + optional S3 mirror
        self.raw_log = LocalRawLog(base / "raw.jsonl.gz") if raw_log else None
        self.raw_mirror = (
            S3RawMirror(s3_bucket, user_id)
            if raw_log and raw_mirror == "s3"
            else None
        )
        
        # 6. Retrieval 레이어
        self.embedder = self._build_embedder(embedding, embedding_model, embedding_fn)
        self.semantic_idx = SemanticIndex(self.conn, self.embedder)
        self.entity_idx = EntityIndex(self.conn)
        self.retrieval = RetrievalPipeline(
            entity_matcher=EntityMatcher(self.entity_idx),
            semantic_idx=self.semantic_idx,
            temporal_filter=TemporalFilter(),
            causal_expander=CausalExpander(self.store),
            ranker=HybridRanker(),
            state_cache=self.state,
            store=self.store,
        )
        self.context_builder = ContextBuilder(
            token_counter=make_token_counter(llm_model),
        )
        
        # 6.5 Action Writeback 레이어
        self.actions = ActionService(
            store=self.store,
            handlers=action_handlers or {},
            policy_engine=PolicyEngine(),
        )

        # 7. Extraction 레이어
        self.queue = WriteQueue(queue_max_size, queue_overflow)
        self.extraction = ExtractionPipeline(
            filter_=LocalFilter(),
            extractor=self._build_extractor(
                extraction,
                llm_client,
                llm_model,
                extractor_version,
            ),
            conflict_detector=ConflictDetector(),
            store=self.store,
            state_cache=self.state,
            relations=self.relations,
            semantic_idx=self.semantic_idx,
            entity_idx=self.entity_idx,
            snapshot_mgr=self.snapshot_mgr,
        )
        self.worker = BackgroundWorker(self.queue, self.extraction)
        self.worker.start()

    def turn(self, user: str, assistant: str) -> None:
        turn_id = str(uuid4())
        entry = (
            self.raw_log.append(turn_id=turn_id, user=user, assistant=assistant)
            if self.raw_log
            else RawTurn(turn_id=turn_id, timestamp=datetime.utcnow(), user=user, assistant=assistant)
        )
        if self.raw_mirror:
            self.raw_mirror.enqueue(entry)
        self.queue.put(QueueItem(
            turn_id=turn_id,
            turn_timestamp=entry.timestamp,
            user=user,
            assistant=assistant,
        ))

    def append(
        self,
        event_type: str,
        data: dict[str, Any],
        *,
        caused_by: str | None = None,
        occurred_at: datetime | None = None,
    ) -> str:
        event = Event(
            id=str(uuid4()),
            occurred_at=occurred_at or datetime.utcnow(),
            recorded_at=datetime.utcnow(),
            type=event_type,
            data=data,
            caused_by=caused_by,
        )
        with self.store.transaction() as tx:
            self.store.append(tx, event)
        return event.id

    def propose_action(self, action_type: str, payload: dict[str, Any], **kwargs) -> ActionIntent:
        return self.actions.propose(action_type, payload, **kwargs)

    def approve_action(self, intent_id: str, approver: str, note: str | None = None) -> ActionIntent:
        return self.actions.approve(intent_id, approver, note)

    def execute_action(self, intent_id: str, *, dry_run: bool = False) -> ActionExecution:
        return self.actions.execute(intent_id, dry_run=dry_run)

    def bind_external_object(self, **kwargs) -> str:
        return self.actions.bind_external_object(**kwargs)

    def record_action_effect(self, **kwargs) -> str:
        return self.actions.record_effect(**kwargs)

    def context(
        self,
        query: str | None = None,
        max_tokens: int = 4000,
        **kwargs,
    ) -> str:
        if query is None:
            # v2.0 호환: 최근성 덤프
            ranked = self._recent_entities_as_candidates(top_k=kwargs.get("top_k", 20))
        else:
            ranked = self.retrieval.run(query, **kwargs)
        
        return self.context_builder.build(
            ranked=ranked,
            state_cache=self.state,
            relations=self.relations,
            store=self.store,
            raw_log=self.raw_log,
            max_tokens=max_tokens,
            include_raw=kwargs.get("include_raw", False),
            raw_turns=kwargs.get("raw_turns", 10),
            include_history=kwargs.get("include_history", True),
        )
    
    # ... 나머지 API 메서드
    
    def close(self) -> None:
        self.queue.close_for_writes()
        self.flush(timeout=10.0)
        self.queue.stop_worker()
        self.worker.stop(timeout=5.0)
        if self.raw_mirror:
            self.raw_mirror._flush()
        self.conn.commit()
        self.conn.close()
```

### 24.2 핵심 데이터 플로우 다이어그램

```
                   ┌──────────┐
                   │   User   │
                   └────┬─────┘
                        │
           ┌────────────┴────────────┐
           ↓                         ↓                        ↓
      turn(u, a)              context(query)          propose_action()
           │                         │                        │
           ↓                         ↓                        ↓
     ┌──────────┐             ┌─────────────┐         ┌─────────────┐
     │ RawLog   │             │ Retrieval   │         │ ActionSvc   │
     │ .append  │             │ .run(query) │         │ .propose    │
     └────┬─────┘             └──────┬──────┘         └──────┬──────┘
          │                          │                        │
          ↓                          ├── seed events          ├── Policy
     ┌──────────┐                    ├── semantic events      ├── intent log
     │ Queue    │                    ├── temporal events      └── execute()
     │ .put     │                    └── causal events               │
     └────┬─────┘                           │                         ↓
          │                                 ↓                   external system
          ↓ (async)                   entity projection               │
     ┌──────────┐                           │                         ↓
     │ Worker   │                           ↓                 external_objects
     └────┬─────┘                     ┌────────────┐          external_bindings
          │                           │ Ranker     │          action_effects
          ↓                           └──────┬─────┘
    ┌───────────┐                            │
    │ Filter    │                            ↓
    └─────┬─────┘                      ┌──────────────┐
          ↓                            │ Context      │
    ┌───────────┐                      │ Builder      │
    │ LLMExtr.  │                      └──────┬───────┘
    └─────┬─────┘                             │
          ↓                                   ↓
    ┌──────────────┐                 "## Current State
    │ ExtractionRun│                  ## Recent Changes ..."
    └─────┬────────┘
          ↓
    ┌───────────┐
    │ Conflict  │
    │ Detector  │
    └─────┬─────┘
          ↓
    ┌──────────────────────────────┐
    │ Commit (tx):                  │
    │  record_run                   │
    │  EventStore.append            │
    │  StateCache.apply             │
    │  RelationIndex.apply          │
    │  EntityIndex.update           │
    └─────┬────────────────────────┘
          ↓
    ┌──────────────┐
    │ SemanticIdx  │ (event + entity)
    └─────┬────────┘
          ↓
    ┌──────────────┐
    │ Snapshot     │
    │ maybe_create │
    └──────────────┘
```

---

## 끝맺음

이 문서는 **Engram v2.2의 개발 완료를 위한 충분한 설계 사양**이다.

구현 시작 전 체크리스트:
- [ ] `sqlite-vec` 의존성 lock
- [ ] LLM 공급자 결정 (Anthropic vs OpenAI)
- [ ] LOCOMO 데이터셋 로컬 준비
- [ ] 프롬프트 v1 확정 (§9.2)
- [ ] 코딩 표준 (black, ruff, mypy strict)
- [ ] CI 파이프라인 셋업

각 주차 Deliverable는 §20 참조.
설계 변경이 필요하면 이 문서를 수정하고 CHANGELOG에 기록.
