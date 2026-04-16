# Meaning-Aware Search 설계 문서 v1

## 1. 왜 이 설계가 필요한가

현재 Engram의 lexical 검색은 `event_search_terms`에 저장된 토큰을 중심으로 후보를 찾는다. 이 방식은 구현과 운영이 단순하고, SQLite 안에서 빠르게 동작하도록 다듬기 쉽다는 장점이 있다.

하지만 복합 표현이 등장하면 문제가 드러난다.

- `Busan-1499`처럼 **붙어서 하나의 의미를 갖는 표현**이 있다.
- 현재 lexical 경로는 이를 `busan`, `1499`처럼 **작은 조각**으로도 본다.
- 그러면 `busan`처럼 흔한 조각이 너무 많은 이벤트를 후보로 끌어오고,
- 검색 의미는 흐려지고,
- semantic 확장과 context build 비용도 함께 커진다.

즉, 현재 병목은 단순히 "SQL이 느리다"가 아니라, 더 본질적으로는 **검색 엔진이 의미 단위가 아니라 토큰 조각을 중심으로 움직인다**는 데 있다.

이 문서의 목표는 Engram을 다음 단계로 옮기는 것이다.

- 의미 판단은 LLM이 한다.
- 저장과 검색은 deterministic engine이 한다.
- `flush("index")`, startup recovery, reprocess, versioning 같은 기존 Engram 계약은 그대로 유지한다.

한 줄 정의:

```text
Meaning-Aware Search =
  write-time/query-cache time meaning analysis by LLM
  + SQLite meaning index
  + deterministic retrieval over protected phrases, aliases, canonical keys, and facets
  + token split is only the fallback
```

---

## 2. 설계 철학

### 2.1 그릇은 구조화되게, 판단은 LLM이

이 설계의 핵심은 다음 두 문장으로 요약된다.

- **LLM은 "무엇이 하나의 의미 단위인가"를 판단한다.**
- **엔진은 그 판단 결과만 저장하고 조회한다.**

중요한 점은, 검색할 때마다 LLM을 다시 부르지 않는다는 것이다.

매 query마다 LLM을 부르면:

- 검색 지연이 다시 커지고,
- 비용이 커지고,
- 결과 재현성이 흔들리고,
- 테스트가 어려워진다.

따라서 비싼 의미 판단은 가급적 **write-time** 또는 **query-plan cache miss** 시점으로 옮겨야 한다.

### 2.2 토큰화는 fallback이어야 한다

토큰 split은 완전히 없앨 수는 없다. 예외와 신규 표현, 분석 실패, analyzer 미설치 환경이 있기 때문이다.

하지만 중심은 바뀌어야 한다.

- 현재: token split 중심, phrase는 보너스
- 목표: phrase/alias/canonical/facet 중심, token split은 fallback

### 2.3 canonical truth는 여전히 events다

이 설계는 검색을 바꾸는 것이지, memory truth를 바꾸는 것이 아니다.

- 원본 대화 truth: `RawTurn`
- 구조화 기억 truth: `events`
- 검색 가속 계층: `vec_events`, `event_search_terms`, 새 meaning index

즉, meaning index는 **파생 계층**이다. 깨지거나 비어 있어도 canonical events만 있으면 재구성 가능해야 한다.

### 2.4 기존 Engram 계약과의 정합성

새 meaning-aware search는 아래 기존 계약을 그대로 따른다.

- `append()`와 `turn()` 모두 같은 search freshness 파이프라인을 탄다.
- `flush("index")`는 search index freshness 전체를 보장한다.
- startup recovery는 missing search index를 다시 맞춘다.
- analyzer version이 바뀌면 re-indexing과 reprocess 경계가 분명해야 한다.
- public API 시그니처는 유지한다.

---

## 3. 현재 구조와의 정합성 분석

현재 코드에서 이미 meaning-aware 설계와 잘 맞는 기반은 많다.

### 3.1 현재 이미 있는 것

- `append()`와 canonical worker는 event write 직후 search term을 inline 저장한다.
- `SemanticIndexer.index_missing()`는 embeddings와 fallback search terms를 backfill한다.
- startup recovery는 projection과 semantic index freshness를 다시 맞춘다.
- `flush("index")`는 search index freshness 엔트리포인트다.
- retrieval는 lexical / semantic / causal 축을 합쳐서 결과를 만든다.

즉, 새 meaning index는 완전히 새로운 레이어가 아니라, **현재 index 계층의 확장판**으로 넣는 것이 가장 자연스럽다.

### 3.2 현재와 달라질 핵심

현재 lexical path는 `event_search_terms` 위에 있다.

새 구조에서는 lexical path가 2층이 된다.

1. meaning units (`protected_phrase`, `alias`, `canonical_key`, `facet`)
2. fallback terms (`event_search_terms`)

따라서 retrieval는 "term first"가 아니라 "meaning units first"로 재구성된다.

---

## 4. 새 개념 모델

### 4.1 의미 단위 종류

이 설계에서 저장 가능한 의미 단위는 닫힌 집합으로 제한한다.

- `protected_phrase`
  - 쪼개면 안 되는 표현
  - 예: `busan-1499`
- `alias`
  - 같은 의미를 가리키는 다른 표기
  - 예: `busan 1499`
- `canonical_key`
  - 표준화된 내부 검색 키
  - 예: `label:busan-1499`
- `facet`
  - 제한된 key/value 속성
  - 예: `role=traveler`, `city=busan`
- `fallback_term`
  - 기존 token split 방식의 보험

핵심은 `unit_kind`를 열린 문자열로 두지 않는 것이다.

### 4.2 LLM이 판단해야 하는 것

LLM은 자유롭게 긴 설명을 생성하는 역할이 아니다. 아래 질문에만 답하는 구조여야 한다.

- 이 표현은 하나의 보호된 phrase인가?
- 별칭이 필요한가?
- canonical key가 필요한가?
- facet으로 뽑아도 되는가?
- confidence는 어느 정도인가?

즉 LLM은 "검색 철학"을 판단하는 것이 아니라, **닫힌 검색 구조 안에 들어갈 값을 채워주는 판정기**다.

---

## 5. 새 데이터 모델

### 5.1 `event_search_units`

이벤트별 의미 단위를 저장한다.

권장 컬럼:

- `event_id TEXT NOT NULL`
- `analyzer_version TEXT NOT NULL`
- `unit_kind TEXT NOT NULL`
- `unit_key TEXT`
- `unit_value TEXT NOT NULL`
- `normalized_value TEXT NOT NULL`
- `confidence REAL`
- `metadata TEXT`
- `PRIMARY KEY(event_id, analyzer_version, unit_kind, unit_key, normalized_value)`

설명:

- `unit_kind`
  - `protected_phrase`, `alias`, `canonical_key`, `facet`, `fallback_term`
- `unit_key`
  - facet일 때만 사용
  - 예: `role`, `city`
- `metadata`
  - 간단한 JSON
  - 필요 시 source span, analyzer note, no_split flag 보조 정보 저장

인덱스:

- `(analyzer_version, unit_kind, normalized_value)`
- `(analyzer_version, unit_kind, unit_key, normalized_value)`
- `(event_id, analyzer_version)`

### 5.2 `meaning_analysis_runs`

이벤트별 meaning analysis 실행 상태를 저장한다.

권장 컬럼:

- `event_id TEXT NOT NULL`
- `analyzer_version TEXT NOT NULL`
- `processed_at TEXT NOT NULL`
- `status TEXT NOT NULL CHECK(status IN ('SUCCEEDED', 'FAILED', 'SKIPPED'))`
- `error TEXT`
- `unit_count INTEGER NOT NULL`
- `PRIMARY KEY(event_id, analyzer_version)`

이 테이블의 목적:

- missing meaning index를 찾기 쉽도록 함
- analyzer version 변경 시 재분석 경계 제공
- 실패를 영구적으로 숨기지 않고 기록

### 5.3 `query_meaning_cache`

query의 의미 계획을 캐시한다.

권장 컬럼:

- `normalized_query TEXT NOT NULL`
- `analyzer_version TEXT NOT NULL`
- `payload TEXT NOT NULL`
- `cached_at TEXT NOT NULL`
- `PRIMARY KEY(normalized_query, analyzer_version)`

이 테이블은 query-time LLM 호출을 예외적인 miss path로 제한하기 위한 장치다.

---

## 6. 새 내부 타입

### 6.1 `MeaningUnit`

```python
@dataclass(slots=True, frozen=True)
class MeaningUnit:
    kind: Literal["protected_phrase", "alias", "canonical_key", "facet", "fallback_term"]
    value: str
    normalized_value: str
    key: str | None = None
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

### 6.2 `MeaningAnalysis`

```python
@dataclass(slots=True, frozen=True)
class MeaningAnalysis:
    units: list[MeaningUnit]
```

### 6.3 `QueryMeaningPlan`

```python
@dataclass(slots=True, frozen=True)
class QueryMeaningPlan:
    units: list[MeaningUnit]
    fallback_terms: list[str]
    planner_confidence: float | None = None
```

### 6.4 Protocol

```python
class MeaningAnalyzer(Protocol):
    version: str
    def analyze_event(self, event: Event) -> MeaningAnalysis: ...
    def plan_query(self, query: str) -> QueryMeaningPlan: ...
```

초기 구현에서는 `NullMeaningAnalyzer`와 `OpenAIMeaningAnalyzer`를 둔다.

---

## 7. 초기 analyzer 전략

### 7.1 `NullMeaningAnalyzer`

기본 구현은 LLM 없이도 항상 동작해야 한다.

전략:

- `protected_phrase`, `alias`, `canonical_key`, `facet`는 만들지 않음
- `fallback_term`만 현재 `event_search_terms` 규칙에서 생성

즉, analyzer가 없는 환경에서도 현재 동작과 의미를 최대한 유지한다.

### 7.2 `OpenAIMeaningAnalyzer`

역할:

- 이벤트 텍스트를 보고 의미 단위를 추출
- query를 보고 의미 계획을 추출

출력은 strict JSON으로 제한한다.

event analysis 응답 예시:

```json
{
  "units": [
    {"kind": "protected_phrase", "value": "Busan-1499", "confidence": 0.98},
    {"kind": "alias", "value": "Busan 1499", "confidence": 0.82},
    {"kind": "canonical_key", "value": "label:busan-1499", "confidence": 0.91},
    {"kind": "facet", "key": "role", "value": "traveler", "confidence": 0.84}
  ]
}
```

query planning 응답 예시:

```json
{
  "units": [
    {"kind": "protected_phrase", "value": "Busan-1499", "confidence": 0.97},
    {"kind": "facet", "key": "role", "value": "traveler", "confidence": 0.78}
  ],
  "fallback_terms": ["busan-1499", "traveler"],
  "planner_confidence": 0.92
}
```

제약:

- dialogue text / event text는 데이터일 뿐, 지시가 아님
- schema 밖의 key는 거부
- analyzer version은 provider + model + base_url identity를 반영

---

## 8. 인덱싱 경로

### 8.1 현재 Engram과 맞는 넣는 위치

meaning indexing은 canonical write path 안이 아니라 **index path**에 둔다.

즉:

- canonical event write
- dirty mark
- projection rebuild
- snapshot save
- index freshness
  - embeddings
  - fallback terms
  - meaning units

이 구조가 현재 `SemanticIndexer.index_missing()` 철학과 가장 잘 맞는다.

### 8.2 `MeaningIndexer.index_missing()`

초기 형태:

1. `events_missing_meaning_units(analyzer_version)` 조회
2. 각 event에 대해 `MeaningAnalyzer.analyze_event(event)` 수행
3. transaction 안에서:
   - 기존 analyzer_version의 old units replace
   - `event_search_units` insert
   - `meaning_analysis_runs` 기록

실패 시:

- `meaning_analysis_runs.status = FAILED`
- 기존 canonical truth는 유지
- fallback search는 계속 동작

### 8.3 `flush("index")` 의미

이제 `flush("index")`는 아래 모두를 보장해야 한다.

- `vec_events`
- `event_search_terms`
- `event_search_units`
- `meaning_analysis_runs`
- optional query cache hygiene

즉, 지금 주석/문서의 "search index freshness 전체" 의미를 그대로 확장한다.

### 8.4 startup recovery

startup recovery는 projection + semantic index만이 아니라 meaning index도 맞춰야 한다.

즉:

1. snapshot/projection 복구
2. `semantic_indexer.index_missing()`
3. `meaning_indexer.index_missing()`
4. missing extraction run queue catch-up

---

## 9. Retrieval 알고리즘

### 9.1 핵심 원칙

검색 순서는 다음과 같이 바뀐다.

1. query meaning plan 생성
2. meaning-unit candidate lookup
3. deterministic lexical score
4. semantic rerank
5. causal expansion
6. fallback term only if needed

즉 "token split first"가 아니다.

### 9.2 Candidate generation 우선순위

우선순위:

1. `protected_phrase` exact match
2. `canonical_key` match
3. `alias` match
4. `facet` conjunction
5. `fallback_term`

원칙:

- 앞 단계에서 충분한 고품질 후보가 있으면, 뒤 단계는 축소 또는 생략 가능
- `fallback_term`은 후보 보강용이지 중심이 아니다

### 9.3 점수 함수

권장 스코어:

```text
score = Σ(unit_kind_weight × idf × confidence × field_weight)
```

권장 상대 가중치:

- `protected_phrase` 가장 큼
- `canonical_key`, `alias` 중간
- `facet` 보조
- `fallback_term` 가장 작음

이유:

- `busan` 같은 흔한 term은 정보량이 작다
- `busan-1499` 같은 희귀 phrase는 정보량이 크다

### 9.4 semantic expansion gating

현재 retrieval는 lexical-hit entity 전체에 대해 semantic candidate를 넓히는 경향이 있다.

meaning-aware path에서는 semantic expansion도 제한해야 한다.

권장 규칙:

- `protected_phrase`, `canonical_key`, 강한 `alias` hit가 있는 entity만 semantic expansion
- `fallback_term`만 맞은 넓은 후보는 semantic expansion 제한

이 규칙은 성능과 의미를 동시에 지킨다.

---

## 10. 구현 단계

### Phase 1. Meaning index foundation

범위:

- 새 테이블 3개
- `MeaningUnit`, `MeaningAnalysis`, `QueryMeaningPlan`
- `MeaningAnalyzer`, `NullMeaningAnalyzer`
- `MeaningIndexer.index_missing()`
- `flush("index")`, startup recovery 연결

의도:

- 검색 의미는 아직 바꾸지 않음
- meaning index가 versioning/recovery/backfill 안에서 제대로 돈다는 기반 먼저 확보

### Phase 2. Retrieval overlay

범위:

- `QueryMeaningPlanner` + cache
- retrieval에서 meaning unit candidate lookup 추가
- fallback term path는 유지

의도:

- 검색 정확도 향상
- common-token 폭발 억제

### Phase 3. Scoring and pruning

범위:

- df/idf 통계
- semantic expansion gating
- exact phrase 우선 정렬

의도:

- long-running agent에서 수만 건까지 의미-정확도와 지연을 같이 잡기

---

## 11. 테스트 계획

### 11.1 Phase 1

- analyzer 미설치 시 현재 lexical fallback 동작 유지
- `flush("index")`가 meaning units도 backfill
- startup recovery가 missing meaning units도 복구
- analyzer version 변경 시 current version 기준으로만 조회

### 11.2 Phase 2

- `Busan-1499 traveler` 검색 시 `busan` 공통 후보보다 exact phrase 쪽이 우선
- alias 검색도 동일 entity를 찾음
- query planner cache hit/miss 경로 검증

### 11.3 Phase 3

- protected phrase hit가 있는 경우 fallback-only 후보가 과도하게 위로 오지 않음
- semantic expansion이 넓은 fallback-only 후보를 무의미하게 키우지 않음
- 기존 search/context 의미 회귀가 허용 범위 안인지 평가

---

## 12. 현재 코드베이스와의 파일 매핑

권장 신규 파일:

- `src/engram/meaning_index.py`
- `src/engram/openai_meaning_analyzer.py`

권장 수정 파일:

- `src/engram/types.py`
- `src/engram/storage/schema.sql`
- `src/engram/storage/store.py`
- `src/engram/semantic_index.py`
- `src/engram/recovery.py`
- `src/engram/retrieval.py`
- `src/engram/__init__.py`

선택적 후속:

- `scripts/benchmark_search_latency.py`
- 새 `meaning-aware` benchmark script

---

## 13. 최종 정리

이 설계는 "더 영리한 토큰화"를 목표로 하지 않는다.

이 설계의 진짜 목표는:

- 사람이 하나의 의미로 보는 것을 LLM이 판정하고,
- 그 결과를 Engram이 구조화해서 기억하고,
- 검색은 그 구조화된 의미를 deterministic하게 재사용하는 것이다.

즉 다음 Engram은:

```text
memory truth = events
search truth = analyzer-versioned meaning units + embeddings + fallback terms
meaning judgment = LLM
retrieval execution = deterministic engine
```

이 구조를 지키면:

- 토큰 split의 common-token 폭발을 줄일 수 있고,
- 검색 의미를 더 사람답게 유지할 수 있으며,
- 현재 Engram의 `flush("index")`, startup recovery, reprocess, versioning 철학과도 정확히 정합성을 유지할 수 있다.
