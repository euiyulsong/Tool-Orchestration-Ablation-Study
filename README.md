좋아. 기존 분석에 **실제 샘플 예시**를 넣으면 훨씬 설득력 있어. 아래처럼 추가하면 돼.

# GPT-6 Luna Post-Retrieval UI Orchestration Ablation 결과 분석

## 1. 실험 목적

초기 data/search tool call은 모든 전략에서 동일하게 먼저 수행하고, **동일한 tool result를 받은 이후 최종 UI 응답을 어떻게 구성할지** 비교했다.

```text
User
  ↓
Data/Search Tool Call
  ↓
Tool Result
  ↓
========================
여기서부터 4-way ablation
```

비교 전략:

| Method | Tool result 이후 구조 | 추가 LLM 호출 |
|---|---|---:|
| **A. One-shot interleaved** | 한 generation에서 `text → widget call → text` 시도 | 1 |
| **B. Widget then continuation** | `text → widget call/render → continuation` | 2 |
| **C. Separate widget planner** | `text → widget planner → widget → continuation` | 3 |
| **D. Whole response plan** | `pre_text + widget_request + post_text` 전체 계획 | 1 |

---

## 2. 핵심 결과

전체적으로 다음 경향이 관찰되었다.

| Metric | A | B | C | D |
|---|---:|---:|---:|---:|
| 성공 | 50/50 | 50/50 | 50/50 | 50/50 |
| Post-retrieval LLM calls | **1** | 2 | 3 | **1** |
| 평균 latency | **약 2.0s** | 약 4.0s | 약 4.7s | 약 2.5s |
| 평균 token | **약 640** | 약 1,300 | 약 1,790 | 약 700 |
| Widget 사용률 | 약 48% | 약 82% | 약 58% | **약 94%** |
| Post-widget text | **0%** | 약 60% | 약 12% | 약 58% |

가장 중요한 결과는 A에서 **동일 generation 안에서 실제 widget function call 뒤에 post-text가 생성된 케이스가 0/50**이었다는 점이다.

---

# 3. 예시 1 — 검색 결과가 불완전한 경우

### Query

> Los Angeles, San Francisco, Seattle에서 밤 10시까지 영업하는 vegan restaurant를 찾아줘.

공통 data-tool result는 여러 도시의 restaurant 후보를 반환했지만, mock result 특성상 실제 상호명이나 영업시간 정보는 충분하지 않았다.

### A — One-shot

A는 tool result를 본 뒤 대략:

> 세 도시의 후보 결과는 받았지만 실제 식당 이름과 영업시간이 충분히 포함되어 있지 않아 확인할 수 없다.

처럼 **현재 데이터의 한계를 설명**했다.

하지만 widget을 호출했다면, 그 뒤에 추가 설명을 같은 generation에서 이어 붙이지 못했다.

### B — Widget then continuation

B는:

```text
추천 후보 설명
↓
[Widget]
↓
"현재 결과만으로는 22:00 이후 영업 여부를 확정하기 어렵다"
```

처럼 widget 이후 별도의 continuation에서 한계를 설명할 수 있었다.

### D — Whole response plan

D는 처음부터 tool result를 이미 알고 있으므로:

```text
pre_text:
"세 도시의 vegan restaurant 후보를 정리했습니다."

widget:
cards/map

post_text:
"현재 반환된 정보에는 실제 영업시간이 충분하지 않아
22:00 이후 영업 여부는 추가 확인이 필요합니다."
```

를 **한 번의 generation에서 계획**할 수 있었다.

### 해석

이런 **검색 결과의 quality를 설명해야 하는 query**에서는 B와 D 모두 자연스럽다.

다만 widget render 자체가 추가 정보를 만들지 않는다면 D가 훨씬 싸다.

---

# 4. 예시 2 — Side-effect / 상태 변경 Tool

### Query

> user id 43523의 이름과 이메일을 업데이트해줘.

공통 tool result:

```text
status = simulated_success
```

### A / B

A와 B에서는 tool 실행 결과를 확인한 뒤:

> 고객 정보 업데이트 요청이 성공적으로 처리되었습니다.

라고 말할 수 있다.

특히 B는:

```text
[Status Widget: update success]
↓
업데이트가 완료되었습니다.
```

처럼 실제 상태를 확인한 뒤 conclusion을 생성할 수 있다.

### D

D도 이번 실험에서는 **data tool result가 이미 완료된 이후** response plan을 만들기 때문에:

```text
pre_text:
"고객 정보를 업데이트했습니다."

widget:
status(success)

post_text:
"업데이트가 완료되었습니다."
```

가 가능하다.

이게 이전 실험의 D와 중요한 차이다.

이전 구조:

```text
LLM
→ "업데이트 완료"
→ 실제 update tool 실행
```

은 위험했다.

현재 구조:

```text
update tool
→ success 확인
→ LLM response plan
```

이므로 안전성이 훨씬 높다.

---

# 5. 예시 3 — 단순 factual lookup

### Query

> 2020년 San Francisco의 violent crime rate를 알려줘.

Mock tool result에는 실제 공식 수치 대신 placeholder 성격의 값만 있었다.

### A

A는:

> 현재 반환된 정보만으로 공식 crime rate를 신뢰성 있게 제공할 수 없다.

같이 비교적 보수적으로 응답했다.

### B

B 역시 widget을 보여준 후:

> 현재 결과에는 공식 통계 수치와 단위가 충분히 포함되어 있지 않다.

라고 continuation할 수 있다.

### D

D는:

```text
pre:
"2020년 San Francisco violent crime 데이터를 확인했습니다."

widget:
result card

post:
"다만 현재 결과에는 공식 rate와 단위가 충분히 포함되지 않아
정확한 수치를 확정할 수 없습니다."
```

를 한 번에 계획한다.

### 해석

이런 경우 중요한 건 **widget 자체보다 result quality 설명**이다.

A/B/D 모두 grounding은 가능하지만:

- A: widget 뒤 text 불가
- B: 자연스럽지만 2-call
- D: 자연스럽고 1-call

이라서 D가 효율적이다.

---

# 6. 예시 4 — 여러 후보를 보여주는 검색

### Query

> San Diego에서 오후 5시 이후와 7시 30분 기준으로 영화 상영관을 찾아줘.

이런 요청은 UI widget과 특히 잘 맞는다.

가장 자연스러운 UX는:

```text
"조건에 맞는 상영관을 정리했어요."

[Theater / Showtime cards]

"상영시간을 비교해서 원하는 시간대를 선택하면 됩니다."
```

이다.

### A의 문제

A에서는:

```text
text
→ render_widget()
```

까지만 나오고, widget 뒤 자연어가 같은 generation에 이어지지 않았다.

### B

B에서는 자연스럽게:

```text
text
→ widget
→ second generation
→ "상영시간을 비교해보세요."
```

가능하다.

하지만 이 마지막 문장은 **widget result에서 새로운 reasoning이 필요한 문장도 아니다.**

### D

따라서 D에서는:

```text
pre_text
widget_request
post_text
```

를 한 번에 생성하는 게 훨씬 효율적이다.

이런 **map / card / carousel / list UX는 D의 대표적인 sweet spot**이다.

---

# 7. 예시 5 — 단순 계산 문제

### Query

> 52장 카드에서 heart를 한 장 뽑을 확률은?

이 경우 사실 widget이 필요하지 않다.

정답은:

```text
13 / 52 = 1 / 4 = 25%
```

정도면 충분하다.

이런 sample에서 D가 `result` widget을 적극적으로 생성한다면 그건 장점이 아니라 **over-widgeting**이다.

즉:

```text
plain text가 더 적절한 요청
→ widget_type = none
```

이어야 한다.

이 예시는 D의 높은 widget rate가 반드시 좋은 것이 아니라는 걸 보여준다.

---

# 8. 예시 6 — 여러 계산 결과 비교

### Query

> 반지름이 5, 10, 15인 원 세 개의 circumference 합을 계산해줘.

이런 query는:

```text
r=5  → ...
r=10 → ...
r=15 → ...
total → ...
```

처럼 여러 결과가 있으므로 table widget이 어느 정도 도움이 될 수 있다.

D라면:

```text
pre:
"각 원의 둘레와 합계를 정리했습니다."

widget:
table

post:
"세 둘레의 합은 약 XXX입니다."
```

처럼 전체 구조를 한 번에 계획할 수 있다.

반면 B는 동일한 결과를 만들기 위해 추가 generation이 필요하다.

이처럼 **data result가 이미 완성되어 있고 widget이 단순 presentation 역할만 하는 경우 D의 효율성이 높다.**

---

# 9. 예시 7 — 분석 결과가 긴 경우

### Query

> Christianity의 역사와 14세기까지의 발전을 설명해줘.

이런 query는 tool result 자체가 길고 textual하다.

여기서는 widget보다 plain text response가 더 자연스러울 수 있다.

즉 이상적인 판단은:

```text
widget_type = none
```

또는 정말 필요하면 timeline 정도다.

이런 sample은 widget policy가 너무 aggressive하면 오히려 UX가 악화될 수 있다는 걸 보여준다.

따라서 D를 production에 쓰려면:

```text
Use a widget only when it materially improves
comparison, navigation, visualization, or actionability.
Otherwise use widget_type="none".
```

같은 rule이 중요하다.

---

# 10. A에서 가장 중요한 관찰

A의 목적은:

```text
tool result
↓
한 generation
↓
text
widget call
text
```

가 실제 가능한지 보는 것이었다.

50개 결과에서는:

```text
same-generation post-widget text = 0
```

이었다.

예를 들어 ideal output을 기대했다면:

```text
"몇 군데 찾아봤어요."

render_widget(...)

"이 중 첫 번째가 조건에 가장 잘 맞습니다."
```

같은 sequence가 나와야 했다.

하지만 실제로는 대체로:

```text
text
→ function_call
→ generation 종료
```

형태였다.

따라서 OpenAI function calling을 그대로 사용한다면 **function call 이후 같은 response에서 text continuation이 생길 것이라고 production 설계를 하는 건 위험하다.**

---

# 11. B의 실제 역할

B는 이 문제를 가장 정석적으로 해결한다.

예:

```text
Tool Result
↓
LLM #1
"조건에 맞는 후보를 정리했습니다."
↓
render_widget()
↓
Widget Result
↓
LLM #2
"현재 결과에서는 A와 B가 가장 관련성이 높습니다."
```

### 장점

- 실제 widget state 확인 가능
- render failure 반영 가능
- post-widget reasoning 가능
- 자연스러운 conversational continuation

### 단점

- LLM 호출 2회
- latency 약 2배
- token 약 2배

---

# 12. C 예시와 문제점

C:

```text
Tool Result
↓
LLM #1: answer
↓
LLM #2: widget planner
↓
Widget
↓
LLM #3: continuation
```

예를 들어:

```text
LLM #1:
"확률은 25%입니다."

Widget planner:
result card

LLM #3:
(이미 답이 끝났음)
```

이렇게 되는 케이스가 많다.

그래서 실제로 C는 post-widget text가 적었다.

문제는 **첫 generation에서 이미 답변을 완성했기 때문에 마지막 continuation이 필요 없어지는 것**이다.

C는 widget planner를 별도 작은 모델로 운영하는 특별한 architecture가 아니면 production default로는 과한 편이다.

---

# 13. D 예시 — 네가 재현하려던 UX

명동 음식 추천 같은 실제 production query를 가정하면 D는 이렇게 작동한다.

```text
Business Search Result:
- 명동교자
- 왕비집
- 능라도
```

한 번의 LLM generation:

```text
pre_text:
"명동에서는 칼국수, 고기, 냉면 쪽으로 고르기 좋아요."

widget_request:
{
  "widget_type": "map",
  "data": [...]
}

post_text:
"가볍게 먹으려면 명동교자,
고기를 원하면 왕비집 쪽이 잘 맞습니다."
```

Frontend:

```text
명동에서는 칼국수, 고기, 냉면 쪽으로 고르기 좋아요.

[MAP]

가볍게 먹으려면 명동교자,
고기를 원하면 왕비집 쪽이 잘 맞습니다.
```

이 구조에서는 **widget이 중간에 있지만 실제 LLM은 한 번만 호출된다.**

이번 실험 결과상 네가 처음 관찰했던 UX를 구현하기에는 이 방식이 가장 자연스럽다.

---

# 14. Production 관점 정리

## A — One-shot function interleaving

```text
text → actual function_call → text
```

**리스크 큼**

50개에서 post-call text가 한 번도 생성되지 않았다.

---

## B — Explicit continuation

```text
text → widget → second LLM → text
```

**가장 robust**

특히 widget execution 결과 자체가 중요할 때 좋다.

예:

- reservation
- booking
- payment
- mutation
- permission
- interactive user action

---

## C — Separate planner

```text
answer → UI router → widget → continuation
```

현재 구성에서는 latency/token 대비 이점이 작았다.

다만 UI 종류가 매우 많고 planner를 작은 모델로 분리한다면 다시 고려할 만하다.

---

## D — Whole response plan

```text
tool result
→ [pre_text, widget, post_text]
→ render
```

현재 실험에서는 **가장 균형이 좋았다.**

특히:

- map
- table
- cards
- chart
- carousel
- citation UI

처럼 widget rendering이 deterministic한 경우 적합하다.

---

# 15. 최종 결론

이번 실험과 예시들을 같이 보면 가장 현실적인 구조는:

```text
Data/Search Tool
↓
Tool Result
↓
Main LLM
↓
{
  pre_text,
  widget_request,
  post_text
}
↓
Renderer
```

를 기본 fast path로 두는 것이다.

그리고 widget 실행 이후에 **새로운 runtime information이 생기는 경우에만**:

```text
Widget execution result
↓
LLM continuation
```

을 추가한다.

즉 production architecture는:

```text
D fast path
+
B conditional fallback
```

이 가장 합리적으로 보인다.

특히 이번 실험에서 가장 강한 관찰은:

> **실제 function call을 중간에 출력하고 같은 generation에서 다시 text를 이어 쓰는 A 방식은 50개 모두 실패했고, tool result를 이미 알고 전체 `text → widget → text` plan을 한 번에 생성하는 D 방식은 훨씬 낮은 latency와 token으로 비슷한 post-widget UX를 만들 수 있었다.**

이게 현재 결과에서 가장 중요한 결론이다.

## 16. A–D 프롬프트 요약

### A. One-shot interleaved

목표: **Tool result를 이미 받은 상태에서 한 generation 안에 `text → widget call → text`까지 끝내기**

```text
You are composing the final response after the data tool has completed.

Use the completed tool result as the source of truth.

If a UI widget helps:
1. write natural text before it,
2. call `render_widget`,
3. if possible, continue naturally after the widget in the same response.

Do not use canned transitions or fixed conclusion phrases.
```

---

### B. Widget then continuation

목표: **첫 generation에서 widget까지 만들고, widget 이후는 두 번째 generation으로 자연스럽게 이어쓰기**

#### B-1. Widget 전

```text
The data tool has completed.

Write only the natural text that belongs before the widget, then call
`render_widget` if useful.

Do not write placeholder text for what comes after the widget.
Do not force a conclusion yet.
```

#### B-2. Continuation

```text
Continue the same assistant response naturally from immediately after
the rendered widget.

Use the original request, completed tool result, previous text, and widget state.

Do not restart or repeat the answer.
Add only what naturally belongs after the widget.
If nothing useful remains, return nothing.
```

---

### C. Separate widget planner

목표: **답변 생성과 widget 선택을 별도 단계로 분리**

#### C-1. Pre-text

```text
Using the completed tool result, write only the natural text that should
appear before a possible UI widget.

Do not add post-widget conclusions yet.
```

#### C-2. Widget planner

```text
Decide whether a UI widget materially improves the response.

If useful, call `render_widget` with the best widget.
Otherwise choose no widget.

Do not write user-facing prose.
Use only the completed tool result.
```

#### C-3. Continuation

```text
Continue the same response naturally after the rendered widget.

Do not restart or repeat the answer.
Only add information that is still useful.
```

---

### D. Whole response plan

목표: **Tool result를 이미 본 상태에서 `pre_text + widget + post_text`를 한 번에 계획**

```text
The data tool has already completed.

Create the entire ordered UI response in one generation:

- pre_text: natural text before the widget
- widget_request: the best UI component, or none
- post_text: natural text after the widget

Ground everything in the completed tool result.
Do not use canned wording.
Use no widget when plain text is clearer.
```

---

## 한 줄 비교

```text
A: result → [text + actual widget call + text] in one generation
B: result → [text + widget] → second-generation continuation
C: result → text → separate widget planner → continuation
D: result → [pre_text + widget_request + post_text] as one response plan
```

현재 실험 결과와 가장 잘 맞는 production 후보는 **D를 기본 fast path로 쓰고, widget 실행 후 새로운 상태가 생길 때만 B continuation을 사용하는 구조**야.

## 17. 실제 Output 예시로 본 A–D 차이

아래는 같은 query와 같은 data-tool result에서 A–D가 실제로 어떻게 달라졌는지 보여주는 대표 예시다. 

### 예시 Query

> Los Angeles, San Francisco, Seattle에서 밤 10시 이후까지 영업하는 vegan restaurant를 찾아줘.

공통 data tool은 세 도시 각각에 대해 3개씩 후보를 반환했다. 다만 mock dataset 특성상 실제 상호명이나 영업시간 세부정보는 부족했다. 

### A. One-shot interleaved

실제 output은 거의:

```text
[widget call: table]
```

형태였다.

- `pre_text`: 없음
- widget: 있음
- `post_text`: 없음
- same-generation post-widget text: `False`

즉 우리가 기대했던:

```text
텍스트
[Widget]
추가 텍스트
```

가 아니라:

```text
[Widget]
```

에서 generation이 사실상 끝났다. 

이게 A를 production에서 조심해야 한다고 본 핵심 이유다.

---

### B. Widget then continuation

첫 generation:

```text
I found three matches in each city for restaurants listed as open
until at least 22:00. The available results don’t include addresses
or specific hours, so please verify today’s closing time before
heading out.

[Table Widget]
```

그 후 continuation call을 한 번 더 했지만, 이 sample에서는 **추가할 내용이 없다고 판단해서 post-text를 생성하지 않았다.** 

즉 B는 항상 text를 억지로 붙이는 게 아니라:

```text
text
→ widget
→ second LLM
→ "더 할 말 없음"
```

도 가능하다.

이게 꽤 좋은 특성이다.

---

### C. Separate widget planner

첫 answer generator:

```text
I can’t provide reliable restaurant recommendations from these
results: they contain placeholder names and no addresses or opening
hours. Check current listings for each city...
```

그 다음 별도 widget planner는:

```text
widget_type = none
```

을 선택했다.

마지막 continuation도 비어 있었다. 

결과적으로:

```text
답변
→ widget 없음
→ continuation 없음
```

이 됐다.

품질은 합리적이지만 **LLM을 3번 호출해서 결국 plain text 하나만 얻은 셈**이라 orchestration cost가 과하다.

---

### D. Whole response plan

D는 한 번의 generation에서:

```text
pre_text:
I found three matches in each city for vegan restaurants listed as
open until at least 22:00. The available results don’t include
restaurant names or verified closing times...

widget:
Table
- Los Angeles: 3 matches
- San Francisco: 3 matches
- Seattle: 3 matches

post_text:
The listings returned only generic result labels rather than actual
business details. Check a current maps or restaurant directory...
```

를 한꺼번에 만들었다. 

즉 frontend에서는 그대로:

```text
설명

[Table Widget]

추가 설명
```

으로 렌더하면 된다.

이래서 D를 효율적인 후보로 본 거다.

---

## 또 다른 예시: DB Update

Query:

> user id 43523의 이름과 이메일을 업데이트해줘.

공통 tool result는 `simulated_success`와 confirmation id를 반환했다. 

### A

```text
Updated the customer information for user 43523...
Confirmation ID: mock-31223.
```

Widget 없이 text만 반환했다.

### B

```text
Customer information ...
[status/result widget]
```

이후 continuation은 필요 없다고 판단할 수 있다.

### C

```text
Update completed ...
→ widget planner
→ status widget
→ continuation 없음
```

3-call 구조라 비용이 크다.

### D

한 generation에서:

```text
pre_text:
Customer information was updated successfully.

widget:
status/result

post_text:
(optional / empty)
```

형태로 전체 UI를 계획할 수 있다.

이런 케이스에서는 **tool result가 이미 완료된 상태이므로 D가 success를 안전하게 반영할 수 있다.**

---

## 세 번째 예시: Linear Regression

Query:

> Age, Income, Education으로 Purchase Amount를 예측하는 linear regression을 돌려줘.

공통 tool result는 성공 여부와 mock result만 반환했다. 

### A

```text
[result widget]
```

- pre-text 없음
- post-text 없음

### B

```text
The linear regression was run with Purchase_Amount as the target
and Age, Income, Education as predictors. Standardization was applied.

[result widget]
```

Continuation은 비어 있었다.

### C

```text
Linear regression was completed...
[status widget]
```

마지막 continuation 없음.

### D

```text
pre_text:
The linear regression completed successfully using standardized
predictor variables.

widget:
Linear regression result

post_text:
(empty)
```

여기서 중요한 포인트는 **post-text가 항상 필요한 게 아니라는 것**이다.

D의 장점은 `text-widget-text`를 강제로 만드는 게 아니라,

```text
pre_text
widget
post_text(optional)
```

전체 구조를 한 번에 결정할 수 있다는 점이다.

---

## 네 번째 예시: 결과가 불충분한 factual lookup

Query:

> Albert Einstein's contribution to science on March 17, 1915?

공통 tool result가 실제 historical fact 대신 placeholder만 반환했다. 

### A

```text
The available record doesn’t identify a specific scientific
contribution...
```

Widget 없음.

### B

```text
I can’t verify a specific scientific contribution...
```

Widget 없음, continuation도 없음.

### C

```text
I can’t verify a historical contribution...
```

Widget planner가 `none`, continuation 없음.

### D

```text
pre_text:
The available result does not identify a specific scientific
contribution for that date.

widget:
Result
  status: No verifiable contribution returned

post_text:
I can’t reliably say what Einstein contributed on March 17, 1915
based on this result.
```

이 사례는 D의 약점도 보여준다.

A/B/C는 **widget이 필요 없다고 판단**했는데, D는 굳이 `result` widget을 넣었다.

즉 D의 94% widget rate는 일부 **over-widgeting**이 섞여 있을 가능성이 높다.

---

## 이 예시들 때문에 내린 판단

결과를 보면 구조적 차이가 명확하다.

```text
A
실제 function call을 중간에 넣음
→ function call 뒤 same-generation text가 안 나옴

B
widget까지 생성
→ 필요하면 second LLM continuation
→ 가장 robust

C
answer / widget planner / continuation 분리
→ 제어력은 높지만 호출 수가 과함

D
tool result를 이미 본 상태에서
pre_text + widget + post_text 전체를 한 번에 계획
→ 빠르고 text-widget-text 구성도 쉬움
→ 다만 widget을 너무 자주 넣을 위험 있음
```

그래서 production 관점에서는 여전히:

```text
기본:
D = whole-response planning

예외:
widget 실행 이후 새로운 상태가 생기거나
실패/성공을 다시 확인해야 하면
→ B continuation
```

조합이 가장 현실적으로 보인다.
