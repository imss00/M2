# Team7 M2: TPCx-AI UC03 + UC08 파이프라인

TPCx-AI **UC03(주간 매출 예측)**과 **UC08(쇼핑 트립 분류)**를 원본 CSV부터 예측·채점까지 한 번에 실행하는
PySpark + Hive 파이프라인이다. 같은 코드가 노트북 한 대(`local`)와 Docker로 띄운 Hadoop 클러스터
(HDFS + YARN, `--master yarn`)에서 모두 돈다. 처리 규칙은 TPCx-AI Tools v2.0.0의 공식 Python 구현
(`UseCase03.py`, `UseCase08.py`)에 맞췄고, 다른 점은 8절에 모았다. 설계 근거는 M1 보고서 절 번호로 코드 주석에 표시했다.

## 0. 결과 요약 (SF1, Docker 클러스터, 키트 버전 라이브러리)

| 항목 | separate | shared | 기준 |
| --- | --- | --- | --- |
| UC03 MSLE | 4.01 | 4.01 | 5.4 이하 (통과) |
| UC08 accuracy | 0.693 | 0.693 | 0.65 이상 (통과) |
| 전체 시간 | 332초 | 377초 | |
| 전처리 합계 (train + serve + score) | 52초 | 73초 | |
| UC08 학습 (전체의 약 60%) | 212초 | 221초 | |
| 검증 실패 | 0건 | 0건 | |

- 실행 환경: Mac(Apple Silicon) 위 Docker, HDFS DataNode 2대, Spark on YARN executor 2개 × 3코어 4GB, driver 3GB. 각 1회 측정.
- 두 모드의 점수는 같고, SF1에서는 조인 결과를 저장하는 shared가 이득이 없다(12.3절).
- 키트와 같은 모델 라이브러리(xgboost 1.7.3 등)를 써야 UC08이 기준을 넘는다. 최신 버전에서는 0.567(11절).

## 폴더 구조

```
M2/
  run.py                 실행 진입점 (train → serve → score)
  src/common/            설정(키트 부서·라벨 목록, 기준값), SparkSession(Hive), 단계 타이머, 모델 작업 실행
  src/prep/              원본 적재·정제·라벨 분리(ingest.py), 조인과 separate/shared 전환(join.py)
  src/uc03/              주간 집계, 쌍별 ETS 학습·서빙, 채점(MSLE)
  src/uc08/              주문 특성(Spark), XGBoost 학습·서빙, 채점(accuracy)
  src/jobs/              모델 단계(UC03 ETS, UC08 XGBoost)를 별도 Python 프로세스로 실행
  src/checks/outputs.py  출력 파일 검증
  tools/                 가짜 데이터 생성, 실행 비교, 키트 로직·버전과의 비교 스크립트
  docker/, docker-compose.yml   Hadoop 클러스터(HDFS + YARN) 이미지·설정·실행 스크립트
  requirements*.txt      파이프라인용 / 키트 버전 모델용 패키지
```

키트 코드와 생성 데이터는 TPC·bankmark 라이선스 대상이라 저장소에 넣지 않는다. 실행 결과(`runs/`)도 올리지 않는다.

## 1. 설치 (로컬 모드)

- **Java 17 또는 21** (Spark 4.0 요구사항). `java -version`으로 확인
- Python 3.10 이상

```bash
pip install -r requirements.txt
```

키트와 같은 결과를 내려면 모델 단계용으로 키트 버전 환경을 하나 더 만든다(11절 참고).

```bash
uv venv --python 3.9 kitenv
uv pip install --python kitenv/bin/python -r requirements-kit.txt
python run.py ... --model-python kitenv/bin/python
```

로컬 모드에서는 Hadoop이나 Hive를 따로 설치하지 않는다. PySpark에 들어 있는 Hive 기능(`enableHiveSupport()`)을 쓰며,
테이블 정보는 `metastore_db/`, 테이블 데이터는 `warehouse/`에 저장된다.

## 2. 빠른 실행 (가짜 데이터)

```bash
python tools/gen_fake_data.py --out data/fake
python run.py --data-dir data/fake --sf fake --mode shared
```

가짜 데이터에는 검증 확인용 이상치가 일부러 들어 있다(매칭 안 되는 키, 중복 상품, 잘못된 날짜).
그래서 `checks failed: 2`가 나오는 것이 정상이다.

## 3. 실제 데이터 (TPCx-AI 키트 PDGF)

Docker를 쓰면 `docker/gen_data.sh`가 아래 과정을 amd64 + Java 8 컨테이너에서 대신 해준다(12절).
직접 돌릴 때는 PDGF를 **x86 Linux + Java 8**에서 실행한다. Java 11 이상에서는 `ClassCastException`으로,
ARM(Apple Silicon)에서는 JHDF5 네이티브 라이브러리가 없어서 실패한다. 처음 실행하면 PDGF 라이선스 동의(Enter 후 `YES`)를 묻는다.
작업 디렉터리는 키트의 `data-gen/`이어야 한다. 시드·옵션은 키트 드라이버와 같다.

```bash
cd <kit>/data-gen
J=<java8>/bin/java
O="-l config/tpcxai-schema.xml -l config/tpcxai-generation.xml -ns"
T="order lineitem product store_dept"
R=<kit>/output/raw_data
run() { printf '\nYES\n' | $J -Djava.awt.headless=true -Xmx4g -jar ../lib/pdgf/pdgf.jar $O "$@"; }
run -sf 1 -sp MY_SEED 1234 -sp includeLabels 1.0 -sp TTVF 1.0 -s $T -output "\"$R/training/\""
run -sf 1 -sp SF_TRAINING 1 -sp MY_SEED 1235 -sp includeLabels 0.0 -sp TTVF 0.1 -s $T -output "\"$R/serving/\""
run -sf 1 -sp SF_TRAINING 1 -sp MY_SEED 1235 -sp includeLabels 2.0 -sp TTVF 0.1 -s $T order_weekly_sales -output "\"$R/scoring/\""
```

학습·서빙·채점은 같은 SF로 만들어야 한다(채점 정답은 `SF_TRAINING`에 의존). 그다음 실행한다.

```bash
python run.py --data-dir <kit>/output/raw_data --sf 1 --mode shared --workers 4 --driver-memory 8g
```

- 폴더를 따로 주려면 `--train-dir`, `--serve-dir`, `--score-dir`
- 구분자가 쉼표가 아니면 `--sep '|'`. 컬럼 이름이 다르면 `src/common/config.py`의 `SOURCE_SCHEMA`만 고친다.

## 4. 실행 옵션

| 옵션 | 기본값 | 설명 |
| --- | --- | --- |
| `--mode` | shared | `separate`: UC03·UC08이 각자 조인 / `shared`: 폴더마다 한 번 조인해 Hive 테이블로 저장 후 공유 |
| `--master` | local | `yarn`이면 Hadoop 클러스터에서 실행(12절) |
| `--executors`, `--executor-cores`, `--executor-memory` | | YARN executor 수·코어·메모리 |
| `--workers` | 2 | 로컬 모드 Spark `local[N]` 코어 수 |
| `--driver-memory` | 4g | Spark driver 메모리 |
| `--sf` | 1 | 실행 이름과 Hive DB 이름(`sf1_clean` 등)에 쓰는 라벨 |
| `--workloads` | both | `uc03` 또는 `uc08`만 실행 |
| `--split-join-timing` | 꺼짐 | shared 모드에서 Order–Lineitem 조인과 Product 조인 시간을 따로 측정(중간 저장이 추가됨) |
| `--fill-missing-weeks` | 꺼짐 | UC03: 판매 없는 주를 0으로 채움(키트는 채우지 않음) |
| `--default-periods` | 52 | store_dept에 `periods`가 없을 때의 예측 주 수 |
| `--xgb-estimators` | 100 | UC08 트리 수(키트 드라이버 `--num-rounds 100`) |
| `--model-python` | 현재 Python | 모델 단계를 실행할 Python. 키트 버전 환경을 주면 키트와 같은 결과 |
| `--reingest` | 꺼짐 | clean 테이블이 있어도 원본부터 다시 적재 |
| `--run-id` | 자동 | 결과 폴더 이름 |

clean 테이블이 이미 있으면 적재를 건너뛰고, 그때의 데이터 품질 검사 결과를 그대로 가져와 기록한다.

## 5. 결과물

```
runs/<run_id>/
  uc03/predictions.csv           serving 예측  (store, department, date, weekly_sales)
  uc03/scoring_predictions.csv   scoring 예측  (채점에 사용)
  uc08/predictions.csv           serving 예측  (o_order_id, trip_type)
  uc08/scoring_predictions.csv   scoring 예측
  uc08/feature_columns.json      특성 컬럼 순서와 dept_NNN -> 부서 이름
  metrics.json                   실행 조건, 단계별 시간, phase_seconds, shapes(단계별 행·열 수)
  stages.csv                     단계별 시간 (pipeline, phase 포함. 여러 실행을 합치기 쉬운 형태)
  validation.json                모든 검증 결과 (PASS / FAIL / INFO). 채점 결과도 여기에
  spark-events/                  Spark 이벤트 로그 (셔플 바이트, 스필, 태스크 시간)
```

채점 결과: `uc03.score.msle`(기준 5.4 이하), `uc08.score.accuracy`(기준 0.65 이상).
기준은 키트 `driver/dat/thresholds.yaml`과 같다.

## 6. 데이터 계층 (Hive DB, M1 Table 4)

| DB | 테이블 | 단위 |
| --- | --- | --- |
| `sf<N>_raw` | `{train,serve,score}_{order,lineitem,product,store_dept}` | 원본 CSV 위의 외부 테이블 (복사 없음) |
| `sf<N>_clean` | 같은 이름 | 타입 변환, 결측·중복 키 제거된 Parquet |
| `sf<N>_labels` | `train_order_labels`, `score_order_labels`, `score_store_dept_labels` | 정답 (특성과 분리) |
| `sf<N>_joined` | `{split}_lineitem_joined` (+ `{split}_order_lineitem`) | line item 1행 (shared 모드) |
| `sf<N>_features` | `uc03_weekly_sales`, `uc03_models`, `uc08_{split}_order_features` | 주간 매출 / 쌍별 모델 / 주문당 특성 |

확인: `spark.sql("SHOW TABLES IN sf1_clean").show()`

## 7. 파이프라인

```
train : raw CSV ─ingest→ clean ─join(Order⋈Lineitem⋈broadcast Product)→ joined
        joined ─(store, department, ISO 주 금요일)→ weekly ─ETS per pair→ uc03_models
        joined ─(order_id)→ order features ─XGBoost(hist)→ model
serve : store_dept(periods) + uc03_models → uc03/predictions.csv
        serving order/lineitem/product → features → uc08/predictions.csv
score : scoring 폴더로 serve와 같은 처리 → 정답 파일과 merge → MSLE / accuracy
```

**UC08 메모리 구조**: SF1 학습 행렬은 주문 520만 × 77열, 클래스 37개라 XGBoost 기울기 버퍼만 약 1.5GB다.
Spark JVM과 XGBoost가 동시에 메모리를 잡으면 OOM으로 종료되어, `run.py`는
(1) Spark로 모든 특성 테이블을 만들고 (2) Spark를 종료한 뒤 (3) `src/jobs/`의 별도 프로세스가 Parquet 파일을
pyarrow로 직접 읽어 희소 행렬 + `QuantileDMatrix`로 학습·예측한다. 알고리즘과 하이퍼파라미터는 키트와 같다.
클러스터 모드에서는 이 프로세스가 client 컨테이너에서 HDFS 파일을 내려받아 실행한다.

## 8. 키트(Python 구현)와의 관계

같은 것

- UC03: quantity × price를 매장·부서·ISO 주별로 합산, 주의 금요일을 날짜로 사용. 빈 주를 채우지 않음.
  쌍마다 `ExponentialSmoothing(seasonal="add", seasonal_periods=52)`. 쌍의 마지막 주 다음 금요일부터
  `periods`주 예측, 음수는 0. 모델 없는 쌍은 건너뜀.
- UC08: scan_count, scan_count_abs, 요일 원-핫 7개, 키트 순서의 68개 부서별 수량 합.
  trip_type 14 제외, 학습 주문 전체로 학습(검증 분할 없음). `XGBClassifier(tree_method="hist",
  objective="multi:softprob", n_estimators=100)`.
- 채점: 예측과 정답을 공통 키로 merge, MSLE(UC03) / accuracy(UC08).
- 검증: `tools/kit_uc08_reference.py`로 키트 UC08 로직과 비교한 결과, 52만 주문 전부 같은 예측(일치율 1.0000).

다른 것 (보고서에 명시)

- **ISO 주 연도 처리**: 키트는 12월 말인데 ISO 1주차인 날짜를 같은 해 1월 첫 주로 넣는 버그가 있다.
  우리는 정확한 ISO 주 금요일을 쓰고, 영향받는 행 수를 `uc03.weekly.kit_year_bug_rows`로 기록한다.
- **계절 모델 실패 시 대체**: 키트는 실패하면 실행이 멈춘다. 우리는 단순 지수평활 → 마지막 값으로 대체하고
  `uc03.train.model_kinds`에 기록한다.
- **라이브러리 버전**: 키트는 statsmodels 0.12.2, xgboost 1.7.3, pandas 1.5.2. 결과가 버전에 크게 좌우되므로(11절)
  보고 기준선은 `--model-python`으로 키트 버전을 쓴다. 기본값(최신 버전)은 비교용이다.
- **실행 엔진**: 키트의 Spark 버전 UC03·UC08은 Scala jar로만 제공되어(소스 없음), Python 구현을 PySpark로 재구현했다.
- 공식 TPCx-AI 벤치마크 실행이 아니므로 결과를 AIUCpm 점수로 표시하지 않는다.

## 9. 실행 비교 (M3용)

```bash
python tools/compare_runs.py runs/<separate 실행> runs/<shared 실행>   # phase별 합계, 단계별 시간, 예측 일치
python tools/compare_runs.py runs/*                                    # runs/all_stages.csv 생성
python -m src.checks.outputs runs/<run_id>                             # 출력만 다시 검증
```

- 모든 단계에 pipeline(train/serve/score)과 phase(load/preprocess/train/predict/score/check)가 붙는다.
  `check`는 검증용이라 work 시간에서 빠진다.
- separate vs shared 비교의 핵심 값은 `train/preprocess`, `serve/preprocess`, `score/preprocess` 합계다.
  load는 두 모드가 공유한다(clean 테이블 재사용 시 0).
- separate 모드는 조인이 집계 단계(`uc03_aggregate_weekly`, `uc08_features_*`) 안에 포함된다(Spark 지연 실행).
- `--split-join-timing`을 켠 실행은 중간 저장이 더해지므로 조인 분리 시간을 볼 때만 쓴다.
- 메모리·스필·셔플 바이트: `spark-events/`를 Spark History Server(`docker/history_server.sh`)로 연다.

## 10. 주의 사항

- **Hive 메타스토어는 드라이버에 내장(Derby)**되어 동시에 한 프로세스만 쓸 수 있다. 같은 폴더에서 두 실행을 동시에
  돌리면 오류가 난다. 측정은 한 번에 하나씩 한다.
- 같은 SF의 clean·joined·features 테이블은 실행 간에 공유되고 덮어써진다. 예측과 기록은 `runs/<run_id>`에 따로 남는다.

## 11. 개발 환경 실행 결과 (클라우드 작업공간, x86 2코어, Spark local[2])

모델 라이브러리 버전에 따라 결과가 크게 달라서 두 가지로 실행했다. 데이터 처리(Spark)는 같고 모델 단계만 다르다.
기계가 12절 클러스터와 다르므로 시간을 클러스터와 직접 비교하지 않는다.

| 항목 | 키트 버전 (`--model-python`, 기준선) | 최신 버전 (기본값) |
| --- | --- | --- |
| 모델 라이브러리 | Python 3.9, statsmodels 0.12.2, xgboost 1.7.3 | Python 3.13, statsmodels 0.15.0, xgboost 3.4.1 |
| UC03 MSLE (기준 ≤ 5.4) | **3.95** 통과 | 3.76 통과 |
| UC03 계절 모델 실패 | 0쌍 | 39쌍 (104주 미만 → 단순 지수평활) |
| UC08 accuracy (기준 ≥ 0.65) | **0.693** 통과 | 0.567 미달 |
| work 시간 shared / separate | 1,310초 / 1,181초 | 1,611초 / 1,498초 |
| separate vs shared 예측 | 4개 파일 모두 동일 | 4개 파일 모두 동일 |

- 같은 특성·하이퍼파라미터에서 xgboost 버전만 바꾸면 UC08 예측의 39%가 달라진다(`tools/kit_versions_uc08.py`).
- 학습 데이터 실제 행 수: order 5,200,000 / lineitem 32,565,017 / product 1,000 / store_dept 748
  (명세 추정치 3,676,955 / 23,026,666 / 707 / 748). 서빙·채점: 각각 order 520,000 / lineitem 3,261,300.
- 같은 키트 버전인데 UC03 MSLE가 x86(3.95)과 Apple Silicon 클러스터(4.01)에서 조금 다르다. 부동소수점 계산 차이로 보이며 둘 다 통과다.

### UC03 채점 데이터의 한계

- 요청 748쌍 중 517쌍만 학습 이력이 있다. 생성된 Product에 부서가 47개뿐이고 store_dept는 68개 부서를 요청한다.
  키트처럼 이력 없는 쌍을 건너뛰어 정답의 69%만 채점된다.
- **채점 정답(`store_dept_labels.csv`)이 거래 데이터와 무관하다.** 쌍별 주간 매출 중앙값의 로그 상관이 학습 거래와
  0.00, 채점 거래와 0.00(517쌍). PDGF가 `order_weekly_sales` 테이블을 주문·구매와 따로 생성하기 때문이다.
  따라서 UC03 MSLE는 예측 모델의 품질보다 생성기 정답의 수준에 좌우된다.
- 빈 주 실험(`--fill-missing-weeks`, `tools/compare_uc03_fill.py`): 빈 주가 있는 39쌍(110주)을 0으로 채우면
  MSLE 3.76 → 3.94로 나빠진다. 정답이 거래와 무관해 예측 품질 비교로 해석할 수 없으므로 기본값은 키트 방식(채우지 않음)으로 둔다.

## 12. Hadoop 클러스터 실행 (Docker, HDFS + YARN + Hive)

수업의 Hadoop 도구(HDFS, YARN, Hive)로 같은 파이프라인을 실행한다. 코드는 같고 `--master yarn`만 다르다.

```
            ┌──────────── Docker network ───────────────┐
 client ──▶ │ namenode (HDFS)     resourcemanager (YARN) │
 run.py     │ worker1: DataNode + NodeManager            │
 (driver,   │ worker2: DataNode + NodeManager            │
  Hive 메타스토어(Derby),                                  │
  모델 작업) └────────────────────────────────────────────┘
```

| 계층 | 로컬 모드 | 클러스터 모드 |
| --- | --- | --- |
| 저장 | 로컬 디스크 | **HDFS** (DataNode 2대, 복제 2) |
| 계산 | Spark `local[N]` | **Spark on YARN** (executor 2개, 워커마다 1개) |
| 테이블 정보 | 내장 Hive 메타스토어(Derby) | 내장 Hive 메타스토어(Derby, client 컨테이너), warehouse는 HDFS |
| 모델 단계 | 별도 프로세스 | 같음 (HDFS 파일을 내려받아 실행) |

### 12.1 실행 방법

준비물: Docker Desktop(Settings > Resources에서 메모리 16GB 이상), 키트 폴더(`tpcx-ai-v2.0.0`).
이미지는 Apple Silicon에서 네이티브(arm64)로 빌드된다. PDGF만 amd64 에뮬레이션으로 돈다.

```bash
# 1) 데이터 생성 (호스트에서, 약 수 분)  -> data/sf1/{training,serving,scoring}
docker/gen_data.sh ~/Downloads/tpcx-ai-v2.0.0 1

# 2) 이미지 빌드(처음 한 번, 약 15분)와 클러스터 시작
docker compose build
docker compose up -d

# 3) HDFS에 적재
docker compose exec client docker/load_hdfs.sh 1

# 4) 실행 (shared, separate 두 번. 키트 버전 모델)
docker compose exec client docker/run_cluster.sh 1            # [SF] [executors] [cores] [memory]
```

- 웹 UI: HDFS http://localhost:9870 (DataNode 2대, 블록 위치), YARN http://localhost:8088 (실행 중인 앱, executor 배치),
  Spark http://localhost:4040 (실행 중 단계·셔플).
- 결과는 로컬 모드와 같은 형식으로 `runs/<run_id>/`에 생긴다(호스트 폴더와 공유).
- 워커당 YARN 자원은 `docker/hadoop-conf/yarn-site.xml`(6GB, 4 vcore).
- 끄기: `docker compose down`(데이터 유지), 완전히 지우기: `docker compose down -v`.

### 12.2 메타스토어

`docker-compose.yml`에는 PostgreSQL 컨테이너(`metastore-db`)와 이를 쓰는 설정(`TPCXAI_METASTORE_URL`)이 있지만
아직 실험 단계라 비워 두었다. 현재 실행은 메타스토어를 client의 Spark 드라이버 안에 Derby로 내장한다.
테이블 데이터는 HDFS에 있으므로 결과에는 영향이 없지만, client 컨테이너를 지우면 테이블 목록이 사라지고 여러 실행이
메타스토어를 동시에 쓸 수 없다. PostgreSQL 분리는 향후 과제다.

### 12.3 SF1 클러스터 결과 (executor 2 × 3코어 4GB, 키트 버전 모델, 각 1회)

| 단계 (초) | separate | shared |
| --- | --- | --- |
| 적재 | 0 (clean 테이블 재사용) | 30 |
| 조인 저장 | (집계 안에 포함) | 16 |
| UC03 집계 + 학습 | 13 | 10 |
| UC08 특성 만들기 | 43 | 53 |
| UC08 학습 | 212 | 221 |
| 예측 + 채점 | 24 | 24 |
| 전체 | 332 | 377 |

- 전처리 합계(9절 기준)는 separate 52초, shared 73초로 SF1에서는 조인 결과를 저장·재사용하는 이득이 없다.
- UC08 학습이 전체의 약 60%다. Spark 밖 단일 프로세스라 executor를 늘려도 줄지 않을 것으로 예상되며, M3에서 확인한다.

## 13. M3로 넘기는 것

- SF3·SF10으로 데이터 확장, executor 수 변경, separate vs shared를 각 3회 반복 측정
- 로컬 모드를 단일 프로세스 기준선으로 같이 측정
- Docker 클러스터는 Mac 한 대의 자원을 나눠 쓰므로, 워커 확장 결과는 이 한계와 함께 해석
