# 🔬 IRIS Analysis System — by Dr. Choi

홍채(Iris) 이미지 기반 **라벨링 · 학습 · 분석 · 진단 보조 웹 플랫폼**입니다.  
Flask 기반 웹 애플리케이션으로, 홍채 이미지를 업로드하고 병소를 라벨링한 뒤 AI 모델을 학습시켜 자동 분류 및 예측까지 수행합니다.

---

## ✨ 주요 기능

| 기능 | 설명 |
|------|------|
| 📤 이미지 업로드 | PNG · JPG · BMP · TIFF · WebP 등 다양한 형식 지원, 최대 50MB |
| 🏷️ 라벨링 도구 | 포인트 · 폴리라인 · 폴리곤 · 홍채 측정(iris_measure) 형태의 기하 라벨 |
| 🔍 홍채 자동 검출 | OpenCV Hough Circle + PIL 폴백으로 동공·홍채 경계 자동 인식 |
| 🤖 AI 학습 | ResNet50 기반 이미지 분류 모델 학습 (에폭·배치·학습률 설정 가능) |
| 📊 예측 분석 | 학습된 모델로 홍채 이미지 자동 분류 및 예측 결과 저장 |
| 👁️ 눈꺼풀 감지 | 눈꺼풀 차폐 영역을 360° 각도 마스크로 분석 |
| 👤 피험자 관리 | 피험자별 이미지·진단·소견 통합 관리 |
| 📁 데이터 내보내기 | CSV / JSON 형식으로 라벨 데이터 일괄 내보내기 |

---

## 🏗️ 프로젝트 구조

```
iris-analysis_by_DrChoi/
├── app.py                  # Flask 메인 앱 (라우팅 · API 전체)
├── database.py             # SQLAlchemy 모델 (Subject, IrisImage, Label, Prediction, TrainingSession)
├── requirements.txt        # Python 의존성
├── run.sh                  # 실행 스크립트
├── model/
│   ├── iris_classifier.py  # 추론(Inference) 모듈
│   └── trainer.py          # ResNet50 학습 모듈
├── training/
│   └── eyelid_pipeline.py  # 눈꺼풀 감지 파이프라인
├── templates/              # Jinja2 HTML 템플릿
│   ├── base.html
│   ├── index.html
│   ├── upload.html
│   ├── gallery.html
│   ├── label.html
│   ├── analyze.html
│   ├── train.html
│   ├── categories.html
│   ├── subjects.html
│   └── subject_detail.html
└── static/
    └── uploads/            # 업로드된 이미지 저장 경로
```

---

## 🩺 홍채진단 라벨 카테고리 (기본 제공)

시스템 최초 실행 시 다음 카테고리가 자동으로 생성됩니다:

**해부학적 구조**
- 눈꺼풀 위 (Upper Eyelid)
- 눈꺼풀 아래 (Lower Eyelid)

**홍채 병소 (Iridology)**
- 라쿠나 (Lacuna) — 장기 기능 약화
- 크립트 (Crypt) — 급성/활동성 병변
- 색소 침착 (Pigment Spot) — 독소·약물 침착
- 방사선 (Radii Solaris) — 동공 방사형 선
- 수축 고리 (Contraction Ring) — 스트레스·긴장
- 신경 고리 (Nerve Ring) — 신경계 스트레스
- 림프 로제리 (Lymph Rosary) — 림프 울체
- 독소 침착 (Toxic Deposit) — 독소 축적
- 아르쿠스 세닐리스 (Arcus Senilis) — 노화·동맥경화
- 혈관 징후 (Vascular Sign) — 순환기 이상

---

## ⚙️ 설치 및 실행

### 요구사항
- Python 3.9 이상
- CUDA (선택, GPU 가속 시)

### 설치

```bash
git clone https://github.com/EmmettHwang/iris-analysis_by_DrChoi.git
cd iris-analysis_by_DrChoi

pip install -r requirements.txt
```

### 실행

```bash
python app.py
# 또는
bash run.sh
```

브라우저에서 `http://localhost:5050` 접속

---
## 🔌 API 엔드포인트 요약

### 이미지
| Method | Endpoint | 설명 |
|--------|----------|------|
| GET | `/api/images` | 이미지 목록 (페이지네이션) |
| POST | `/api/images/upload` | 이미지 업로드 |
| GET | `/api/images/<id>` | 이미지 상세 |
| PUT | `/api/images/<id>` | 이미지 정보 수정 |
| DELETE | `/api/images/<id>` | 이미지 삭제 |
| POST | `/api/images/<id>/detect_iris` | 홍채 자동 검출 |

### 라벨
| Method | Endpoint | 설명 |
|--------|----------|------|
| POST | `/api/labels` | 라벨 추가 |
| PUT | `/api/labels/<id>` | 라벨 수정 |
| DELETE | `/api/labels/<id>` | 라벨 삭제 |
| GET | `/api/labels/image/<id>` | 이미지별 라벨 조회 |

### 학습 · 예측
| Method | Endpoint | 설명 |
|--------|----------|------|
| POST | `/api/train/start` | 모델 학습 시작 |
| GET | `/api/train/<id>/log` | 학습 로그 조회 |
| POST | `/api/predict/<image_id>` | 이미지 예측 |

### 내보내기
| Method | Endpoint | 설명 |
|--------|----------|------|
| GET | `/api/export/csv` | CSV 내보내기 |
| GET | `/api/export/json` | JSON 내보내기 |

---

## 🧠 AI 모델 구성

- **백본**: ResNet50 (torchvision 사전학습 모델)
- **학습 방식**: 전이학습 (Transfer Learning)
- **입력**: 홍채 이미지 (다양한 해상도 지원)
- **출력**: 라벨 카테고리별 분류 확률
- **설정 가능 하이퍼파라미터**: 에폭, 배치 크기, 학습률

---

## 👁️ 홍채 자동 검출 알고리즘

1. **1차 (OpenCV)**: CLAHE 전처리 → Hough Circle 변환으로 동공·홍채 경계 검출
2. **2차 폴백 (PIL)**: 밝기 기반 동공 중심 탐색 → 반경 추정

검출 결과는 정규화 좌표(0~1)로 반환되며 라벨링 도구와 연동됩니다.

---

## 🛠️ 기술 스택

| 분류 | 기술 |
|------|------|
| 백엔드 | Flask 3.0, SQLAlchemy, SQLite |
| AI/ML | PyTorch 2.3, torchvision, scikit-learn |
| 이미지 처리 | OpenCV, Pillow, NumPy |
| 프론트엔드 | Jinja2 템플릿 (HTML/JS) |
| 데이터 | Pandas, CSV/JSON 내보내기 |

---

## 📄 라이선스

본 프로젝트는 연구 및 교육 목적으로 개발되었습니다.  
상업적 이용 전 저작권자(Dr. Choi)의 허가를 받으시기 바랍니다.

---

## 👨‍⚕️ 개발자

**Dr. Choi** | 홍채 분석 연구  

