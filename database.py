from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json as _json

db = SQLAlchemy()


class Subject(db.Model):
    __tablename__ = "subjects"
    id          = db.Column(db.Integer, primary_key=True)
    subject_id  = db.Column(db.String(100), unique=True, nullable=False)  # IrisImage.subject_id와 일치
    name        = db.Column(db.String(200))
    birth_year  = db.Column(db.Integer)
    gender      = db.Column(db.String(10))   # '남', '여', '기타'
    diagnosis   = db.Column(db.Text)          # 자유 텍스트 진단/소견
    notes       = db.Column(db.Text)
    created_at  = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at  = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        return {
            "id":         self.id,
            "subject_id": self.subject_id,
            "name":       self.name,
            "birth_year": self.birth_year,
            "gender":     self.gender,
            "diagnosis":  self.diagnosis,
            "notes":      self.notes,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class IrisImage(db.Model):
    __tablename__ = "iris_images"
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    original_filename = db.Column(db.String(255), nullable=False)
    file_path = db.Column(db.String(512), nullable=False)
    eye_side = db.Column(db.String(10))  # 'left', 'right', 'unknown'
    subject_id = db.Column(db.String(100))
    notes = db.Column(db.Text)
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)
    labels = db.relationship("Label", backref="image", lazy=True, cascade="all, delete-orphan")
    predictions = db.relationship("Prediction", backref="image", lazy=True, cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "id": self.id,
            "filename": self.filename,
            "original_filename": self.original_filename,
            "eye_side": self.eye_side,
            "subject_id": self.subject_id,
            "notes": self.notes,
            "uploaded_at": self.uploaded_at.isoformat(),
            "label_count": len(self.labels),
        }


class LabelCategory(db.Model):
    __tablename__ = "label_categories"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False, unique=True)
    color = db.Column(db.String(20), default="#3B82F6")
    description = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    labels = db.relationship("Label", backref="category", lazy=True)

    @property
    def label_count(self):
        return len(self.labels)

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "color": self.color,
            "description": self.description,
            "label_count": self.label_count,
        }


class Label(db.Model):
    __tablename__ = "labels"
    id = db.Column(db.Integer, primary_key=True)
    image_id = db.Column(db.Integer, db.ForeignKey("iris_images.id"), nullable=False)
    category_id = db.Column(db.Integer, db.ForeignKey("label_categories.id"), nullable=True)
    zone = db.Column(db.String(50))
    severity = db.Column(db.Integer)
    confidence = db.Column(db.Float)
    notes = db.Column(db.Text)
    geometry = db.Column(db.Text)   # JSON: {"type":"point|polyline|polygon","coords":[[nx,ny],...]}
    labeled_at = db.Column(db.DateTime, default=datetime.utcnow)

    @property
    def geometry_parsed(self):
        try:
            return _json.loads(self.geometry) if self.geometry else None
        except Exception:
            return None

    def to_dict(self):
        return {
            "id": self.id,
            "image_id": self.image_id,
            "category_id": self.category_id,
            "category_name": self.category.name if self.category else None,
            "category_color": self.category.color if self.category else None,
            "zone": self.zone,
            "severity": self.severity,
            "confidence": self.confidence,
            "notes": self.notes,
            "geometry": self.geometry_parsed,

            "labeled_at": self.labeled_at.isoformat(),
        }


class Prediction(db.Model):
    __tablename__ = "predictions"
    id = db.Column(db.Integer, primary_key=True)
    image_id = db.Column(db.Integer, db.ForeignKey("iris_images.id"), nullable=False)
    model_name = db.Column(db.String(100))
    results = db.Column(db.Text)  # JSON string
    predicted_at = db.Column(db.DateTime, default=datetime.utcnow)

    def get_results(self):
        return json.loads(self.results) if self.results else {}

    def to_dict(self):
        return {
            "id": self.id,
            "image_id": self.image_id,
            "model_name": self.model_name,
            "results": self.get_results(),
            "predicted_at": self.predicted_at.isoformat(),
        }


class TrainingSession(db.Model):
    __tablename__ = "training_sessions"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    model_type = db.Column(db.String(50), default="resnet50")
    status = db.Column(db.String(20), default="pending")  # pending, running, completed, failed
    epochs = db.Column(db.Integer, default=10)
    batch_size = db.Column(db.Integer, default=16)
    learning_rate = db.Column(db.Float, default=0.001)
    train_accuracy = db.Column(db.Float)
    val_accuracy = db.Column(db.Float)
    model_path = db.Column(db.String(512))
    log = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    completed_at = db.Column(db.DateTime)

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "model_type": self.model_type,
            "status": self.status,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "train_accuracy": self.train_accuracy,
            "val_accuracy": self.val_accuracy,
            "model_path": self.model_path,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


IRIDOLOGY_PRESETS = [
    # ── 해부학적 구조 (눈꺼풀) ────────────────────────────────────────────────
    ("눈꺼풀 위 (Upper Eyelid)",   "#92400E", "해부 구조 | 상안검 — 눈꺼풀 차폐 영역 표시용"),
    ("눈꺼풀 아래 (Lower Eyelid)", "#78350F", "해부 구조 | 하안검 — 눈꺼풀 차폐 영역 표시용"),
    # ── 홍채 병소 (진단 기호) ──────────────────────────────────────────────────
    ("라쿠나 (Lacuna)",            "#DC2626", "홍채 병소 | 장기 기능 약화 표시 (타원형 구멍)"),
    ("크립트 (Crypt)",             "#F97316", "홍채 병소 | 급성/활동성 병변 (작은 구멍)"),
    ("색소 침착 (Pigment Spot)",   "#B45309", "홍채 병소 | 독소·약물 침착 표시"),
    ("방사선 (Radii Solaris)",     "#EF4444", "홍채 병소 | 동공에서 방사형 선"),
    ("수축 고리 (Contraction Ring)","#7C3AED","홍채 병소 | 스트레스·긴장 표시"),
    ("신경 고리 (Nerve Ring)",     "#8B5CF6", "홍채 병소 | 신경계 스트레스"),
    ("림프 로제리 (Lymph Rosary)", "#0EA5E9", "홍채 병소 | 림프 울체 표시"),
    ("독소 침착 (Toxic Deposit)",  "#065F46", "홍채 병소 | 독소 축적"),
    ("아르쿠스 세닐리스 (Arcus Senilis)", "#9CA3AF", "홍채 병소 | 노화·동맥경화 표시"),
    ("혈관 징후 (Vascular Sign)",  "#BE123C", "홍채 병소 | 순환기 이상 표시"),
]


def seed_default_categories(app):
    with app.app_context():
        if LabelCategory.query.count() == 0:
            for name, color, desc in IRIDOLOGY_PRESETS:
                db.session.add(LabelCategory(name=name, color=color, description=desc))
            db.session.commit()


def add_iridology_presets(app):
    """Add iridology preset categories, skipping names that already exist."""
    with app.app_context():
        added = 0
        for name, color, desc in IRIDOLOGY_PRESETS:
            if not LabelCategory.query.filter_by(name=name).first():
                db.session.add(LabelCategory(name=name, color=color, description=desc))
                added += 1
        db.session.commit()
        return added
