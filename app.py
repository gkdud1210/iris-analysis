import os
import json
import uuid
import threading
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory, abort
from werkzeug.utils import secure_filename
from database import db, IrisImage, LabelCategory, Label, Prediction, TrainingSession, Subject, seed_default_categories, add_iridology_presets

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "model", "saved")
EXPORT_DIR = os.path.join(BASE_DIR, "exports")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tiff", "webp"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)

app = Flask(__name__)
app.config["SECRET_KEY"] = "iris-research-dev-key-2024"
app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{os.path.join(BASE_DIR, 'iris_data.db')}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50MB

db.init_app(app)

with app.app_context():
    db.create_all()
    # Migrate: add geometry column to existing labels table
    try:
        from sqlalchemy import inspect as _inspect, text as _text
        _cols = [c['name'] for c in _inspect(db.engine).get_columns('labels')]
        if 'geometry' not in _cols:
            db.session.execute(_text('ALTER TABLE labels ADD COLUMN geometry TEXT'))
            db.session.commit()
    except Exception:
        db.session.rollback()
    seed_default_categories(app)

TRAINING_LOGS = {}  # session_id -> list of log strings


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ─── Pages ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    with app.app_context():
        total_images = IrisImage.query.count()
        labeled_images = db.session.query(IrisImage.id).join(Label).distinct().count()
        total_labels = Label.query.count()
        total_categories = LabelCategory.query.count()
        total_sessions = TrainingSession.query.count()
    return render_template(
        "index.html",
        total_images=total_images,
        labeled_images=labeled_images,
        total_labels=total_labels,
        total_categories=total_categories,
        total_sessions=total_sessions,
    )


@app.route("/upload")
def upload_page():
    return render_template("upload.html")


@app.route("/gallery")
def gallery_page():
    return render_template("gallery.html")


@app.route("/label/<int:image_id>")
def label_page(image_id):
    image = IrisImage.query.get_or_404(image_id)
    categories = LabelCategory.query.all()
    labels = Label.query.filter_by(image_id=image_id).all()
    prev_img = IrisImage.query.filter(IrisImage.id < image_id).order_by(IrisImage.id.desc()).first()
    next_img = IrisImage.query.filter(IrisImage.id > image_id).order_by(IrisImage.id.asc()).first()
    return render_template(
        "label.html",
        image=image,
        categories=categories,
        labels=labels,
        labels_data=json.dumps([l.to_dict() for l in labels], ensure_ascii=False),
        prev_id=prev_img.id if prev_img else None,
        next_id=next_img.id if next_img else None,
    )


@app.route("/analyze/<int:image_id>")
def analyze_page(image_id):
    image = IrisImage.query.get_or_404(image_id)
    predictions = Prediction.query.filter_by(image_id=image_id).order_by(Prediction.predicted_at.desc()).all()
    sessions = TrainingSession.query.filter_by(status="completed").all()
    return render_template("analyze.html", image=image, predictions=predictions, sessions=sessions)


@app.route("/train")
def train_page():
    sessions = TrainingSession.query.order_by(TrainingSession.created_at.desc()).all()
    categories = LabelCategory.query.all()
    labeled_count = db.session.query(IrisImage.id).join(Label).distinct().count()
    return render_template("train.html", sessions=sessions, categories=categories, labeled_count=labeled_count)


@app.route("/categories")
def categories_page():
    categories = LabelCategory.query.all()
    return render_template("categories.html", categories=categories)


@app.route("/subjects")
def subjects_page():
    return render_template("subjects.html")


@app.route("/subjects/<subject_id_str>")
def subject_detail_page(subject_id_str):
    subject = Subject.query.filter_by(subject_id=subject_id_str).first()
    images  = IrisImage.query.filter_by(subject_id=subject_id_str).order_by(IrisImage.uploaded_at.desc()).all()
    return render_template("subject_detail.html", subject=subject, subject_id=subject_id_str, images=images)


# ─── API: Subjects ────────────────────────────────────────────────────────────

@app.route("/api/subjects", methods=["GET"])
def api_subjects():
    # IrisImage에서 unique subject_id 목록 수집
    rows = db.session.query(
        IrisImage.subject_id,
        db.func.count(IrisImage.id).label("image_count"),
        db.func.max(IrisImage.uploaded_at).label("last_upload"),
    ).filter(IrisImage.subject_id.isnot(None), IrisImage.subject_id != "").group_by(IrisImage.subject_id).all()

    result = []
    for row in rows:
        sub = Subject.query.filter_by(subject_id=row.subject_id).first()
        result.append({
            "subject_id":   row.subject_id,
            "image_count":  row.image_count,
            "last_upload":  row.last_upload.isoformat() if row.last_upload else None,
            "name":         sub.name if sub else None,
            "gender":       sub.gender if sub else None,
            "birth_year":   sub.birth_year if sub else None,
            "has_diagnosis": bool(sub and sub.diagnosis),
        })
    result.sort(key=lambda x: x["last_upload"] or "", reverse=True)
    return jsonify(result)


@app.route("/api/subjects/<subject_id_str>", methods=["GET"])
def api_subject_detail(subject_id_str):
    sub = Subject.query.filter_by(subject_id=subject_id_str).first()
    images = IrisImage.query.filter_by(subject_id=subject_id_str).order_by(IrisImage.uploaded_at.desc()).all()
    return jsonify({
        "subject": sub.to_dict() if sub else {"subject_id": subject_id_str},
        "images":  [img.to_dict() for img in images],
    })


@app.route("/api/subjects/<subject_id_str>", methods=["PUT"])
def api_subject_save(subject_id_str):
    data = request.get_json()
    sub  = Subject.query.filter_by(subject_id=subject_id_str).first()
    if not sub:
        sub = Subject(subject_id=subject_id_str)
        db.session.add(sub)
    sub.name       = data.get("name",       sub.name)
    sub.birth_year = data.get("birth_year", sub.birth_year)
    sub.gender     = data.get("gender",     sub.gender)
    sub.diagnosis  = data.get("diagnosis",  sub.diagnosis)
    sub.notes      = data.get("notes",      sub.notes)
    sub.updated_at = datetime.utcnow()
    db.session.commit()
    return jsonify(sub.to_dict())


# ─── API: Images ──────────────────────────────────────────────────────────────

@app.route("/api/images", methods=["GET"])
def api_images():
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 20, type=int)
    labeled_only = request.args.get("labeled_only", "false") == "true"
    unlabeled_only = request.args.get("unlabeled_only", "false") == "true"
    eye_side = request.args.get("eye_side", "")

    query = IrisImage.query
    if eye_side:
        query = query.filter_by(eye_side=eye_side)
    if labeled_only:
        query = query.join(Label).distinct()
    elif unlabeled_only:
        labeled_ids = db.session.query(Label.image_id).distinct().subquery()
        query = query.filter(~IrisImage.id.in_(labeled_ids))

    paginated = query.order_by(IrisImage.uploaded_at.desc()).paginate(page=page, per_page=per_page, error_out=False)
    return jsonify({
        "images": [img.to_dict() for img in paginated.items],
        "total": paginated.total,
        "pages": paginated.pages,
        "current_page": page,
    })


@app.route("/api/images/upload", methods=["POST"])
def api_upload():
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "파일이 없습니다"}), 400

    eye_side = request.form.get("eye_side", "unknown")
    subject_id = request.form.get("subject_id", "")
    notes = request.form.get("notes", "")

    saved = []
    errors = []
    for f in files:
        if not f or not f.filename:
            continue
        if not allowed_file(f.filename):
            errors.append(f"{f.filename}: 지원하지 않는 형식")
            continue
        ext = f.filename.rsplit(".", 1)[1].lower()
        unique_name = f"{uuid.uuid4().hex}.{ext}"
        save_path = os.path.join(UPLOAD_FOLDER, unique_name)
        f.save(save_path)
        img = IrisImage(
            filename=unique_name,
            original_filename=secure_filename(f.filename),
            file_path=save_path,
            eye_side=eye_side,
            subject_id=subject_id,
            notes=notes,
        )
        db.session.add(img)
        db.session.flush()
        saved.append(img.to_dict())

    db.session.commit()
    return jsonify({"saved": saved, "errors": errors})


@app.route("/api/images/<int:image_id>", methods=["GET"])
def api_image_detail(image_id):
    image = IrisImage.query.get_or_404(image_id)
    data = image.to_dict()
    data["labels"] = [l.to_dict() for l in image.labels]
    data["predictions"] = [p.to_dict() for p in image.predictions]
    return jsonify(data)


@app.route("/api/images/<int:image_id>", methods=["PUT"])
def api_update_image(image_id):
    image = IrisImage.query.get_or_404(image_id)
    data = request.get_json()
    image.eye_side = data.get("eye_side", image.eye_side)
    image.subject_id = data.get("subject_id", image.subject_id)
    image.notes = data.get("notes", image.notes)
    db.session.commit()
    return jsonify(image.to_dict())


@app.route("/api/images/<int:image_id>", methods=["DELETE"])
def api_delete_image(image_id):
    image = IrisImage.query.get_or_404(image_id)
    try:
        if os.path.exists(image.file_path):
            os.remove(image.file_path)
    except Exception:
        pass
    db.session.delete(image)
    db.session.commit()
    return jsonify({"success": True})


# ─── API: Labels ──────────────────────────────────────────────────────────────

@app.route("/api/labels", methods=["POST"])
def api_add_label():
    data = request.get_json()
    image_id = data.get("image_id")
    category_id = data.get("category_id")
    geom = data.get("geometry")

    # iris_measure 타입은 카테고리 없이 저장 허용
    is_iris_measure = isinstance(geom, dict) and geom.get("type") == "iris_measure"
    if not image_id or (not category_id and not is_iris_measure):
        return jsonify({"error": "image_id, category_id 필수"}), 400

    IrisImage.query.get_or_404(image_id)
    if category_id:
        LabelCategory.query.get_or_404(category_id)

    label = Label(
        image_id=image_id,
        category_id=category_id if category_id else None,
        zone=data.get("zone"),
        severity=data.get("severity", 0),
        confidence=data.get("confidence", 1.0),
        notes=data.get("notes", ""),
        geometry=json.dumps(geom) if geom else None,
    )
    db.session.add(label)
    db.session.commit()
    return jsonify(label.to_dict()), 201


@app.route("/api/labels/<int:label_id>", methods=["PUT"])
def api_update_label(label_id):
    label = Label.query.get_or_404(label_id)
    data = request.get_json()
    label.category_id = data.get("category_id", label.category_id)
    label.zone = data.get("zone", label.zone)
    label.severity = data.get("severity", label.severity)
    label.confidence = data.get("confidence", label.confidence)
    label.notes = data.get("notes", label.notes)
    if "geometry" in data:
        label.geometry = json.dumps(data["geometry"]) if data["geometry"] else None
    db.session.commit()
    return jsonify(label.to_dict())


@app.route("/api/labels/<int:label_id>", methods=["DELETE"])
def api_delete_label(label_id):
    label = Label.query.get_or_404(label_id)
    db.session.delete(label)
    db.session.commit()
    return jsonify({"success": True})


@app.route("/api/labels/image/<int:image_id>", methods=["GET"])
def api_image_labels(image_id):
    labels = Label.query.filter_by(image_id=image_id).all()
    return jsonify([l.to_dict() for l in labels])


# ─── API: Categories ──────────────────────────────────────────────────────────

@app.route("/api/categories", methods=["GET"])
def api_categories():
    cats = LabelCategory.query.all()
    return jsonify([c.to_dict() for c in cats])


@app.route("/api/categories", methods=["POST"])
def api_add_category():
    data = request.get_json()
    name = data.get("name", "").strip()
    if not name:
        return jsonify({"error": "name 필수"}), 400
    if LabelCategory.query.filter_by(name=name).first():
        return jsonify({"error": "이미 존재하는 카테고리"}), 409
    cat = LabelCategory(name=name, color=data.get("color", "#3B82F6"), description=data.get("description", ""))
    db.session.add(cat)
    db.session.commit()
    return jsonify(cat.to_dict()), 201


@app.route("/api/categories/<int:cat_id>", methods=["PUT"])
def api_update_category(cat_id):
    cat = LabelCategory.query.get_or_404(cat_id)
    data = request.get_json()
    cat.name = data.get("name", cat.name)
    cat.color = data.get("color", cat.color)
    cat.description = data.get("description", cat.description)
    db.session.commit()
    return jsonify(cat.to_dict())


@app.route("/api/categories/preset/iridology", methods=["POST"])
def api_iridology_preset():
    added = add_iridology_presets(app)
    return jsonify({"added": added, "message": f"홍채진단 카테고리 {added}개 추가됨"})


@app.route("/api/categories/<int:cat_id>", methods=["DELETE"])
def api_delete_category(cat_id):
    cat = LabelCategory.query.get_or_404(cat_id)
    db.session.delete(cat)
    db.session.commit()
    return jsonify({"success": True})


# ─── API: Training ────────────────────────────────────────────────────────────

@app.route("/api/train/start", methods=["POST"])
def api_train_start():
    data = request.get_json()
    name = data.get("name", f"Training {datetime.now().strftime('%Y%m%d_%H%M%S')}")
    model_type = data.get("model_type", "resnet50")
    epochs = int(data.get("epochs", 10))
    batch_size = int(data.get("batch_size", 16))
    learning_rate = float(data.get("learning_rate", 0.001))
    selected_categories = data.get("category_ids", [])  # empty = all

    # Gather training samples
    query = db.session.query(Label.image_id, Label.category_id, IrisImage.file_path)\
        .join(IrisImage, Label.image_id == IrisImage.id)
    if selected_categories:
        query = query.filter(Label.category_id.in_(selected_categories))
    rows = query.all()

    if len(rows) < 4:
        return jsonify({"error": "라벨된 이미지가 최소 4개 이상 필요합니다"}), 400

    # Build class map
    cat_ids = sorted(set(r.category_id for r in rows))
    if selected_categories:
        cat_ids = [c for c in cat_ids if c in selected_categories]
    cat_objects = LabelCategory.query.filter(LabelCategory.id.in_(cat_ids)).all()
    class_names = [c.name for c in cat_objects]
    cat_id_to_idx = {c.id: i for i, c in enumerate(cat_objects)}

    samples = [(r.file_path, cat_id_to_idx[r.category_id]) for r in rows if os.path.exists(r.file_path)]

    if len(samples) < 4:
        return jsonify({"error": "유효한 이미지 파일이 부족합니다"}), 400

    session = TrainingSession(
        name=name,
        model_type=model_type,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        status="running",
    )
    db.session.add(session)
    db.session.commit()
    session_id = session.id
    TRAINING_LOGS[session_id] = []

    def run_training():
        from model.trainer import train_model
        logs = []

        def log_cb(msg):
            logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
            TRAINING_LOGS[session_id] = logs[:]

        try:
            model_path, best_acc, history = train_model(
                samples=samples,
                class_names=class_names,
                session_id=session_id,
                model_type=model_type,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                save_dir=MODEL_SAVE_DIR,
                log_callback=log_cb,
            )
            with app.app_context():
                s = TrainingSession.query.get(session_id)
                s.status = "completed"
                s.model_path = model_path
                s.val_accuracy = best_acc
                s.train_accuracy = history[-1]["train_acc"] if history else None
                s.log = "\n".join(logs)
                s.completed_at = datetime.utcnow()
                # store class names in log header
                s.log = json.dumps({"class_names": class_names}) + "\n" + s.log
                db.session.commit()
        except Exception as e:
            with app.app_context():
                s = TrainingSession.query.get(session_id)
                s.status = "failed"
                s.log = "\n".join(logs) + f"\n[ERROR] {str(e)}"
                db.session.commit()
            TRAINING_LOGS[session_id].append(f"[ERROR] {str(e)}")

    t = threading.Thread(target=run_training, daemon=True)
    t.start()

    return jsonify({"session_id": session_id, "message": "학습 시작됨"})


@app.route("/api/train/<int:session_id>/log", methods=["GET"])
def api_train_log(session_id):
    logs = TRAINING_LOGS.get(session_id, [])
    session = TrainingSession.query.get_or_404(session_id)
    return jsonify({"logs": logs, "status": session.status})


@app.route("/api/train/<int:session_id>", methods=["GET"])
def api_train_detail(session_id):
    session = TrainingSession.query.get_or_404(session_id)
    return jsonify(session.to_dict())


@app.route("/api/train/<int:session_id>", methods=["DELETE"])
def api_delete_session(session_id):
    session = TrainingSession.query.get_or_404(session_id)
    if session.model_path and os.path.exists(session.model_path):
        os.remove(session.model_path)
    db.session.delete(session)
    db.session.commit()
    return jsonify({"success": True})


# ─── API: Prediction ──────────────────────────────────────────────────────────

@app.route("/api/predict/<int:image_id>", methods=["POST"])
def api_predict(image_id):
    data = request.get_json() or {}
    session_id = data.get("session_id")

    image = IrisImage.query.get_or_404(image_id)
    if not os.path.exists(image.file_path):
        return jsonify({"error": "이미지 파일을 찾을 수 없습니다"}), 404

    if session_id:
        training_session = TrainingSession.query.get_or_404(session_id)
        if training_session.status != "completed":
            return jsonify({"error": "완료된 학습 세션이 아닙니다"}), 400
        model_path = training_session.model_path
        # parse class names from log
        try:
            first_line = training_session.log.split("\n")[0]
            meta = json.loads(first_line)
            class_names = meta["class_names"]
        except Exception:
            return jsonify({"error": "클래스 정보를 불러올 수 없습니다"}), 500
    else:
        # Use latest completed session
        latest = TrainingSession.query.filter_by(status="completed").order_by(TrainingSession.completed_at.desc()).first()
        if not latest:
            return jsonify({"error": "완료된 학습 모델이 없습니다. 먼저 학습을 진행해주세요."}), 400
        model_path = latest.model_path
        try:
            first_line = latest.log.split("\n")[0]
            meta = json.loads(first_line)
            class_names = meta["class_names"]
        except Exception:
            return jsonify({"error": "클래스 정보를 불러올 수 없습니다"}), 500

    try:
        from model.iris_classifier import predict_image
        results = predict_image(image.file_path, model_path, class_names)
        pred = Prediction(
            image_id=image_id,
            model_name=training_session.name if session_id else "latest",
            results=json.dumps(results),
        )
        db.session.add(pred)
        db.session.commit()
        return jsonify({"predictions": results, "prediction_id": pred.id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─── API: Export ──────────────────────────────────────────────────────────────

@app.route("/api/export/csv", methods=["GET"])
def api_export_csv():
    import csv, io
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["image_id", "filename", "eye_side", "subject_id", "category", "zone", "severity", "confidence", "notes", "uploaded_at"])

    labels = Label.query.join(IrisImage).join(LabelCategory).all()
    for l in labels:
        writer.writerow([
            l.image_id, l.image.filename, l.image.eye_side, l.image.subject_id,
            l.category.name, l.zone or "", l.severity or 0, l.confidence or 1.0,
            l.notes or "", l.image.uploaded_at.isoformat(),
        ])

    from flask import Response
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=iris_labels.csv"},
    )


@app.route("/api/export/json", methods=["GET"])
def api_export_json():
    images = IrisImage.query.all()
    data = []
    for img in images:
        d = img.to_dict()
        d["labels"] = [l.to_dict() for l in img.labels]
        data.append(d)
    from flask import Response
    return Response(
        json.dumps(data, ensure_ascii=False, indent=2),
        mimetype="application/json",
        headers={"Content-Disposition": "attachment; filename=iris_dataset.json"},
    )


# ─── API: Iris Auto-Detection ─────────────────────────────────────────────────

def _detect_iris_opencv(image_path):
    """Hough circle detection using OpenCV. Returns normalized circle dicts."""
    import cv2
    import numpy as np

    img = cv2.imread(image_path)
    if img is None:
        return None
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    blurred = cv2.medianBlur(gray, 7)

    # ── Iris outer boundary (large circle, 25-55% of width) ──
    iris_raw = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1,
        minDist=w // 3,
        param1=50, param2=28,
        minRadius=int(w * 0.22), maxRadius=int(w * 0.52),
    )
    if iris_raw is not None:
        candidates = np.round(iris_raw[0]).astype(int)
        # prefer circle closest to image center
        candidates = sorted(candidates, key=lambda c: (c[0]-w//2)**2 + (c[1]-h//2)**2)
        icx, icy, ir = candidates[0]
    else:
        icx, icy, ir = w // 2, h // 2, int(w * 0.38)

    # ── Pupil (dark, smaller, near iris center) ──
    # Step 1: 카메라 반사광(밝은 점) 인페인팅 → 동공 내 흰 점을 주변 어두운 값으로 채움
    gray_raw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, glare_mask = cv2.threshold(gray_raw, 200, 255, cv2.THRESH_BINARY)
    glare_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    glare_mask = cv2.dilate(glare_mask, glare_kernel, iterations=1)
    gray_clean = cv2.inpaint(gray_raw, glare_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
    blurred_clean = cv2.medianBlur(gray_clean, 7)

    # Step 2: ROI 크롭 (홍채 영역 내부)
    crop_margin = max(0, ir - 10)
    x1 = max(0, icx - crop_margin); x2 = min(w, icx + crop_margin)
    y1 = max(0, icy - crop_margin); y2 = min(h, icy + crop_margin)
    roi = blurred_clean[y1:y2, x1:x2] if y2 > y1 and x2 > x1 else blurred_clean

    # Step 3: Hough 원 검출 (minRadius를 12%로 높여 반사광 소원 제거)
    pupil_raw = cv2.HoughCircles(
        roi, cv2.HOUGH_GRADIENT, dp=1,
        minDist=max(1, (x2 - x1) // 3),
        param1=60, param2=22,
        minRadius=max(5, int(ir * 0.12)),
        maxRadius=int(ir * 0.55),
    )
    if pupil_raw is not None:
        pcands = np.round(pupil_raw[0]).astype(int)
        pcands[:, 0] += x1; pcands[:, 1] += y1   # restore full-image coords
        # 홍채 중심에 가장 가깝고 크기가 적절한 원 선택
        pcands = sorted(pcands, key=lambda c: (c[0]-icx)**2 + (c[1]-icy)**2)
        pcx, pcy, pr = pcands[0]
    else:
        # 폴백: 인페인팅 이미지에서 가장 어두운 중심 영역으로 동공 추정
        roi_clean = gray_clean[y1:y2, x1:x2] if y2 > y1 and x2 > x1 else gray_clean
        roi_dark = cv2.GaussianBlur(roi_clean, (21, 21), 0)
        _, _, min_loc, _ = cv2.minMaxLoc(roi_dark)
        pcx = min_loc[0] + x1; pcy = min_loc[1] + y1
        pr = int(ir * 0.28)

    f = lambda v: round(float(v), 4)   # numpy scalar → Python float
    return {
        "pupil": {"center": [f(pcx/w), f(pcy/h)], "radius": f(pr/w)},
        "iris":  {"center": [f(icx/w), f(icy/h)], "radius": f(ir/w)},
        "ratio": f(pr / ir) if ir else 0.28,
        "method": "opencv",
    }


def _detect_iris_pil(image_path):
    """Brightness-based fallback using PIL + numpy (no OpenCV required)."""
    from PIL import Image, ImageFilter
    import numpy as np

    img = Image.open(image_path).convert("L")
    w, h = img.size
    arr = np.array(img, dtype=np.float32)

    smooth = np.array(
        Image.fromarray(arr.astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=max(3, w//60))),
        dtype=np.float32,
    )

    # 반사광 제거: 밝은 점(>200)을 주변값으로 대체
    bright_mask = arr > 200
    if bright_mask.any():
        from scipy.ndimage import binary_dilation, label as ndi_label
        bright_mask = binary_dilation(bright_mask, iterations=4)
        from PIL import ImageFilter as _IF
        filled = np.array(Image.fromarray(arr.astype(np.uint8)).filter(_IF.GaussianBlur(radius=max(5, w//40))), dtype=np.float32)
        arr = np.where(bright_mask, filled, arr)
        smooth = np.array(
            Image.fromarray(arr.astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=max(3, w//60))),
            dtype=np.float32,
        )

    # Find darkest point in center 50% region (likely pupil center)
    roi = smooth[h//4:3*h//4, w//4:3*w//4]
    idx = np.unravel_index(np.argmin(roi), roi.shape)
    pcy_abs = idx[0] + h // 4
    pcx_abs = idx[1] + w // 4

    # Walk outward from center until brightness jumps (pupil edge)
    thresh = smooth[pcy_abs, pcx_abs] * 2.2
    radii = []
    for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
        for r in range(1, w // 3):
            nx, ny = pcx_abs + dx*r, pcy_abs + dy*r
            if not (0 <= nx < w and 0 <= ny < h):
                break
            if smooth[ny, nx] > thresh:
                radii.append(r)
                break
        else:
            radii.append(w // 12)
    pr = max(4, int(np.median(radii)))
    ir = min(int(pr * 3.4), int(min(w, h) * 0.48))

    return {
        "pupil": {"center": [round(pcx_abs/w, 4), round(pcy_abs/h, 4)], "radius": round(pr/w, 4)},
        "iris":  {"center": [round(pcx_abs/w, 4), round(pcy_abs/h, 4)], "radius": round(ir/w, 4)},
        "ratio": round(pr / ir, 4),
        "method": "pil",
    }


@app.route("/api/images/<int:image_id>/detect_iris", methods=["POST"])
def api_detect_iris(image_id):
    image = IrisImage.query.get_or_404(image_id)
    image_path = os.path.join(UPLOAD_FOLDER, image.filename)
    if not os.path.exists(image_path):
        return jsonify({"error": "이미지 파일을 찾을 수 없습니다."}), 404

    try:
        result = _detect_iris_opencv(image_path)
        if result is None:
            raise RuntimeError("OpenCV detection returned None")
    except Exception:
        try:
            result = _detect_iris_pil(image_path)
        except Exception as e2:
            return jsonify({"error": f"인식 실패: {e2}"}), 500

    return jsonify({"success": True, "circles": result})


# ─── 눈꺼풀 감지 파이프라인 ────────────────────────────────────────────────────

@app.route("/api/eyelid/from_labels/<int:image_id>", methods=["GET"])
def api_eyelid_from_labels(image_id):
    """이미 라벨링된 눈꺼풀 polygon/polyline → 각도 마스크로 변환."""
    from training.eyelid_pipeline import (
        polygon_to_angle_mask, _polyline_to_closed_polygon, _is_iris_measure
    )
    from database import Label, LabelCategory, IrisImage
    import os as _os

    image = IrisImage.query.get_or_404(image_id)
    img_w = img_h = None

    # 이미지 크기 읽기
    if _os.path.exists(image.file_path):
        import cv2 as _cv2
        _img = _cv2.imread(image.file_path)
        if _img is not None:
            img_h, img_w = _img.shape[:2]

    if img_w is None:
        return jsonify({"error": "이미지 파일을 읽을 수 없습니다."}), 400

    # iris_measure 찾기
    all_labels = Label.query.filter_by(image_id=image_id).all()
    iris_geom = next(
        (l.geometry_parsed for l in all_labels if _is_iris_measure(l.geometry_parsed)),
        None
    )
    if not iris_geom:
        return jsonify({"count": 0, "mask": [0] * 360})

    icx = iris_geom["iris"]["center"][0] * img_w
    icy = iris_geom["iris"]["center"][1] * img_h
    ir  = iris_geom["iris"]["radius"]    * img_w
    pr  = iris_geom["pupil"]["radius"]   * img_w

    # 눈꺼풀 카테고리 찾기
    cats = LabelCategory.query.filter(
        LabelCategory.name.ilike('%눈꺼풀%') |
        LabelCategory.name.ilike('%eyelid%')
    ).all()
    if not cats:
        return jsonify({"count": 0, "mask": [0] * 360})
    cat_map = {c.id: c.name for c in cats}

    eyelid_labels = [
        l for l in all_labels
        if l.category_id in cat_map
        and l.geometry_parsed
        and l.geometry_parsed.get("type") in ("polygon", "polyline")
    ]
    if not eyelid_labels:
        return jsonify({"count": 0, "mask": [0] * 360})

    import numpy as np
    combined    = np.zeros(360, dtype=np.uint8)
    labels_data = []
    for label in eyelid_labels:
        gp       = label.geometry_parsed
        coords   = gp["coords"]
        is_upper = "위" in cat_map[label.category_id] or "upper" in cat_map[label.category_id].lower()
        if gp["type"] == "polyline":
            coords = _polyline_to_closed_polygon(coords, is_upper)
        mask = polygon_to_angle_mask(coords, img_w, img_h, icx, icy, ir, pr)
        combined = np.maximum(combined, mask)
        labels_data.append({
            "coords": gp["coords"],   # 원본 라벨 좌표 (정규화)
            "type": gp["type"],
            "is_upper": is_upper,
        })

    return jsonify({"count": len(eyelid_labels), "mask": combined.tolist(), "labels": labels_data})


@app.route("/api/eyelid/status", methods=["GET"])
def api_eyelid_status():
    from training.eyelid_pipeline import model_exists, MODEL_PATH
    import os, datetime
    exists = model_exists()
    mtime  = None
    if exists:
        mtime = datetime.datetime.fromtimestamp(os.path.getmtime(MODEL_PATH)).isoformat()
    return jsonify({"model_exists": exists, "trained_at": mtime})


@app.route("/api/eyelid/train", methods=["POST"])
def api_eyelid_train():
    from training.eyelid_pipeline import build_dataset, train
    try:
        X, y, msg = build_dataset(app)
        if X is None:
            return jsonify({"error": msg}), 400
        model, metrics = train(X, y)
        return jsonify({
            "success": True,
            "data_info": msg,
            "metrics": metrics,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/eyelid/detect/<int:image_id>", methods=["POST"])
def api_eyelid_detect(image_id):
    from training.eyelid_pipeline import predict, model_exists
    import cv2

    if not model_exists():
        return jsonify({"error": "학습된 모델이 없습니다. 먼저 학습을 실행하세요."}), 400

    body = request.get_json(silent=True) or {}
    iris_geom = body.get("iris_geometry")
    if not iris_geom:
        return jsonify({"error": "iris_geometry 파라미터가 필요합니다."}), 400

    image = IrisImage.query.get_or_404(image_id)
    img   = cv2.imread(image.file_path)
    if img is None:
        return jsonify({"error": "이미지를 읽을 수 없습니다."}), 500

    h, w  = img.shape[:2]
    icx   = iris_geom['iris']['center'][0]  * w
    icy   = iris_geom['iris']['center'][1]  * h
    ir    = iris_geom['iris']['radius']     * w
    pcx   = iris_geom['pupil']['center'][0] * w
    pcy   = iris_geom['pupil']['center'][1] * h
    pr    = iris_geom['pupil']['radius']    * w

    try:
        mask, probs = predict(img, icx, icy, ir, pcx, pcy, pr)
        occluded_pct = round(sum(mask) / len(mask) * 100, 1)
        return jsonify({
            "success": True,
            "mask": mask,          # list[int] 360개, 1=차폐
            "probs": [round(p, 3) for p in probs],
            "occluded_pct": occluded_pct,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─── Static ───────────────────────────────────────────────────────────────────

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5050)
