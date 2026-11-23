
"""DENGU HUNTER unified prototype engine.

Real YOLO inference for live submissions.
Human review before worker assignment.
GPS-supported case management.
Drone routes are simulations, not real flights.
Inspection priority is not dengue outbreak prediction.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from threading import RLock

import hashlib
import html
import io
import json
import math
import re
import secrets
import statistics
import uuid

import cv2
import folium
import numpy as np
import pandas as pd

from PIL import Image, ImageDraw

UTC = timezone.utc
CASE_PATTERN = re.compile(r"^DH-[A-F0-9]{12}$")
MAX_PIXELS = 20_000_000


def utc_now():
    return datetime.now(UTC).isoformat(timespec="seconds")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(
            lambda: stream.read(1024 * 1024), b""
        ):
            h.update(block)
    return h.hexdigest()


def image_rgb(image):
    if image is None:
        raise ValueError("An actual image is required.")

    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image))

    if image.width * image.height > MAX_PIXELS:
        raise ValueError("Image exceeds 20 megapixels.")

    if image.width < 32 or image.height < 32:
        raise ValueError("Image is too small.")

    return image.convert("RGB")


def check_gps(lat, lon):
    if lat in ("", None) and lon in ("", None):
        return None, None

    if lat in ("", None) or lon in ("", None):
        raise ValueError(
            "Provide BOTH latitude and longitude."
        )

    a, b = float(lat), float(lon)

    if not (
        math.isfinite(a)
        and math.isfinite(b)
        and -90 <= a <= 90
        and -180 <= b <= 180
    ):
        raise ValueError("Invalid GPS coordinates.")

    return a, b


def draw_detections(image, detections):
    annotated = image_rgb(image).copy()
    drawer = ImageDraw.Draw(annotated)

    for det in detections:
        x1, y1, x2, y2 = (
            int(round(float(n)))
            for n in det["bbox_xyxy"]
        )

        x1 = max(0, min(annotated.width, x1))
        x2 = max(0, min(annotated.width, x2))
        y1 = max(0, min(annotated.height, y1))
        y2 = max(0, min(annotated.height, y2))

        if x2 <= x1 or y2 <= y1:
            continue

        drawer.rectangle(
            (x1, y1, x2, y2),
            outline="#D82E2E",
            width=4
        )

        label = (
            f'{det["class_name"]}: '
            f'{float(det["confidence"]):.2f}'
        )

        drawer.text(
            (x1 + 2, max(0, y1 - 17)),
            label,
            fill="#D82E2E",
            stroke_width=1,
            stroke_fill="white"
        )

    return annotated


class RealYoloDetector:
    """Loads actual trained YOLO weights."""

    def __init__(self, weights, device=None):
        path = Path(weights)

        if not path.is_file():
            raise FileNotFoundError(
                f"YOLO weights not found: {path}"
            )

        import torch
        from ultralytics import YOLO

        self.weights = str(path)
        self.device = (
            device
            if device is not None
            else (0 if torch.cuda.is_available() else "cpu")
        )

        self.model = YOLO(self.weights)

    def predict(self, image):
        rgb = image_rgb(image)

        result = self.model.predict(
            source=np.asarray(rgb),
            imgsz=640,
            conf=0.25,
            device=self.device,
            verbose=False
        )[0]

        names = result.names
        found = []

        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            classes = (
                result.boxes.cls.cpu().numpy().astype(int)
            )

            for xyxy, score, class_id in zip(
                boxes, scores, classes
            ):
                if isinstance(names, dict):
                    label = names.get(
                        int(class_id),
                        f"class_{class_id}"
                    )
                else:
                    label = names[int(class_id)]

                found.append({
                    "class_name": str(label),
                    "class_id": int(class_id),
                    "confidence": round(float(score), 5),
                    "bbox_xyxy": [
                        round(float(v), 2)
                        for v in xyxy
                    ]
                })

        return found


def heuristic_priority(
    detections,
    rain_context="Unknown",
    previous_reports=0
):
    """Inspection priority, not dengue forecasting."""

    allowed = {
        "Unknown",
        "Dry",
        "Recent rain",
        "Heavy rain"
    }

    if rain_context not in allowed:
        raise ValueError("Invalid rain context")

    count = int(previous_reports)

    if count < 0:
        raise ValueError(
            "Previous reports cannot be negative"
        )

    if not detections:
        return {
            "score": 0,
            "level": "UNCONFIRMED",
            "explanation": (
                "No visible model detection. "
                "Human inspection may still be needed."
            ),
            "method": (
                "HEURISTIC_INSPECTION_PRIORITY_"
                "NOT_DENGUE_FORECAST"
            )
        }

    max_conf = max(
        float(d["confidence"])
        for d in detections
    )

    labels = " ".join(
        str(d["class_name"]).lower()
        for d in detections
    )

    keywords = (
        "tire",
        "tyre",
        "vase",
        "container",
        "tank",
        "drain"
    )

    bonus = (
        12
        if any(word in labels for word in keywords)
        else 6
    )

    rain_bonus = {
        "Unknown": 0,
        "Dry": 0,
        "Recent rain": 8,
        "Heavy rain": 14
    }[rain_context]

    history_bonus = min(15, count * 3)

    score = min(
        100,
        round(
            15
            + 38 * max_conf
            + bonus
            + rain_bonus
            + history_bonus
        )
    )

    level = (
        "HIGH"
        if score >= 70
        else "MEDIUM"
        if score >= 45
        else "LOW"
    )

    return {
        "score": score,
        "level": level,
        "explanation": (
            f"Confidence contribution="
            f"{round(38 * max_conf)}; "
            f"site type={bonus}; "
            f"rain={rain_bonus}; "
            f"history={history_bonus}; base=15."
        ),
        "method": (
            "HEURISTIC_INSPECTION_PRIORITY_"
            "NOT_DENGUE_FORECAST"
        )
    }


def recommend_action(detections):
    labels = " ".join(
        str(d["class_name"]).lower()
        for d in detections
    )

    if not detections:
        return (
            "Request a closer human inspection "
            "before assigning intervention."
        )

    if "tire" in labels or "tyre" in labels:
        return (
            "Inspect discarded tyres and safely "
            "remove collected water."
        )

    if "drain" in labels:
        return (
            "Inspect drainage and clear safe, "
            "accessible blockages."
        )

    if "tank" in labels:
        return (
            "Inspect water tank and ensure "
            "proper covering."
        )

    return (
        "Inspect potential water-holding "
        "containers; empty or cover where safe."
    )


def alignment_evidence(before, after):
    """Visual similarity, not proof of same location."""

    try:
        def gray(im):
            arr = np.asarray(image_rgb(im))
            h, w = arr.shape[:2]

            scale = min(
                1.0,
                960 / max(h, w)
            )

            if scale < 1:
                arr = cv2.resize(
                    arr,
                    (
                        max(32, round(w * scale)),
                        max(32, round(h * scale))
                    )
                )

            return cv2.cvtColor(
                arr,
                cv2.COLOR_RGB2GRAY
            )

        a = gray(before)
        b = gray(after)

        orb = cv2.ORB_create(nfeatures=600)

        k1, d1 = orb.detectAndCompute(a, None)
        k2, d2 = orb.detectAndCompute(b, None)

        if (
            d1 is None
            or d2 is None
            or len(k1) < 10
            or len(k2) < 10
        ):
            return {
                "status": "INCONCLUSIVE",
                "good_matches": 0,
                "homography_inlier_ratio": None,
                "same_site_proven": False
            }

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        pairs = matcher.knnMatch(d1, d2, k=2)

        good = [
            m
            for pair in pairs
            if len(pair) == 2
            for m, n in [pair]
            if m.distance < 0.75 * n.distance
        ]

        if len(good) < 12:
            return {
                "status": "INCONCLUSIVE",
                "good_matches": len(good),
                "homography_inlier_ratio": None,
                "same_site_proven": False
            }

        p1 = np.float32([
            k1[m.queryIdx].pt for m in good
        ]).reshape(-1, 1, 2)

        p2 = np.float32([
            k2[m.trainIdx].pt for m in good
        ]).reshape(-1, 1, 2)

        _, mask = cv2.findHomography(
            p1,
            p2,
            cv2.RANSAC,
            5.0
        )

        ratio = (
            round(float(mask.mean()), 3)
            if mask is not None
            else None
        )

        status = (
            "GEOMETRIC_MATCH_CANDIDATE"
            if ratio is not None and ratio >= 0.45
            else "INCONCLUSIVE"
        )

        return {
            "status": status,
            "good_matches": len(good),
            "homography_inlier_ratio": ratio,
            "same_site_proven": False
        }

    except (cv2.error, ValueError, TypeError):
        return {
            "status": "INCONCLUSIVE",
            "good_matches": 0,
            "homography_inlier_ratio": None,
            "same_site_proven": False
        }

# ============================================================
# PART 2 — CASE MANAGEMENT ENGINE
# ============================================================

class DenguHunterEngine:

    def __init__(
        self,
        data_root,
        f2_detector,
        f3_detector=None,
        f2_model_label="EXISTING_TRAINED_F2_YOLO",
        f3_model_label=None
    ):
        if f2_detector is None:
            raise ValueError(
                "Live F2 detector is mandatory."
            )

        self.root = Path(data_root)
        self.cases_dir = self.root / "cases"
        self.audit_dir = self.root / "audits"

        self.cases_dir.mkdir(
            parents=True, exist_ok=True
        )
        self.audit_dir.mkdir(
            parents=True, exist_ok=True
        )

        self.f2 = f2_detector
        self.f3 = f3_detector
        self.f2_label = f2_model_label
        self.f3_label = f3_model_label

        self.f2_weights_sha256 = (
            sha256_file(f2_detector.weights)
            if isinstance(f2_detector, RealYoloDetector)
            else None
        )

        self.f3_weights_sha256 = (
            sha256_file(f3_detector.weights)
            if isinstance(f3_detector, RealYoloDetector)
            else None
        )

        self.lock = RLock()

    def _case_path(self, case_id):
        if not CASE_PATTERN.fullmatch(str(case_id or "")):
            raise ValueError("Invalid Case ID.")

        return self.cases_dir / case_id / "case.json"

    def _read(self, case_id):
        path = self._case_path(case_id)

        if not path.is_file():
            raise ValueError(
                f"Unknown Case ID: {case_id}"
            )

        return json.loads(
            path.read_text(encoding="utf-8")
        )

    def _save(self, case):
        path = self._case_path(case["case_id"])
        path.parent.mkdir(
            parents=True, exist_ok=True
        )

        temporary = path.with_name(
            f".case_{uuid.uuid4().hex}.tmp"
        )

        temporary.write_text(
            json.dumps(
                case,
                indent=2,
                ensure_ascii=False
            ),
            encoding="utf-8"
        )

        temporary.replace(path)

    @staticmethod
    def _event(case, action, actor, note=""):
        case["events"].append({
            "at_utc": utc_now(),
            "action": action,
            "actor": str(actor),
            "note": str(note or "")[:500]
        })

        case["updated_at_utc"] = utc_now()

    # --------------------------------------------------------
    # CITIZEN REPORT + REAL YOLO
    # --------------------------------------------------------

    def submit(
        self,
        image,
        source="Ground photo",
        latitude=None,
        longitude=None,
        rain_context="Unknown",
        previous_reports=0,
        closeup=None,
        notes="",
        evidence_mode="DEMO_CASE_STAGED"
    ):
        with self.lock:

            if evidence_mode not in {
                "DEMO_CASE_STAGED",
                "FIELD_REPORT_UNVERIFIED"
            }:
                raise ValueError(
                    "Invalid evidence mode."
                )

            if source not in {
                "Ground photo",
                "Aerial photo (user-provided)",
                "Drone simulation frame"
            }:
                raise ValueError(
                    "Invalid image source."
                )

            before = image_rgb(image)

            previous_reports = int(
                previous_reports or 0
            )

            if previous_reports < 0:
                raise ValueError(
                    "Previous reports cannot be negative."
                )

            lat, lon = check_gps(
                latitude, longitude
            )

            f2_detections = self.f2.predict(before)

            if not isinstance(f2_detections, list):
                raise RuntimeError(
                    "Unexpected YOLO output."
                )

            closeup_image = (
                image_rgb(closeup)
                if closeup is not None
                else None
            )

            if (
                closeup_image is not None
                and self.f3 is None
            ):
                raise ValueError(
                    "F3 close-up model unavailable."
                )

            f3_detections = (
                self.f3.predict(closeup_image)
                if closeup_image is not None
                else []
            )

            priority = heuristic_priority(
                f2_detections,
                rain_context,
                previous_reports
            )

            case_id = (
                "DH-" + uuid.uuid4().hex[:12].upper()
            )

            folder = self._case_path(
                case_id
            ).parent

            folder.mkdir(
                parents=True,
                exist_ok=False
            )

            before_path = folder / "before.png"
            overlay_path = (
                folder / "before_overlay.png"
            )

            before.save(before_path)

            overlay_image = draw_detections(
                before,
                f2_detections
            )
            overlay_image.save(overlay_path)

            closeup_path = None
            closeup_sha = None

            if closeup_image is not None:
                cp = folder / "larvae_closeup.png"
                closeup_image.save(cp)
                closeup_path = str(cp)
                closeup_sha = sha256_file(cp)

            case = {
                "case_id": case_id,
                "created_at_utc": utc_now(),
                "updated_at_utc": utc_now(),

                "source": source,
                "evidence_mode": evidence_mode,
                "image_kind": "USER_UPLOADED_REAL_IMAGE",

                "latitude": lat,
                "longitude": lon,
                "gps_provided": lat is not None,
                "gps_verified_by_device": False,

                "rain_context": rain_context,
                "previous_reports": previous_reports,
                "notes": str(notes or "")[:1000],

                "before_path": str(before_path),
                "before_sha256": sha256_file(before_path),

                "before_overlay_path": str(overlay_path),
                "before_overlay_sha256": sha256_file(overlay_path),

                "f2_model": self.f2_label,
                "f2_weights_sha256": self.f2_weights_sha256,
                "f2_detections": f2_detections,

                "f3_model": (
                    self.f3_label
                    if closeup_image is not None
                    else None
                ),
                "f3_weights_sha256": (
                    self.f3_weights_sha256
                    if closeup_image is not None
                    else None
                ),
                "f3_detections": f3_detections,
                "f3_closeup_path": closeup_path,
                "f3_closeup_sha256": closeup_sha,

                "aedes_species_confirmed": False,

                "priority": priority,
                "recommendation": recommend_action(
                    f2_detections
                ),

                "review_status": "PENDING",
                "reviewer": None,
                "reviewed_at_utc": None,

                "task_id": None,
                "task_status": "NOT_CREATED",
                "assigned_worker": None,
                "task_created_at_utc": None,

                "after_path": None,
                "after_sha256": None,
                "after_overlay_path": None,
                "after_overlay_sha256": None,
                "after_detections": None,

                "alignment": None,
                "ai_change_assessment": None,

                "final_review": "NOT_REVIEWED",
                "final_reviewer": None,

                "real_drone_flight": False,
                "real_field_verified": False,
                "verified_cleanup_success": False,

                "stage": "AWAITING_HUMAN_REVIEW",
                "events": []
            }

            self._event(
                case,
                "CITIZEN_SUBMITTED_REAL_F2_INFERENCE",
                "citizen",
                "Human review required."
            )

            self._save(case)

            return case, overlay_image

    # --------------------------------------------------------
    # HUMAN REVIEW
    # --------------------------------------------------------

    def review(
        self,
        case_id,
        reviewer,
        decision,
        notes=""
    ):
        with self.lock:
            case = self._read(case_id)
            reviewer = str(reviewer or "").strip()

            if not reviewer:
                raise ValueError(
                    "Reviewer name required."
                )

            if decision not in {
                "APPROVE",
                "REJECT",
                "REQUEST_MORE_EVIDENCE"
            }:
                raise ValueError(
                    "Invalid review decision."
                )

            if case["stage"] != "AWAITING_HUMAN_REVIEW":
                raise ValueError(
                    "Case is not awaiting initial review."
                )

            case["review_status"] = decision
            case["reviewer"] = reviewer
            case["reviewed_at_utc"] = utc_now()

            if decision == "APPROVE":
                case["task_id"] = (
                    "DH-TASK-" + case_id[3:]
                )
                case["task_created_at_utc"] = utc_now()
                case["task_status"] = "READY_TO_ASSIGN"
                case["stage"] = "TASK_READY"
            else:
                case["stage"] = (
                    "REJECTED"
                    if decision == "REJECT"
                    else "MORE_EVIDENCE_REQUIRED"
                )

            self._event(
                case,
                "HUMAN_REVIEW_" + decision,
                reviewer,
                notes
            )

            self._save(case)
            return case

    # --------------------------------------------------------
    # WORKER ASSIGNMENT
    # --------------------------------------------------------

    def assign(
        self,
        case_id,
        worker,
        notes=""
    ):
        with self.lock:
            case = self._read(case_id)
            worker = str(worker or "").strip()

            if not worker:
                raise ValueError(
                    "Worker name required."
                )

            if (
                case["stage"] != "TASK_READY"
                or case["review_status"] != "APPROVE"
            ):
                raise ValueError(
                    "Approved unassigned task required."
                )

            case["assigned_worker"] = worker
            case["task_status"] = "ASSIGNED"
            case["stage"] = "TASK_ASSIGNED"

            self._event(
                case,
                "TASK_ASSIGNED",
                worker,
                notes
            )

            self._save(case)
            return case

    def start(self, case_id, worker):
        with self.lock:
            case = self._read(case_id)

            worker = str(worker or "").strip()

            if (
                case["stage"] != "TASK_ASSIGNED"
                or case["assigned_worker"] != worker
            ):
                raise ValueError(
                    "Only assigned worker can start task."
                )

            case["task_status"] = "IN_PROGRESS"
            case["stage"] = "WORK_IN_PROGRESS"

            self._event(
                case,
                "WORK_STARTED",
                worker
            )

            self._save(case)
            return case

    # --------------------------------------------------------
    # BEFORE / AFTER EVIDENCE
    # --------------------------------------------------------

    def submit_after(
        self,
        case_id,
        worker,
        image,
        notes="",
        same_site_attested=False
    ):
        with self.lock:
            case = self._read(case_id)
            worker = str(worker or "").strip()

            if (
                case["stage"] != "WORK_IN_PROGRESS"
                or case["assigned_worker"] != worker
            ):
                raise ValueError(
                    "Only assigned worker can submit after photo."
                )

            if not same_site_attested:
                raise ValueError(
                    "Same-site worker attestation required."
                )

            after = image_rgb(image)

            after_detections = self.f2.predict(after)

            with Image.open(
                case["before_path"]
            ) as original:
                before = original.convert("RGB")

            alignment = alignment_evidence(
                before,
                after
            )

            before_count = len(
                case["f2_detections"]
            )
            after_count = len(
                after_detections
            )

            if before_count == 0:
                change = (
                    "INCONCLUSIVE_NO_ORIGINAL_DETECTION"
                )
            elif (
                alignment["status"]
                != "GEOMETRIC_MATCH_CANDIDATE"
            ):
                change = (
                    "INCONCLUSIVE_ALIGNMENT_REVIEW_REQUIRED"
                )
            elif after_count < before_count:
                change = (
                    "POSSIBLE_REDUCTION_HUMAN_REVIEW_REQUIRED"
                )
            else:
                change = (
                    "POSSIBLE_PERSISTENCE_HUMAN_REVIEW_REQUIRED"
                )

            folder = self._case_path(
                case_id
            ).parent

            path = folder / "after.png"
            overlay_path = (
                folder / "after_overlay.png"
            )

            if path.exists():
                raise ValueError(
                    "After evidence already exists."
                )

            after.save(path)

            overlay = draw_detections(
                after,
                after_detections
            )
            overlay.save(overlay_path)

            case["after_path"] = str(path)
            case["after_sha256"] = sha256_file(path)

            case["after_overlay_path"] = str(
                overlay_path
            )
            case["after_overlay_sha256"] = (
                sha256_file(overlay_path)
            )

            case["after_detections"] = after_detections
            case["alignment"] = alignment
            case["ai_change_assessment"] = change

            case["worker_same_site_attestation"] = True

            case["task_status"] = (
                "AFTER_PHOTO_SUBMITTED"
            )
            case["stage"] = (
                "AWAITING_FINAL_HUMAN_REVIEW"
            )

            self._event(
                case,
                "AFTER_EVIDENCE_SUBMITTED_REAL_F2_INFERENCE",
                worker,
                notes
            )

            self._save(case)
            return case, overlay

    # --------------------------------------------------------
    # FINAL HUMAN VERIFICATION
    # --------------------------------------------------------

    def finalize(
        self,
        case_id,
        reviewer,
        decision,
        notes=""
    ):
        with self.lock:
            case = self._read(case_id)
            reviewer = str(reviewer or "").strip()

            if not reviewer:
                raise ValueError(
                    "Final reviewer name required."
                )

            if (
                case["stage"]
                != "AWAITING_FINAL_HUMAN_REVIEW"
            ):
                raise ValueError(
                    "After-action evidence required."
                )

            if decision not in {
                "RESOLVED_EVIDENCE_REVIEWED",
                "UNRESOLVED",
                "REINSPECTION_REQUIRED"
            }:
                raise ValueError(
                    "Invalid final decision."
                )

            case["final_review"] = decision
            case["final_reviewer"] = reviewer
            case["task_status"] = decision

            case["stage"] = (
                "CLOSED_EVIDENCE_REVIEWED"
                if decision == "RESOLVED_EVIDENCE_REVIEWED"
                else decision
            )

            case["real_field_verified"] = False
            case["verified_cleanup_success"] = False

            self._event(
                case,
                "FINAL_HUMAN_REVIEW_" + decision,
                reviewer,
                notes
            )

            self._save(case)
            return case

    # --------------------------------------------------------
    # CASE INSPECTION + FILE INTEGRITY
    # --------------------------------------------------------

    def inspect_case(self, case_id):
        with self.lock:
            case = self._read(case_id)

            pairs = (
                ("before_path", "before_sha256"),
                (
                    "before_overlay_path",
                    "before_overlay_sha256"
                ),
                ("after_path", "after_sha256"),
                (
                    "after_overlay_path",
                    "after_overlay_sha256"
                )
            )

            for path_key, sha_key in pairs:
                path = case.get(path_key)

                if path:
                    if (
                        not Path(path).is_file()
                        or sha256_file(path) != case[sha_key]
                    ):
                        raise ValueError(
                            f"Evidence integrity failed: {path_key}"
                        )

            def open_optional(key):
                if not case.get(key):
                    return None

                with Image.open(case[key]) as img:
                    return img.convert("RGB")

            return (
                open_optional("before_path"),
                open_optional("before_overlay_path"),
                open_optional("after_path"),
                open_optional("after_overlay_path"),
                safe_case_summary(case)
            )

    def cases(self):
        with self.lock:
            found = []

            for path in sorted(
                self.cases_dir.glob("DH-*/case.json")
            ):
                try:
                    obj = json.loads(
                        path.read_text(encoding="utf-8")
                    )

                    if CASE_PATTERN.fullmatch(
                        obj.get("case_id", "")
                    ):
                        found.append(obj)

                except (OSError, ValueError, TypeError):
                    continue

            return sorted(
                found,
                key=lambda c: c.get(
                    "created_at_utc", ""
                ),
                reverse=True
            )

    # --------------------------------------------------------
    # DASHBOARD TABLE
    # --------------------------------------------------------

    def table(self):
        rows = []

        for case in self.cases():
            lat = case.get("latitude")
            lon = case.get("longitude")

            has_gps = (
                case.get("gps_provided")
                and lat is not None
                and lon is not None
                and (float(lat), float(lon)) != (0.0, 0.0)
            )

            gps = (
                f"{float(lat):.5f}, {float(lon):.5f}"
                if has_gps
                else "NOT PROVIDED"
            )

            rows.append({
                "Case ID": case["case_id"],
                "Created (UTC)": case["created_at_utc"],
                "Source": case["source"],
                "GPS": gps,
                "Detections": len(case["f2_detections"]),
                "Risk": case["priority"]["level"],
                "Priority Score": case["priority"]["score"],
                "Human Review": case["review_status"],
                "Task ID": case["task_id"] or "—",
                "Worker": case["assigned_worker"] or "—",
                "Stage": case["stage"],
                "Evidence Mode": case.get(
                    "evidence_mode",
                    "LEGACY_UNSPECIFIED"
                ),
                "Field Verified": "NO"
            })

        return pd.DataFrame(
            rows,
            columns=[
                "Case ID",
                "Created (UTC)",
                "Source",
                "GPS",
                "Detections",
                "Risk",
                "Priority Score",
                "Human Review",
                "Task ID",
                "Worker",
                "Stage",
                "Evidence Mode",
                "Field Verified"
            ]
        )

    # --------------------------------------------------------
    # GPS MAP
    # --------------------------------------------------------

    def map_html(self):
        cases = [
            case
            for case in self.cases()
            if (
                case.get("gps_provided")
                and case.get("latitude") is not None
                and case.get("longitude") is not None
                and (
                    float(case["latitude"]),
                    float(case["longitude"])
                ) != (0.0, 0.0)
            )
        ]

        if cases:
            center = [
                statistics.mean(
                    float(c["latitude"])
                    for c in cases
                ),
                statistics.mean(
                    float(c["longitude"])
                    for c in cases
                )
            ]
        else:
            center = [23.685, 90.3563]

        fmap = folium.Map(
            location=center,
            zoom_start=12 if cases else 6,
            control_scale=True,
            tiles="OpenStreetMap"
        )

        colors = {
            "HIGH": "red",
            "MEDIUM": "orange",
            "LOW": "green",
            "UNCONFIRMED": "gray"
        }

        for case in cases:
            description = html.escape(
                f'{case["case_id"]} | '
                f'{case["priority"]["level"]} | '
                f'{case["stage"]} | '
                f'{case.get("evidence_mode", "LEGACY_UNSPECIFIED")}'
            )

            folium.Marker(
                [
                    float(case["latitude"]),
                    float(case["longitude"])
                ],
                popup=description,
                tooltip=html.escape(case["case_id"]),
                icon=folium.Icon(
                    color=colors.get(
                        case["priority"]["level"],
                        "blue"
                    ),
                    icon="info-sign"
                )
            ).add_to(fmap)

        return fmap._repr_html_()

    # --------------------------------------------------------
    # DASHBOARD SUMMARY
    # --------------------------------------------------------

    def dashboard(self):
        all_cases = self.cases()
        total = len(all_cases)

        tasks = [
            c for c in all_cases
            if c.get("task_id")
        ]

        afters = [
            c for c in tasks
            if c.get("after_path")
        ]

        gps_count = sum(
            bool(c.get("gps_provided"))
            and c.get("latitude") is not None
            and c.get("longitude") is not None
            and (
                float(c["latitude"]),
                float(c["longitude"])
            ) != (0.0, 0.0)
            for c in all_cases
        )

        delays = []

        for case in tasks:
            try:
                created = datetime.fromisoformat(
                    case["created_at_utc"]
                )
                task_created = datetime.fromisoformat(
                    case["task_created_at_utc"]
                )

                delays.append(
                    (task_created - created).total_seconds()
                )

            except (ValueError, TypeError):
                pass

        summary = {
            "citizen_reports": total,

            "staged_demo_cases": sum(
                c.get("evidence_mode")
                == "DEMO_CASE_STAGED"
                for c in all_cases
            ),

            "unverified_field_reports": sum(
                c.get("evidence_mode")
                == "FIELD_REPORT_UNVERIFIED"
                for c in all_cases
            ),

            "verified_cleanup_successes": 0,

            "human_review_pending": sum(
                c.get("review_status") == "PENDING"
                for c in all_cases
            ),

            "human_approved": sum(
                c.get("review_status") == "APPROVE"
                for c in all_cases
            ),

            "worker_tasks_created": len(tasks),
            "after_photos_submitted": len(afters),

            "evidence_reviewed_resolved_not_field_verified": sum(
                c.get("final_review")
                == "RESOLVED_EVIDENCE_REVIEWED"
                for c in all_cases
            ),

            "gps_completeness_percent": (
                round(100 * gps_count / total, 1)
                if total
                else None
            ),

            "task_followup_evidence_percent": (
                round(100 * len(afters) / len(tasks), 1)
                if tasks
                else None
            ),

            "median_submission_to_task_seconds": (
                round(statistics.median(delays), 1)
                if delays
                else None
            ),

            "false_alert_rate": evaluation_status(self),

            "real_field_verification": "NOT PERFORMED",

            "risk_method": (
                "Explainable rule-based inspection priority; "
                "not a trained outbreak forecast"
            ),

            "drone_status": (
                "Simulation/available imagery only; "
                "no autonomous flight claimed"
            )
        }

        return summary, self.table(), self.map_html()

    # --------------------------------------------------------
    # DATA INTEGRITY AUDIT
    # --------------------------------------------------------

    def audit(self):
        cases = self.cases()
        findings = []

        for case in cases:
            case_id = case["case_id"]

            def check(label, passed):
                findings.append({
                    "case_id": case_id,
                    "check": label,
                    "pass": bool(passed)
                })

            before = Path(case["before_path"])

            check(
                "before_image_sha256",
                before.is_file()
                and sha256_file(before)
                == case["before_sha256"]
            )

            overlay = Path(
                case["before_overlay_path"]
            )

            check(
                "before_overlay_sha256",
                overlay.is_file()
                and sha256_file(overlay)
                == case["before_overlay_sha256"]
            )

            check(
                "real_model_provenance_recorded",
                bool(case.get("f2_model"))
                and bool(case.get("f2_weights_sha256"))
                and "FAKE" not in case["f2_model"].upper()
                and "TEST" not in case["f2_model"].upper()
            )

            check(
                "gps_pair_consistent",
                (case.get("latitude") is None)
                == (case.get("longitude") is None)
            )

            check(
                "task_requires_human_approval",
                not case.get("task_id")
                or (
                    case.get("review_status") == "APPROVE"
                    and case.get("reviewed_at_utc") is not None
                )
            )

            check(
                "no_auto_aedes_claim",
                case.get("aedes_species_confirmed") is False
            )

            check(
                "no_real_field_claim",
                case.get("real_field_verified") is False
            )

            check(
                "no_verified_cleanup_success_claim",
                case.get("verified_cleanup_success") is False
            )

            check(
                "explicit_demo_or_field_label",
                case.get("evidence_mode") in {
                    "DEMO_CASE_STAGED",
                    "FIELD_REPORT_UNVERIFIED"
                }
            )

            if case.get("after_path"):
                after = Path(case["after_path"])

                check(
                    "after_image_sha256",
                    after.is_file()
                    and sha256_file(after)
                    == case["after_sha256"]
                )

                after_overlay = Path(
                    case["after_overlay_path"]
                )

                check(
                    "after_overlay_sha256",
                    after_overlay.is_file()
                    and sha256_file(after_overlay)
                    == case["after_overlay_sha256"]
                )

                check(
                    "after_requires_task_and_worker",
                    bool(
                        case.get("task_id")
                        and case.get("assigned_worker")
                    )
                )

                check(
                    "after_never_auto_resolves",
                    case.get("ai_change_assessment")
                    != "RESOLVED"
                )

        passed = sum(
            item["pass"]
            for item in findings
        )

        report = {
            "generated_at_utc": utc_now(),
            "case_count": len(cases),
            "checks_passed": passed,
            "checks_total": len(findings),

            "audit_status": (
                "PASS"
                if findings and passed == len(findings)
                else "NEEDS_LIVE_CASES_OR_REVIEW"
            ),

            "checks": findings,
            "proposal_coverage_percent": None,

            "notes": (
                "Integrity and workflow audit only. "
                "Does not prove dengue reduction, "
                "real cleanup, or full proposal coverage."
            )
        }

        path = self.audit_dir / (
            "Audit_"
            + datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            + "_"
            + uuid.uuid4().hex[:6]
            + ".json"
        )

        path.write_text(
            json.dumps(report, indent=2),
            encoding="utf-8"
        )

        return report, str(path)


# ============================================================
# SAFE CASE SUMMARY
# ============================================================

def safe_case_summary(case):
    keys = (
        "case_id",
        "stage",
        "source",
        "evidence_mode",
        "verified_cleanup_success",
        "gps_provided",
        "latitude",
        "longitude",
        "priority",
        "review_status",
        "task_id",
        "task_status",
        "assigned_worker",
        "recommendation",
        "f2_model",
        "f2_detections",
        "f3_model",
        "f3_detections",
        "ai_change_assessment",
        "alignment",
        "final_review",
        "real_field_verified",
        "aedes_species_confirmed"
    )

    return {
        key: case.get(key)
        for key in keys
    }

# ============================================================
# PART 3 — DRONE ROUTE SIMULATION
# ============================================================

def _distance_xy(a, b):
    return math.hypot(
        float(a[0]) - float(b[0]),
        float(a[1]) - float(b[1])
    )


def _mission_xy_html(rows, center_available=False):
    """Offline XY visualization; all points are simulated."""
    width, height, pad = 760, 410, 45

    xs = [float(r["x_m"]) for r in rows] + [0.0]
    ys = [float(r["y_m"]) for r in rows] + [0.0]

    lo_x, hi_x = min(xs), max(xs)
    lo_y, hi_y = min(ys), max(ys)

    span_x = max(1.0, hi_x - lo_x)
    span_y = max(1.0, hi_y - lo_y)

    def px(x):
        return pad + (float(x) - lo_x) / span_x * (width - 2 * pad)

    def py(y):
        return height - pad - (float(y) - lo_y) / span_y * (height - 2 * pad)

    colors = ["#2563eb", "#d97706", "#16a34a", "#7c3aed"]
    lines = []
    groups = {}

    for r in rows:
        groups.setdefault(
            (r["drone"], r["sortie"]), []
        ).append(r)

    for (drone, sortie), stops in groups.items():
        points = (
            [(0, 0)]
            + [(s["x_m"], s["y_m"]) for s in stops]
            + [(0, 0)]
        )

        coord = " ".join(
            f"{px(x):.1f},{py(y):.1f}"
            for x, y in points
        )

        color = colors[
            (int(drone.split("-")[-1]) - 1) % len(colors)
        ]

        lines.append(
            f'<polyline points="{coord}" fill="none" '
            f'stroke="{color}" stroke-width="2.5" opacity="0.75" />'
        )

    circles = []

    for r in rows:
        color = colors[
            (int(r["drone"].split("-")[-1]) - 1) % len(colors)
        ]

        circles.append(
            f'<circle cx="{px(r["x_m"]):.1f}" '
            f'cy="{py(r["y_m"]):.1f}" r="5" fill="{color}" />'
        )

        circles.append(
            f'<text x="{px(r["x_m"])+6:.1f}" '
            f'y="{py(r["y_m"])-6:.1f}" '
            f'font-size="10" fill="#374151">'
            f'{html.escape(r["waypoint_id"])}</text>'
        )

    circles.append(
        f'<circle cx="{px(0):.1f}" cy="{py(0):.1f}" '
        'r="8" fill="#dc2626" />'
    )

    return (
        '<div style="font:14px sans-serif;padding:10px;'
        'border:1px solid #aaa">'
        '<b>SIMULATED MISSION — NO REAL FLIGHT OR CAPTURED PHOTOS</b>'
        f'<svg viewBox="0 0 {width} {height}" width="100%" '
        'role="img" aria-label="Simulated XY drone routes">'
        + "".join(lines + circles)
        + '</svg><p>Local XY in metres; origin is a simulated base. '
        'No GPS coordinates were invented.</p></div>'
    )


def plan_simulated_drone_mission(
    data_root,
    center_lat=None,
    center_lon=None,
    rows=3,
    cols=4,
    spacing_m=30,
    drones=2,
    speed_mps=5,
    battery_minutes=12,
    dwell_seconds=12,
    manual_revisit_ids=""
):
    """Battery-constrained planned waypoints, not a real flight."""

    lat, lon = check_gps(center_lat, center_lon)

    rows = int(rows)
    cols = int(cols)
    drones = int(drones)

    spacing_m, speed_mps, battery_minutes, dwell_seconds = map(
        float,
        (spacing_m, speed_mps, battery_minutes, dwell_seconds)
    )

    if not (2 <= rows <= 12 and 2 <= cols <= 12 and rows * cols <= 100):
        raise ValueError("Choose 2–12 rows/columns, maximum 100 waypoints.")

    if not (
        1 <= drones <= 4
        and 5 <= spacing_m <= 300
        and 1 <= speed_mps <= 20
        and 3 <= battery_minutes <= 60
        and 0 <= dwell_seconds <= 120
    ):
        raise ValueError("Invalid drone mission parameters.")

    if lat is not None and abs(lat) >= 89:
        raise ValueError("GPS grid unsupported near poles.")

    base_waypoints = []

    for r in range(rows):
        indices = (
            range(cols)
            if r % 2 == 0
            else range(cols - 1, -1, -1)
        )

        for c in indices:
            base_waypoints.append({
                "waypoint_id": f"WP-{len(base_waypoints)+1:03d}",
                "x_m": round((c - (cols - 1) / 2) * spacing_m, 2),
                "y_m": round(((rows - 1) / 2 - r) * spacing_m, 2),
                "visit_type": "PLANNED_GRID_SCAN"
            })

    lookup = {
        w["waypoint_id"]: w
        for w in base_waypoints
    }

    selected = [
        v.strip().upper()
        for v in str(manual_revisit_ids or "").split(",")
        if v.strip()
    ]

    if (
        len(selected) != len(set(selected))
        or any(v not in lookup for v in selected)
    ):
        raise ValueError("Invalid or duplicate revisit waypoint IDs.")

    all_visits = base_waypoints + [
        dict(
            lookup[k],
            visit_type="MANUALLY_REQUESTED_REINSPECTION"
        )
        for k in selected
    ]

    budget_seconds = battery_minutes * 60 * 0.8

    drone_state = [
        {
            "last": (0.0, 0.0),
            "used": 0.0,
            "sortie": 1,
            "sequence": 0
        }
        for _ in range(drones)
    ]

    records = []

    for i, wp in enumerate(all_visits):
        drone_index = i % drones
        state = drone_state[drone_index]
        point = (wp["x_m"], wp["y_m"])

        single_cost = (
            _distance_xy((0, 0), point) * 2 / speed_mps
            + dwell_seconds
        )

        if single_cost > budget_seconds + 1e-8:
            raise ValueError(
                f'{wp["waypoint_id"]} exceeds battery budget.'
            )

        leg = (
            _distance_xy(state["last"], point) / speed_mps
            + dwell_seconds
        )

        return_seconds = (
            _distance_xy(point, (0, 0)) / speed_mps
        )

        if state["used"] + leg + return_seconds > budget_seconds + 1e-8:
            state["sortie"] += 1
            state["last"] = (0.0, 0.0)
            state["used"] = 0.0

            leg = (
                _distance_xy((0, 0), point) / speed_mps
                + dwell_seconds
            )

        state["used"] += leg
        state["last"] = point
        state["sequence"] += 1

        if lat is None:
            gps_lat, gps_lon = None, None
        else:
            gps_lat = round(
                lat + wp["y_m"] / 111320.0, 7
            )
            gps_lon = round(
                lon + wp["x_m"] /
                (111320.0 * math.cos(math.radians(lat))),
                7
            )

        records.append({
            **wp,
            "drone": f"DRONE-{drone_index+1}",
            "sortie": state["sortie"],
            "sequence": state["sequence"],
            "latitude": gps_lat,
            "longitude": gps_lon,
            "coordinate_source": (
                "DERIVED_SIMULATED_FROM_USER_CENTER"
                if lat is not None
                else "SIMULATED_LOCAL_XY_ONLY"
            ),
            "photo_status": "NOT_CAPTURED_SIMULATION",
            "real_flight": False,
            "ai_suspicious_site_confirmed": False
        })

    sorties = []

    for drone, sortie in sorted({
        (r["drone"], r["sortie"])
        for r in records
    }):
        stops = [
            r for r in records
            if r["drone"] == drone and r["sortie"] == sortie
        ]

        pts = (
            [(0.0, 0.0)]
            + [(r["x_m"], r["y_m"]) for r in stops]
            + [(0.0, 0.0)]
        )

        distance = sum(
            _distance_xy(a, b)
            for a, b in zip(pts, pts[1:])
        )

        seconds = (
            distance / speed_mps
            + len(stops) * dwell_seconds
        )

        assert seconds <= budget_seconds + 1e-7

        sorties.append({
            "drone": drone,
            "sortie": sortie,
            "waypoints": len(stops),
            "planned_distance_m": round(distance, 1),
            "planned_duration_s": round(seconds, 1),
            "battery_budget_s_including_20pct_reserve":
                round(budget_seconds, 1)
        })

    summary = {
        "scope": "PLANNED_SIMULATION_ONLY",
        "real_drone_flights": 0,
        "real_photos_captured": 0,
        "waypoint_visits": len(records),
        "unique_waypoints": len(base_waypoints),
        "manually_flagged_revisits": len(selected),
        "drones": drones,
        "sorties": len(sorties),
        "battery_reserve_percent": 20,
        "user_supplied_gps_center": lat is not None,
        "gps_coordinates_are_simulated": lat is not None,
        "not_real_autonomous_capture": True,
        "sortie_details": sorties,
        "created_at_utc": utc_now()
    }

    mission_root = (
        Path(data_root)
        / "drone_simulations"
        / (
            "Mission_"
            + datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
            + "_"
            + uuid.uuid4().hex[:6]
        )
    )

    mission_root.mkdir(parents=True, exist_ok=False)

    csv_path = mission_root / "simulated_waypoints.csv"

    pd.DataFrame(records).to_csv(
        csv_path, index=False
    )

    json_path = mission_root / "mission_summary.json"
    json_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8"
    )

    if lat is None:
        visual = _mission_xy_html(records)
    else:
        m = folium.Map(
            location=[lat, lon],
            zoom_start=17,
            tiles="OpenStreetMap"
        )

        folium.Marker(
            [lat, lon],
            tooltip="USER-ENTERED SIMULATED BASE",
            icon=folium.Icon(color="red")
        ).add_to(m)

        palette = ["blue", "orange", "green", "purple"]

        for drone, sortie in sorted({
            (r["drone"], r["sortie"])
            for r in records
        }):
            stops = [
                r for r in records
                if r["drone"] == drone
                and r["sortie"] == sortie
            ]

            path = (
                [[lat, lon]]
                + [[r["latitude"], r["longitude"]] for r in stops]
                + [[lat, lon]]
            )

            color = palette[
                (int(drone.split("-")[-1]) - 1) % len(palette)
            ]

            folium.PolyLine(
                path,
                color=color,
                weight=3,
                tooltip=f"{drone} sortie {sortie} SIMULATED"
            ).add_to(m)

            for r in stops:
                folium.CircleMarker(
                    [r["latitude"], r["longitude"]],
                    radius=5,
                    color=color,
                    fill=True,
                    tooltip=(
                        f'{r["waypoint_id"]} — '
                        f'{r["visit_type"]} — SIMULATED'
                    )
                ).add_to(m)

        visual = (
            "<b>SIMULATED GPS ROUTE — "
            "no drone telemetry or photo capture</b>"
            + m._repr_html_()
        )

    return (
        summary,
        pd.DataFrame(records),
        visual,
        str(csv_path)
    )


def legacy_mission_evidence(legacy_root):
    """Read-only display of historic F5 simulation."""

    root = Path(legacy_root)

    paths = sorted(
        root.glob(
            "Mission_Planner/F5_Optimized_*/"
            "F5_Optimized_Mission_Assignments.csv"
        )
    )

    if not paths:
        return {
            "status": "NOT FOUND",
            "note": "No historic F5 simulation found."
        }, pd.DataFrame()

    path = paths[-1]
    df = pd.read_csv(path)

    return {
        "status": "READ_ONLY_HISTORIC_SIMULATION",
        "source": str(path),
        "rows": len(df),
        "no_real_drone_flight_claim": True
    }, df.head(50)


# ============================================================
# PART 3 — MODEL EVALUATION
# ============================================================

def _safe_eval_image_path(value, image_root):
    root = Path(image_root).expanduser().resolve()

    candidate = Path(
        str(value).strip()
    ).expanduser()

    if not candidate.is_absolute():
        candidate = root / candidate

    candidate = candidate.resolve()

    if not candidate.is_relative_to(root):
        raise ValueError(
            "Image path must remain inside evaluation root."
        )

    if not candidate.is_file():
        raise FileNotFoundError(
            f"Evaluation image missing: {candidate}"
        )

    if candidate.suffix.lower() not in {
        ".jpg", ".jpeg", ".png", ".webp", ".bmp"
    }:
        raise ValueError(
            "Unsupported evaluation image format."
        )

    return candidate


def evaluate_labelled_image_manifest(
    engine,
    manifest_csv,
    image_root,
    max_images=300,
    allowed_drive_root=None
):
    """Image-level evaluation against human-labelled data."""

    if (
        allowed_drive_root is not None
        and not Path(image_root).expanduser().resolve().is_relative_to(
            Path(allowed_drive_root).resolve()
        )
    ):
        raise ValueError(
            "Evaluation root must be inside allowed drive."
        )

    if not manifest_csv:
        raise ValueError(
            "Human-labelled CSV required."
        )

    path = Path(manifest_csv)

    if not path.is_file() or path.suffix.lower() != ".csv":
        raise ValueError("A CSV file is required.")

    if path.stat().st_size > 2_000_000:
        raise ValueError("CSV exceeds 2 MB.")

    df = pd.read_csv(
        path,
        dtype={"image_path": "string"}
    )

    required = {
        "image_path",
        "ground_truth_present"
    }

    if not required.issubset(df.columns):
        raise ValueError(
            "CSV must contain image_path and ground_truth_present."
        )

    if not (1 <= len(df) <= int(max_images)):
        raise ValueError(
            f"CSV must contain 1–{max_images} images."
        )

    labels = []
    paths = []

    for _, row in df.iterrows():
        label = str(
            row["ground_truth_present"]
        ).strip()

        if label not in {"0", "1"}:
            raise ValueError(
                "Ground truth must be 0 or 1."
            )

        labels.append(int(label))
        paths.append(
            _safe_eval_image_path(
                row["image_path"],
                image_root
            )
        )

    if len(set(paths)) != len(paths):
        raise ValueError(
            "Duplicate evaluation images are not allowed."
        )

    observations = []

    for imgpath, actual in zip(paths, labels):
        with Image.open(imgpath) as img:
            predictions = engine.f2.predict(
                image_rgb(img)
            )

        flagged = bool(predictions)

        prediction = (
            "TP" if actual and flagged
            else "TN" if not actual and not flagged
            else "FP" if flagged
            else "FN"
        )

        observations.append({
            "image_path": str(imgpath),
            "ground_truth_present": actual,
            "ai_alert": int(flagged),
            "detections": len(predictions),
            "max_confidence": max(
                [
                    float(x["confidence"])
                    for x in predictions
                ],
                default=None
            ),
            "prediction": prediction
        })

    counts = {
        key: sum(
            x["prediction"] == key
            for x in observations
        )
        for key in ("TP", "FP", "TN", "FN")
    }

    tp = counts["TP"]
    fp = counts["FP"]
    tn = counts["TN"]
    fn = counts["FN"]

    def safe_ratio(a, b):
        return round(a / b, 5) if b else None

    summary = {
        "status": "EVALUATED_FROM_HUMAN_LABELLED_IMAGES",
        "evaluated_at_utc": utc_now(),
        "sample_size": len(observations),
        "positive_ground_truth": sum(labels),
        "negative_ground_truth": len(labels) - sum(labels),

        **counts,

        "image_level_precision": safe_ratio(tp, tp + fp),
        "image_level_recall": safe_ratio(tp, tp + fn),

        "false_alert_fraction_among_alerts":
            safe_ratio(fp, tp + fp),

        "false_positive_rate_among_negative_images":
            safe_ratio(fp, fp + tn),

        "definitions": {
            "false_alert_fraction_among_alerts":
                "FP / (TP + FP)",
            "false_positive_rate_among_negative_images":
                "FP / (FP + TN)",
            "image_level_precision":
                "TP / (TP + FP)",
            "image_level_recall":
                "TP / (TP + FN)"
        },

        "model_label": engine.f2_label,
        "model_sha256": engine.f2_weights_sha256,
        "manifest_sha256": sha256_file(path),

        "scope": "IMAGE_LEVEL_ONLY_NOT_BOX_PRECISION_RECALL",

        "ground_truth_source":
            "USER_SUPPLIED_LABELS_NOT_INDEPENDENTLY_VERIFIED",

        "warning": (
            "Use held-out human-labelled positive and negative "
            "images to evaluate false alerts."
        )
    }

    folder = (
        engine.root
        / "evaluations"
        / (
            "Eval_"
            + datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
            + "_"
            + uuid.uuid4().hex[:6]
        )
    )

    folder.mkdir(parents=True, exist_ok=False)

    pd.DataFrame(observations).to_csv(
        folder / "per_image_predictions.csv",
        index=False
    )

    report_path = folder / "image_level_metrics.json"

    report_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8"
    )

    return (
        summary,
        pd.DataFrame(observations),
        str(report_path)
    )


def validate_yolo_boxes(
    engine,
    dataset_yaml,
    split="test",
    drive_root=None
):
    """Ultralytics validation on an annotated dataset."""

    if not isinstance(engine.f2, RealYoloDetector):
        raise ValueError(
            "Actual trained YOLO model required."
        )

    yaml_path = Path(
        str(dataset_yaml or "").strip()
    ).expanduser().resolve()

    if (
        drive_root is not None
        and not yaml_path.is_relative_to(
            Path(drive_root).resolve()
        )
    ):
        raise ValueError(
            "Dataset YAML outside allowed drive."
        )

    if (
        not yaml_path.is_file()
        or yaml_path.suffix.lower() not in {".yaml", ".yml"}
    ):
        raise ValueError(
            "Valid dataset YAML required."
        )

    if split not in {"val", "test"}:
        raise ValueError(
            "Split must be val or test."
        )

    import yaml

    dataset = yaml.safe_load(
        yaml_path.read_text(encoding="utf-8")
    )

    if (
        not isinstance(dataset, dict)
        or not dataset.get(split)
    ):
        raise ValueError(
            f"Dataset YAML does not define {split} split."
        )

    output = (
        engine.root
        / "evaluations"
        / "yolo_validation"
    )

    output.mkdir(parents=True, exist_ok=True)

    result = engine.f2.model.val(
        data=str(yaml_path),
        split=split,
        imgsz=640,
        device=engine.f2.device,
        plots=False,
        save_json=False,
        project=str(output),
        name="Validation_" + uuid.uuid4().hex[:8],
        verbose=False
    )

    box = getattr(result, "box", None)

    if box is None:
        raise RuntimeError(
            "YOLO validation returned no box metrics."
        )

    def value(attr):
        x = getattr(box, attr, None)
        return round(float(x), 5) if x is not None else None

    summary = {
        "status": "EVALUATED_FROM_LABELLED_YOLO_SPLIT",
        "split": split,
        "dataset_yaml": str(yaml_path),
        "dataset_yaml_sha256": sha256_file(yaml_path),
        "evaluated_at_utc": utc_now(),

        "box_precision": value("mp"),
        "box_recall": value("mr"),
        "map50": value("map50"),
        "map50_95": value("map"),

        "model_sha256": engine.f2_weights_sha256,

        "false_alert_rate": (
            "NOT DERIVED FROM YOLO mAP; "
            "use labelled image CSV with negative images"
        ),

        "note": (
            "Label quality and held-out split must "
            "be independently confirmed."
        )
    }

    report_path = output / (
        "YOLO_Box_Metrics_"
        + uuid.uuid4().hex[:8]
        + ".json"
    )

    report_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8"
    )

    return summary, str(report_path)


def evaluation_status(engine):
    files = sorted(
        (engine.root / "evaluations").glob(
            "Eval_*/image_level_metrics.json"
        )
    )

    if not files:
        return {
            "status": "NOT EVALUATED",
            "false_alert_rate": None,
            "note": (
                "Upload human-labelled positive "
                "and negative images."
            )
        }

    result = json.loads(
        files[-1].read_text(encoding="utf-8")
    )

    return {
        "status": result["status"],
        "sample_size": result["sample_size"],
        "false_alert_fraction_among_alerts":
            result["false_alert_fraction_among_alerts"],
        "false_positive_rate_among_negative_images":
            result["false_positive_rate_among_negative_images"],
        "image_level_precision":
            result["image_level_precision"],
        "image_level_recall":
            result["image_level_recall"],
        "report_path": str(files[-1])
    }

# ============================================================
# PART 4 — GRADIO FRONTEND + FASTAPI BACKEND
# ============================================================

def build_gradio_app(engine, legacy_root=None):
    import gradio as gr

    legacy_root = Path(legacy_root) if legacy_root else None

    def submit_handler(
        image, source, lat, lon, rain, history,
        closeup, notes, evidence_mode
    ):
        case, overlay = engine.submit(
            image, source, lat, lon, rain, history,
            closeup, notes, evidence_mode=evidence_mode
        )
        return (
            overlay,
            safe_case_summary(case),
            f'Created **{case["case_id"]}** — **{evidence_mode}**. '
            "Next: human review. No worker task until approval; "
            "no verified cleanup claim."
        )

    def inspect_handler(case_id):
        return engine.inspect_case(case_id)

    def review_handler(case_id, reviewer, decision, notes):
        case = engine.review(
            case_id, reviewer, decision, notes
        )
        return safe_case_summary(case)

    def assign_handler(case_id, worker, notes):
        return safe_case_summary(
            engine.assign(case_id, worker, notes)
        )

    def start_handler(case_id, worker):
        return safe_case_summary(
            engine.start(case_id, worker)
        )

    def after_handler(
        case_id, worker, image, notes, attested
    ):
        case, overlay = engine.submit_after(
            case_id, worker, image, notes, attested
        )
        return overlay, safe_case_summary(case)

    def finalize_handler(
        case_id, reviewer, decision, notes
    ):
        return safe_case_summary(
            engine.finalize(
                case_id, reviewer, decision, notes
            )
        )

    def refresh_handler():
        return engine.dashboard()

    def audit_handler():
        return engine.audit()

    def mission_handler(
        lat, lon, rows, cols, spacing, drones,
        speed, battery, dwell, revisits
    ):
        return plan_simulated_drone_mission(
            engine.root, lat, lon, rows, cols,
            spacing, drones, speed, battery,
            dwell, revisits
        )

    def legacy_mission_handler():
        if legacy_root:
            return legacy_mission_evidence(legacy_root)
        return {"status": "NO_LEGACY_ROOT"}, pd.DataFrame()

    def eval_handler(manifest, image_root):
        return evaluate_labelled_image_manifest(
            engine,
            manifest,
            image_root,
            allowed_drive_root=(
                legacy_root.parent if legacy_root else None
            )
        )

    def box_eval_handler(yaml_path, split):
        return validate_yolo_boxes(
            engine,
            yaml_path,
            split,
            drive_root=(
                legacy_root.parent if legacy_root else None
            )
        )

    def eval_status_handler():
        return evaluation_status(engine)

    legacy_note = ""

    if legacy_root and legacy_root.is_dir():
        registry = sorted(legacy_root.glob(
            "Feature_8_Community_Government/"
            "F8_Unified_Data/Integration_*/"
            "F8_Unified_Task_Registry.csv"
        ))

        mission = sorted(legacy_root.glob(
            "Mission_Planner/F5_Optimized_*/"
            "F5_Optimized_Mission_Assignments.csv"
        ))

        legacy_note = (
            "Existing read-only evidence: F8 registry "
            f"{'found' if registry else 'not found'}; "
            "F5 simulation "
            f"{'found' if mission else 'not found'}. "
            "These are not new live cases."
        )

    with gr.Blocks(
        title="DENGU HUNTER | Unified Prototype"
    ) as demo:

        gr.Markdown(
            "# DENGU HUNTER — Unified Hackathon Prototype\n"
            "**Real F2 YOLO inference • Human-reviewed tasks "
            "• GPS map • Before/after evidence**"
        )

        gr.Markdown(
            "**Boundaries:** Inspection-priority rules, "
            "**not dengue outbreak prediction**. "
            "Close-up detections do **not** confirm Aedes species. "
            "**DEMO_CASE_STAGED** and **FIELD_REPORT_UNVERIFIED** "
            "are explicitly labelled; no verified cleanup success "
            "or autonomous flight. Use non-sensitive imagery/GPS "
            "for public demo."
        )

        if legacy_note:
            gr.Markdown(legacy_note)

        # ====================================================
        # TAB 1 — CITIZEN REPORT
        # ====================================================

        with gr.Tab("1. Citizen + Real AI"):
            with gr.Row():
                with gr.Column():
                    photo = gr.Image(
                        label="Original ground/aerial image",
                        type="pil"
                    )

                    closeup = gr.Image(
                        label=(
                            "Optional larvae close-up "
                            "(requires existing F3 model)"
                        ),
                        type="pil"
                    )

                    source = gr.Dropdown(
                        [
                            "Ground photo",
                            "Aerial photo (user-provided)",
                            "Drone simulation frame"
                        ],
                        value="Ground photo",
                        label="Source"
                    )

                    with gr.Row():
                        lat = gr.Number(
                            label="Latitude (optional)",
                            value=None
                        )
                        lon = gr.Number(
                            label="Longitude (optional)",
                            value=None
                        )

                    rain = gr.Dropdown(
                        [
                            "Unknown",
                            "Dry",
                            "Recent rain",
                            "Heavy rain"
                        ],
                        value="Unknown",
                        label="Rain context (user-provided)"
                    )

                    history = gr.Number(
                        label="Prior site reports (user-provided)",
                        value=0,
                        precision=0,
                        minimum=0
                    )

                    notes = gr.Textbox(
                        label="Notes (avoid personal data)",
                        lines=2
                    )

                    evidence_mode = gr.Dropdown(
                        [
                            "DEMO_CASE_STAGED",
                            "FIELD_REPORT_UNVERIFIED"
                        ],
                        value="DEMO_CASE_STAGED",
                        label=(
                            "Evidence mode — distinguish staged "
                            "demo from unverified field report"
                        )
                    )

                    submit = gr.Button(
                        "Run REAL YOLO and submit for review",
                        variant="primary"
                    )

                with gr.Column():
                    annotated = gr.Image(
                        label="Real AI bounding boxes",
                        type="pil",
                        interactive=False
                    )
                    submission = gr.JSON(
                        label="Saved case + AI evidence"
                    )
                    submitted_msg = gr.Markdown()

            submit.click(
                submit_handler,
                [
                    photo, source, lat, lon, rain,
                    history, closeup, notes, evidence_mode
                ],
                [
                    annotated,
                    submission,
                    submitted_msg
                ],
                api_name="submit_report"
            )

        # ====================================================
        # TAB 2 — HUMAN REVIEW
        # ====================================================

        with gr.Tab("2. Human Review"):
            gr.Markdown(
                "All new citizen reports require a named human "
                "decision before a task is created. Approving "
                "a potential site does **not** confirm Aedes larvae."
            )

            review_id = gr.Textbox(
                label="Case ID (from Citizen tab / Dashboard)"
            )

            inspect_btn = gr.Button(
                "Load original + AI evidence before reviewing"
            )

            with gr.Row():
                review_original = gr.Image(
                    label="Original evidence",
                    type="pil",
                    interactive=False
                )
                review_overlay = gr.Image(
                    label="Real YOLO detections",
                    type="pil",
                    interactive=False
                )

            review_details = gr.JSON(
                label="Case evidence and priority"
            )

            def inspect_before(cid):
                original, overlay, _, _, details = (
                    inspect_handler(cid)
                )
                return original, overlay, details

            inspect_btn.click(
                inspect_before,
                [review_id],
                [
                    review_original,
                    review_overlay,
                    review_details
                ],
                api_name="inspect_case"
            )

            reviewer = gr.Textbox(
                label="Reviewer name"
            )

            decision = gr.Dropdown(
                [
                    "APPROVE",
                    "REJECT",
                    "REQUEST_MORE_EVIDENCE"
                ],
                value="APPROVE",
                label="Decision"
            )

            review_notes = gr.Textbox(
                label="Review notes"
            )

            review_btn = gr.Button(
                "Save human decision",
                variant="primary"
            )

            review_result = gr.JSON(
                label="Case status / task creation"
            )

            review_btn.click(
                review_handler,
                [
                    review_id,
                    reviewer,
                    decision,
                    review_notes
                ],
                review_result,
                api_name="review_case"
            )

        # ====================================================
        # TAB 3 — WORKER TASK
        # ====================================================

        with gr.Tab("3. Worker Task"):
            worker_id = gr.Textbox(
                label="Approved Case ID"
            )
            worker_name = gr.Textbox(
                label="Worker name"
            )
            worker_notes = gr.Textbox(
                label="Assignment notes"
            )

            with gr.Row():
                assign_btn = gr.Button(
                    "Assign worker"
                )
                start_btn = gr.Button(
                    "Start task"
                )

            worker_result = gr.JSON(
                label="Worker task status"
            )

            assign_btn.click(
                assign_handler,
                [
                    worker_id,
                    worker_name,
                    worker_notes
                ],
                worker_result,
                api_name="assign_worker"
            )

            start_btn.click(
                start_handler,
                [
                    worker_id,
                    worker_name
                ],
                worker_result,
                api_name="start_task"
            )

        # ====================================================
        # TAB 4 — AFTER PHOTO + VERIFICATION
        # ====================================================

        with gr.Tab("4. After Photo + Verification"):
            after_id = gr.Textbox(
                label="Started Case ID"
            )
            after_worker = gr.Textbox(
                label="Assigned worker name"
            )
            after_photo = gr.Image(
                label="Real after-action photo",
                type="pil"
            )
            after_notes = gr.Textbox(
                label="Work notes"
            )

            attestation = gr.Checkbox(
                label=(
                    "I confirm this is intended to show the "
                    "same site (worker statement, NOT independent proof)"
                )
            )

            after_btn = gr.Button(
                "Run REAL YOLO on after photo and save evidence",
                variant="primary"
            )

            after_overlay = gr.Image(
                label="After image: real YOLO boxes",
                type="pil",
                interactive=False
            )

            after_result = gr.JSON(
                label=(
                    "Alignment, SHA-256 evidence "
                    "and tentative AI comparison"
                )
            )

            after_btn.click(
                after_handler,
                [
                    after_id,
                    after_worker,
                    after_photo,
                    after_notes,
                    attestation
                ],
                [
                    after_overlay,
                    after_result
                ],
                api_name="submit_after_photo"
            )

            gr.Markdown(
                "### Final human evidence review "
                "(never automatic resolution)\n"
                "Staged demo decisions are NOT independently "
                "verified physical cleanup."
            )

            final_id = gr.Textbox(
                label="Case ID for final review"
            )

            final_inspect_btn = gr.Button(
                "Load before / after evidence for human comparison"
            )

            with gr.Row():
                final_before = gr.Image(
                    label="Before photo",
                    type="pil",
                    interactive=False
                )
                final_after = gr.Image(
                    label="After photo",
                    type="pil",
                    interactive=False
                )

            final_details = gr.JSON(
                label="AI comparison + alignment (not proof)"
            )

            def inspect_after(cid):
                original, _, after, _, details = (
                    inspect_handler(cid)
                )
                return original, after, details

            final_inspect_btn.click(
                inspect_after,
                [final_id],
                [
                    final_before,
                    final_after,
                    final_details
                ],
                api_name="inspect_after_evidence"
            )

            final_reviewer = gr.Textbox(
                label="Final reviewer"
            )

            final_decision = gr.Dropdown(
                [
                    "RESOLVED_EVIDENCE_REVIEWED",
                    "UNRESOLVED",
                    "REINSPECTION_REQUIRED"
                ],
                value="REINSPECTION_REQUIRED",
                label="Evidence-based decision"
            )

            final_notes = gr.Textbox(
                label="Why this decision?"
            )

            final_btn = gr.Button(
                "Save final human evidence review"
            )

            final_result = gr.JSON(
                label=(
                    "Final case status "
                    "(not independent field verification)"
                )
            )

            final_btn.click(
                finalize_handler,
                [
                    final_id,
                    final_reviewer,
                    final_decision,
                    final_notes
                ],
                final_result,
                api_name="finalize_case"
            )

        # ====================================================
        # TAB 5 — GPS MAP + DASHBOARD
        # ====================================================

        with gr.Tab("5. GPS Map + Government Dashboard"):
            refresh_btn = gr.Button(
                "Refresh map and dashboard",
                variant="primary"
            )

            stats = gr.JSON(
                label="Measured prototype workflow metrics"
            )

            cases_table = gr.Dataframe(
                label="Unified citizen-to-worker cases",
                interactive=False,
                wrap=True
            )

            gps_map = gr.HTML(
                label="Interactive GPS map — no fabricated pins"
            )

            refresh_btn.click(
                refresh_handler,
                [],
                [
                    stats,
                    cases_table,
                    gps_map
                ],
                api_name="get_dashboard"
            )

            demo.load(
                refresh_handler,
                [],
                [
                    stats,
                    cases_table,
                    gps_map
                ]
            )

            gr.Markdown(
                "Map markers appear only when both "
                "user-provided GPS coordinates exist. "
                "No GPS = no pin. Coordinates are not "
                "independently authenticated."
            )

        # ====================================================
        # TAB 6 — DRONE ROUTE SIMULATION
        # ====================================================

        with gr.Tab("6. Drone Route Simulation"):
            gr.Markdown(
                "### F1 — Battery-aware planned routes, "
                "**SIMULATION ONLY**\n"
                "No real drone flights, telemetry, photographs "
                "or autonomous suspicious-site findings are claimed. "
                "Generated waypoint GPS is simulated."
            )

            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        mission_lat = gr.Number(
                            label="Optional base latitude",
                            value=None
                        )
                        mission_lon = gr.Number(
                            label="Optional base longitude",
                            value=None
                        )

                    with gr.Row():
                        mission_rows = gr.Slider(
                            2, 8,
                            value=3,
                            step=1,
                            label="Grid rows"
                        )
                        mission_cols = gr.Slider(
                            2, 8,
                            value=4,
                            step=1,
                            label="Grid columns"
                        )

                    mission_spacing = gr.Slider(
                        10, 100,
                        value=30,
                        step=5,
                        label="Waypoint spacing (metres)"
                    )

                    mission_drones = gr.Slider(
                        1, 4,
                        value=2,
                        step=1,
                        label="Simulated drones"
                    )

                    mission_speed = gr.Slider(
                        1, 12,
                        value=5,
                        step=1,
                        label="Assumed speed (m/s)"
                    )

                    mission_battery = gr.Slider(
                        3, 30,
                        value=12,
                        step=1,
                        label="Battery duration (minutes); 20% reserve"
                    )

                    mission_dwell = gr.Slider(
                        0, 60,
                        value=12,
                        step=1,
                        label="Camera dwell per waypoint (seconds)"
                    )

                    mission_revisit = gr.Textbox(
                        label=(
                            "Optional manually flagged revisit IDs, "
                            "e.g. WP-001,WP-003"
                        ),
                        value=""
                    )

                    mission_btn = gr.Button(
                        "Simulate planned mission (NO REAL FLIGHT)",
                        variant="primary"
                    )

                with gr.Column():
                    mission_summary = gr.JSON(
                        label="Honest mission summary"
                    )
                    mission_csv = gr.Textbox(
                        label="Saved simulated route CSV",
                        interactive=False
                    )

            mission_table = gr.Dataframe(
                label="Planned waypoints (not real captures)",
                interactive=False
            )

            mission_visual = gr.HTML(
                label="Simulated route map / local XY"
            )

            mission_btn.click(
                mission_handler,
                [
                    mission_lat,
                    mission_lon,
                    mission_rows,
                    mission_cols,
                    mission_spacing,
                    mission_drones,
                    mission_speed,
                    mission_battery,
                    mission_dwell,
                    mission_revisit
                ],
                [
                    mission_summary,
                    mission_table,
                    mission_visual,
                    mission_csv
                ],
                api_name="simulate_drone_mission"
            )

            gr.Markdown(
                "### Earlier F5 drone-planning evidence (read-only)"
            )

            legacy_btn = gr.Button(
                "Load saved mission assignments, if available"
            )

            legacy_status = gr.JSON(
                label="Existing F5 evidence status"
            )

            legacy_table = gr.Dataframe(
                label="Existing simulated mission assignments",
                interactive=False
            )

            legacy_btn.click(
                legacy_mission_handler,
                [],
                [
                    legacy_status,
                    legacy_table
                ],
                api_name="get_legacy_mission"
            )

        # ====================================================
        # TAB 7 — DETECTION EVALUATION
        # ====================================================

        with gr.Tab("7. Detection Evaluation"):
            gr.Markdown(
                "### F2 — Evaluate real YOLO predictions "
                "on human-labelled images\n"
                "No fabricated metrics. Upload a CSV containing "
                "`image_path,ground_truth_present` (0 or 1). "
                "Include held-out positive and negative images."
            )

            gr.Textbox(
                value=(
                    "image_path,ground_truth_present\n"
                    "Evaluation/images/positive_01.jpg,1\n"
                    "Evaluation/images/negative_01.jpg,0"
                ),
                lines=3,
                label="CSV format example (NOT real data)",
                interactive=False
            )

            eval_manifest = gr.File(
                label="Upload human-labelled evaluation CSV",
                file_types=[".csv"],
                type="filepath"
            )

            eval_image_root = gr.Textbox(
                label="Evaluation image root directory",
                value=str(
                    legacy_root.parent
                    if legacy_root
                    else engine.root
                )
            )

            eval_btn = gr.Button(
                "Run REAL YOLO on labelled images",
                variant="primary"
            )

            eval_result = gr.JSON(
                label=(
                    "Confusion matrix, false alerts, "
                    "precision and recall"
                )
            )

            eval_table = gr.Dataframe(
                label="Per-image real predictions",
                interactive=False
            )

            eval_path = gr.Textbox(
                label="Saved evaluation JSON",
                interactive=False
            )

            eval_btn.click(
                eval_handler,
                [
                    eval_manifest,
                    eval_image_root
                ],
                [
                    eval_result,
                    eval_table,
                    eval_path
                ],
                api_name="evaluate_labelled_images"
            )

            gr.Markdown(
                "### Optional YOLO bounding-box precision, "
                "recall and mAP on annotated validation/test split"
            )

            yaml_path = gr.Textbox(
                label="Existing dataset.yaml path",
                value=""
            )

            split_choice = gr.Dropdown(
                ["test", "val"],
                value="test",
                label="Held-out split"
            )

            box_btn = gr.Button(
                "Run official YOLO box validation"
            )

            box_result = gr.JSON(
                label="YOLO precision / recall / mAP"
            )

            box_path = gr.Textbox(
                label="Saved YOLO evaluation JSON",
                interactive=False
            )

            box_btn.click(
                box_eval_handler,
                [
                    yaml_path,
                    split_choice
                ],
                [
                    box_result,
                    box_path
                ],
                api_name="evaluate_yolo_boxes"
            )

            eval_status_btn = gr.Button(
                "Show latest real evaluation status"
            )

            eval_latest = gr.JSON(
                label="Most recent labelled evaluation"
            )

            eval_status_btn.click(
                eval_status_handler,
                [],
                eval_latest,
                api_name="get_evaluation_status"
            )

            gr.Markdown(
                "**Metric definitions:** False-alert fraction "
                "= FP/(TP+FP); false-positive rate = FP/(FP+TN). "
                "Image-level evaluation is not bounding-box mAP. "
                "Software evaluation does not prove fewer dengue cases."
            )

        # ====================================================
        # TAB 8 — EVIDENCE AUDIT
        # ====================================================

        with gr.Tab("8. Evidence Audit"):
            audit_btn = gr.Button(
                "Run integrity and workflow audit"
            )

            audit_result = gr.JSON(
                label=(
                    "Integrity checks; NOT a "
                    "proposal-coverage percentage"
                )
            )

            audit_file = gr.Textbox(
                label="Saved audit JSON path",
                interactive=False
            )

            audit_btn.click(
                audit_handler,
                [],
                [
                    audit_result,
                    audit_file
                ],
                api_name="run_audit"
            )

            gr.Markdown(
                "**Important:** Automated test PASS never "
                "means 100% proposal completion, verified "
                "real cleanup or improved public-health outcomes."
            )

    return demo


# ============================================================
# FASTAPI BACKEND
# ============================================================

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Header,
    UploadFile,
    File,
    Form
)

from pydantic import BaseModel


class ActionBody(BaseModel):
    reviewer: str = ""
    worker: str = ""
    decision: str = ""
    notes: str = ""


def build_fastapi_router(engine, api_token):

    api_token = str(api_token or "").strip()

    if len(api_token) < 24:
        raise ValueError(
            "Use a strong, randomly generated API token "
            "with at least 24 characters."
        )

    router = APIRouter(
        prefix="/api/v1",
        tags=["dengu-hunter"]
    )

    # --------------------------------------------------------
    # BEARER TOKEN AUTHENTICATION — FIXED
    # --------------------------------------------------------

    def authorized(
        authorization: str | None = Header(
            default=None,
            alias="Authorization"
        )
    ):
        auth = str(authorization or "").strip()

        if not auth:
            raise HTTPException(
                status_code=401,
                detail="Authorization header missing"
            )

        scheme, separator, supplied_token = (
            auth.partition(" ")
        )

        if (
            not separator
            or scheme.lower() != "bearer"
            or not supplied_token.strip()
        ):
            raise HTTPException(
                status_code=401,
                detail=(
                    "Invalid authorization format. "
                    "Expected: Bearer YOUR_API_TOKEN"
                )
            )

        if not secrets.compare_digest(
            supplied_token.strip(),
            api_token
        ):
            raise HTTPException(
                status_code=401,
                detail="Invalid API token"
            )

        return True

    # --------------------------------------------------------
    # ERROR HANDLING
    # --------------------------------------------------------

    def convert_error(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)

        except (
            ValueError,
            FileNotFoundError,
            TypeError
        ) as exc:
            raise HTTPException(
                status_code=422,
                detail=str(exc)
            ) from exc

    # --------------------------------------------------------
    # IMAGE UPLOAD VALIDATION
    # --------------------------------------------------------

    async def read_image(upload: UploadFile):
        payload = await upload.read(
            12 * 1024 * 1024 + 1
        )

        if len(payload) > 12 * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail="Image exceeds 12 MB"
            )

        try:
            with Image.open(
                io.BytesIO(payload)
            ) as raw:
                return image_rgb(raw)

        except (
            OSError,
            ValueError,
            Image.DecompressionBombError
        ) as exc:
            raise HTTPException(
                status_code=422,
                detail="Invalid/oversized image"
            ) from exc

    # --------------------------------------------------------
    # HEALTH CHECK
    # --------------------------------------------------------

    @router.get(
        "/health",
        dependencies=[Depends(authorized)]
    )
    def health():
        return {
            "status": "LIVE_AI_BACKEND",
            "model": engine.f2_label,
            "model_sha256": engine.f2_weights_sha256,
            "scope": (
                "breeding-site inspection priority; "
                "not dengue prediction"
            )
        }

    # --------------------------------------------------------
    # GET ALL CASES
    # --------------------------------------------------------

    @router.get(
        "/cases",
        dependencies=[Depends(authorized)]
    )
    def cases():
        return [
            safe_case_summary(case)
            for case in engine.cases()
        ]

    # --------------------------------------------------------
    # GET SINGLE CASE
    # --------------------------------------------------------

    @router.get(
        "/cases/{case_id}",
        dependencies=[Depends(authorized)]
    )
    def case_details(case_id: str):
        return convert_error(
            lambda: safe_case_summary(
                engine.inspect_case(case_id)[-1]
            )
        )

    # --------------------------------------------------------
    # DASHBOARD
    # --------------------------------------------------------

    @router.get(
        "/dashboard",
        dependencies=[Depends(authorized)]
    )
    def dashboard_api():
        summary, table, _ = engine.dashboard()

        return {
            "summary": summary,
            "cases": table.to_dict(
                orient="records"
            )
        }

    # --------------------------------------------------------
    # CITIZEN SUBMISSION
    # --------------------------------------------------------

    @router.post(
        "/cases",
        dependencies=[Depends(authorized)]
    )
    async def submit_api(
        image: UploadFile = File(...),
        evidence_mode: str = Form("DEMO_CASE_STAGED"),
        source: str = Form("Ground photo"),
        latitude: float | None = Form(None),
        longitude: float | None = Form(None),
        rain_context: str = Form("Unknown"),
        previous_reports: int = Form(0),
        notes: str = Form(""),
        closeup: UploadFile | None = File(None)
    ):
        img = await read_image(image)

        close = (
            await read_image(closeup)
            if closeup is not None
            else None
        )

        case, _ = convert_error(
            engine.submit,
            img,
            source,
            latitude,
            longitude,
            rain_context,
            previous_reports,
            close,
            notes,
            evidence_mode=evidence_mode
        )

        return safe_case_summary(case)

    # --------------------------------------------------------
    # HUMAN REVIEW
    # --------------------------------------------------------

    @router.post(
        "/cases/{case_id}/review",
        dependencies=[Depends(authorized)]
    )
    def review_api(
        case_id: str,
        body: ActionBody
    ):
        case = convert_error(
            engine.review,
            case_id,
            body.reviewer,
            body.decision,
            body.notes
        )

        return safe_case_summary(case)

    # --------------------------------------------------------
    # WORKER ASSIGNMENT
    # --------------------------------------------------------
    #@router.post(
      # "/cases/{case_id}/assign",
   # dependencies=[Depends(authorized)]
#)
    
    # --------------------------------------------------------
    # ASSIGN TASK
    # --------------------------------------------------------

    @router.post(
        "/cases/{case_id}/assign",
        dependencies=[Depends(authorized)]
    )
    def assign_api(
        case_id: str,
        body: ActionBody
    ):
        worker = str(body.worker or "").strip()

        if not worker:
            raise HTTPException(
                status_code=422,
                detail="Worker username required"
            )

        case = convert_error(
            engine.assign,
            case_id,
            worker,
            body.notes
        )

        return safe_case_summary(case)

    # --------------------------------------------------------
    # START TASK
    # --------------------------------------------------------

    @router.post(
        "/cases/{case_id}/start",
        dependencies=[Depends(authorized)]
    )
    def start_api(
        case_id: str,
        body: ActionBody
    ):
        case = convert_error(
            engine.start,
            case_id,
            body.worker
        )

        return safe_case_summary(case)

    # --------------------------------------------------------
    # AFTER PHOTO SUBMISSION
    # --------------------------------------------------------

    @router.post(
        "/cases/{case_id}/after",
        dependencies=[Depends(authorized)]
    )
    async def after_api(
        case_id: str,
        image: UploadFile = File(...),
        worker: str = Form(...),
        notes: str = Form(""),
        same_site_attested: bool = Form(False)
    ):
        img = await read_image(image)

        case, _ = convert_error(
            engine.submit_after,
            case_id,
            worker,
            img,
            notes,
            same_site_attested
        )

        return safe_case_summary(case)









    

    
    # --------------------------------------------------------
    # FINAL HUMAN VERIFICATION
    # --------------------------------------------------------

    @router.post(
        "/cases/{case_id}/finalize",
        dependencies=[Depends(authorized)]
    )
    def finalize_api(
        case_id: str,
        body: ActionBody
    ):
        case = convert_error(
            engine.finalize,
            case_id,
            body.reviewer,
            body.decision,
            body.notes
        )

        return safe_case_summary(case)

    @router.get("/drone/mission", dependencies=[Depends(authorized)])
    def drone_mission():
        mission_dir = engine.root / "drone_simulations"

        missions = sorted(
            mission_dir.glob("Mission_*/mission_summary.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )

        if not missions:
            raise HTTPException(
                status_code=404,
                detail="No drone simulation found"
            )

        summary_file = missions[0]
        csv_file = summary_file.parent / "simulated_waypoints.csv"

        if not csv_file.is_file():
            raise HTTPException(
                status_code=404,
                detail="Mission waypoints not found"
            )

        summary = json.loads(
            summary_file.read_text(encoding="utf-8")
        )

        
        df = pd.read_csv(csv_file)
        df = df.astype(object).where(pd.notna(df), None)
        waypoints = df.to_dict(orient="records")


        return {
            "mission_id": summary_file.parent.name,
            "summary": summary,
            "waypoints": waypoints
        }

    return router


# ============================================================
# END OF ENGINE.PY
# ============================================================
