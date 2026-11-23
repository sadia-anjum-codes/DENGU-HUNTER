# 🦟 DENGU HUNTER
## AI-Powered Breeding Site Detection, Risk Prioritization & Verified Mosquito Prevention System

**CoU AI Hackathon 2026**

**Primary Domain:** Healthcare  
**Secondary Domain:** Smart Cities

**Project Repository:** https://github.com/sadia-anjum-codes/DENGU-HUNTER

---

## 1. Project Overview

DENGU HUNTER is an AI-assisted, location-aware mosquito breeding-site surveillance and response platform designed for urban and semi-urban communities in Bangladesh.

The system combines computer vision, explainable risk prioritization, location mapping and a human-verified field response workflow.

Instead of simply detecting potential breeding-site objects, DENGU HUNTER aims to support the complete prevention workflow:

**Detect → Prioritize → Review → Assign → Clean → Verify**

The system does not diagnose dengue or confirm the presence of Aedes mosquito larvae.

## 2. Problem Statement

Dengue prevention in Bangladesh faces several practical challenges:

- Potential mosquito breeding sites are scattered across rooftops, construction areas, discarded tyres and water containers.
- Manual inspection is time-consuming and resource-intensive.
- Community reports may lack accurate locations or supporting evidence.
- Inspection teams need a systematic method for prioritizing reported locations.
- Cleanup activities require reliable follow-up and verification.

Detecting standing water alone does not establish the presence of mosquito larvae. Therefore, an effective system must combine AI assistance with human inspection and evidence-based follow-up.

## 3. Proposed Solution

DENGU HUNTER provides a unified platform where citizens can report potential breeding sites using photographs and available location information.

A YOLO-family object-detection model analyzes submitted images and identifies visible objects associated with possible mosquito breeding environments.

The system uses detection evidence and available contextual information to generate an explainable priority score.

Administrators can review reports and assign inspection tasks to field workers. Workers can submit after-action photographs, allowing administrators to review the intervention evidence and determine the next case status.

The platform supports ground-level photographs and aerial imagery, including clearly labelled simulated drone scenarios.

## 4. Core Features

### AI-Powered Object Detection
- Image upload and analysis
- YOLO-based object detection
- Bounding-box visualization
- Detection confidence scores
- Identification of potential breeding-site indicators

### Explainable Risk Prioritization
- Risk scoring based on available evidence
- Environmental and location context where supplied
- Priority-based inspection support
- Human review before operational decisions

### Location-Based Surveillance
- GPS coordinate support
- Map-based location visualization
- Location-linked reporting
- Clear distinction between real and simulated locations

### Citizen Interface
- Submit potential breeding-site reports
- Upload photographs
- Provide location information
- Track submitted report status

### Admin Interface
- Review submitted reports
- Approve or reject reports
- Assign inspection tasks
- Monitor worker progress
- Review after-action evidence
- Finalize case decisions

### Worker Interface
- View assigned tasks
- Start inspection or cleanup activities
- Upload after-action photographs
- Submit field notes
- Send completed work for administrator review

### Before/After Verification
- Before-action evidence
- After-action evidence
- AI-assisted image comparison
- Human verification
- Reinspection support
- Case status tracking

## 5. System Workflow

1. **Citizen Submission:** A citizen uploads an image and available location information.
2. **AI Detection:** The system analyzes the image using the trained object-detection model.
3. **Risk Prioritization:** Available detection and contextual evidence contribute to a priority score.
4. **Admin Review:** An administrator examines the submitted report.
5. **Task Assignment:** An approved case is assigned to a field worker.
6. **Field Intervention:** The worker inspects or cleans the reported location.
7. **Evidence Submission:** The worker uploads an after-action photograph and notes.
8. **Final Verification:** The administrator reviews the evidence and determines whether the case is resolved or requires further action.
9. **Status Tracking:** The updated case status becomes available through the reporting workflow.

## 6. Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Backend Framework | FastAPI |
| Frontend | HTML, CSS, JavaScript |
| Computer Vision | YOLO-family model |
| Image Processing | OpenCV |
| API Communication | HTTP / REST |
| Location Visualization | Map integration |
| Version Control | Git and GitHub |
| Local Development | Windows, Python |

**Model details to verify:** Exact YOLO version, checkpoint, training configuration and model classes.

## 7. Project Structure

The repository is organized around the following components:

```text

DENGU-HUNTER/
├── backend/
│   ├── main.py
│   ├── engine.py
│   ├── evaluate_correct.py
│   ├── evaluate_fixed.py
│   ├── evaluate_model.py
│   ├── requirements.txt
│   ├── setup_local.py
│   └── README.md
├── frontend/
│   ├── index.html
│   ├── roles.html
│   ├── app.js
│   ├── styles.css
│   └── README.md
├── models/
│   ├── best.pt.zip
│   └── data/
│       └── README.md
├── results/
│   ├── association_results.csv
│   ├── closed_loop_summary.json
│   ├── image_inspection_summary.csv
│   └── intervention_register.csv
├── docs/
│   └── README.md
└── README.md
  
```

This structure represents the intended organization. Verify the actual GitHub repository paths after the backend upload.

## 8. Local Installation and Setup

### Requirements

- Python
- Required backend dependencies
- Trained model checkpoint
- Modern web browser

### Step 1: Clone the Repository

```bash
git clone https://github.com/sadia-anjum-codes/DENGU-HUNTER.git
cd DENGU-HUNTER
```

### Step 2: Set Up the Backend

Open a PowerShell terminal inside the backend directory.

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Configure any required environment variables using `.env.example`.

### Step 3: Start the Backend

```powershell
python -m uvicorn main:app --host 127.0.0.1 --port 8003 --reload
```

API documentation:

http://127.0.0.1:8003/docs

### Step 4: Start the Frontend

Open another terminal in the repository directory.

```powershell
cd frontend
python -m http.server 5173
```

### Step 5: Open the Application

Main Dashboard:

http://127.0.0.1:5173/index.html

Role-Based Interface:

http://127.0.0.1:5173/roles.html

**Important:** Keep both frontend and backend servers running. Ensure the frontend API configuration points to the backend on port `8003`.

These instructions assume the backend entry point is `main:app` and all required model files and dependencies are available.

## 9. Demo Workflow

The demonstration is designed to show the complete application workflow.

1. Open the DENGU HUNTER application.
2. Submit a report from the Citizen interface.
3. Upload an image for actual AI inference.
4. Display detections, confidence values and risk priority.
5. Show available GPS/map information.
6. Review the report through the Admin interface.
7. Assign an inspection task.
8. Open the Worker interface.
9. Submit after-action evidence.
10. Perform final administrator review.
11. Display the updated case status.

Demo evidence should distinguish real detections and locations from simulated examples.


### Demo Video

Watch the complete DENGU HUNTER project demonstration:

**[Watch Demo Video](https://drive.google.com/file/d/1Ar74C2fg-yJ8wtOArBrOaXXoy73IbuTo/view?usp=sharing)**

The video demonstrates the AI detection and human-reviewed inspection workflow.

## 10. Model Evaluation

The object-detection model should be evaluated using a separate validation or test dataset.

Relevant evaluation measures include:

- Precision
- Recall
- mAP@0.5
- mAP@0.5:0.95
- False-positive examples
- False-negative examples
- Inference performance

**TODO:** Insert actual measured values, evaluation dataset details and supporting result files. No evaluation scores are claimed here without verified evidence.

## 11. Dataset Information

The project uses image data for identifying potential mosquito breeding-site indicators.

Before final submission, document:

- Dataset name and source
- Dataset licensing or usage permissions
- Number of training, validation and test images
- Object classes
- Annotation format
- Preprocessing and augmentation
- Dataset limitations
- Any self-collected images

### Dataset Download

The breeding-site detection dataset is hosted on Google Drive due to its large file size (approximately 450 MB).

**Download Dataset:** [breeding.zip](https://drive.google.com/file/d/1s5qB3dOgLuc-pBCooqBsVRjkW_82D2Nr/view?usp=sharing)

The ZIP archive contains the dataset files used for the project. Dataset access is provided for research, evaluation, and reproducibility, subject to the dataset's applicable permissions and licensing.

**TODO:** Add verified dataset information and attribution.
## 12. AI Tools and Disclosure

### 12.1 AI Model and Framework

- **Primary AI Task:** Potential mosquito breeding-site object detection
- **Model Architecture:** YOLOv8 Small (YOLOv8s), as specified in the original training code
- **Framework:** Ultralytics YOLO
- **Programming Language:** Python
- **Training Environment:** Google Colab
- **GPU:** CUDA GPU for training; T4 supported for inference
- **CPU Support:** Available for inference
- **Primary Trained Model:** F2 `best.pt`
- **Optional Secondary Model:** F3 larvae close-up detection
- **Backend:** FastAPI
- **Interactive Demo:** Gradio
- **Image Processing:** OpenCV, Pillow
- **Mapping:** Folium

### 12.2 Dataset Information

**Primary Dataset:** Breeding Place Detection

The original training configuration specifies five object classes:

1. Bottle
2. Coconut-Exocarp
3. Drain-Inlet
4. Tire
5. Vase

**Annotation Format:** YOLO bounding-box annotations

**Dataset Splits:** Training, Validation and Testing

**Dataset Configuration:** `data.yaml`

**Dataset Image Counts and Original Source:** Pending verification from dataset records.

A separate larvae detection dataset is referenced for the optional F3 model.

### 12.3 Model Training Configuration

The archived F2 training code specifies:

| Parameter | Configuration |
|---|---|
| Base Model | `yolov8s.pt` |
| Epochs | 50 |
| Image Size | 640 × 640 |
| Batch Size | 16 |
| Optimizer | AdamW |
| Learning Rate | 0.001 |
| Early Stopping Patience | 15 |
| Random Seed | 42 |
| Workers | 2 |
| Deterministic Training | Enabled |

**Training Output:** `DENGUE_HUNTER_RESULTS/DengueHunter_v1/weights/best.pt`

These settings are taken from the preserved original training code. Final training execution details and checkpoint metadata should be cross-checked with the saved training artifacts.

### 12.4 Model Evaluation

The notebook includes independent test evaluation code using:

- Precision
- Recall
- mAP@0.5
- mAP@0.5:0.95

Actual numerical results must be reported from saved evaluation outputs. No unverified performance scores are claimed.

### 12.5 AI-Assisted Development

The project uses Ultralytics YOLO for object detection and rule-based logic for explainable inspection prioritization.

Any AI coding assistants, third-party AI APIs, or AI-generated assets used during development must be disclosed according to the team's actual development records.

### 12.6 Human Contributions and Verification

The project team is responsible for dataset preparation, model training, frontend and backend development, integration, testing, documentation, and verification.

The system follows a human-reviewed workflow:

**Citizen Submission → AI Detection → Risk Prioritization → Human Review → Worker Assignment → Before/After Evidence → Final Human Decision**

### 12.7 Responsible AI and Limitations

- AI object detection does not confirm Aedes mosquito larvae or dengue infection.
- Risk scores represent rule-based inspection priorities, not predicted dengue incidence.
- Drone-route functionality is simulated; autonomous drone flight is not claimed.
- Before/after image analysis does not independently prove successful cleanup.
- Real field verification and end-to-end live testing remain necessary.
- AI findings support human decisions rather than replacing them.
  
## 13. Team Members and Contributions

| Team Member | Contribution |
|---|---|
| **Sadia Anjum** | AI/ML model development, dataset preparation, model training and testing |
| **Marjahan Begum Urmi** | Frontend development, backend development and integration |
| **Sadia Afreen Jinia** | GitHub repository management, documentation, project coordination and submission preparation |
## 14. Testing

### 14.1 Software and Workflow Testing

The project includes software-level testing for:

- Image input validation
- GPS coordinate validation
- Rule-based inspection priority
- Citizen report submission
- Human review and approval workflow
- Worker task assignment
- Before/after evidence handling
- Case audit and data integrity checks

### 14.2 AI Model Testing

The project provides an evaluation workflow for the trained YOLO model using:

- Precision
- Recall
- mAP@0.5
- mAP@0.5:0.95

Actual evaluation results must be confirmed from the saved model outputs.

### 14.3 Testing Limitations

Static and synthetic software tests do not establish successful live YOLO inference or field validation.

The following require additional verification:

- Real-image inference using the trained F2 checkpoint
- Complete end-to-end execution in Google Colab
- False-positive and false-negative evaluation
- Real-world field inspection
- Cleanup effectiveness verification

### 14.4 Testing Status

DENGU HUNTER includes testing and evaluation code, but complete live demonstration readiness and real-world effectiveness have not yet been independently established.
## 15. Expected Impact

DENGU HUNTER aims to:

- Support earlier identification of potential breeding sites
- Improve prioritization of inspection tasks
- Help coordinate community reporting and field response
- Improve evidence collection during cleanup
- Support location-aware monitoring
- Strengthen accountability through human-reviewed case closure

The prototype does not establish that dengue infections or outbreaks have been reduced. Such impact requires real-world field evaluation and public-health data.
## 16. Limitations

DENGU HUNTER is an AI-assisted prototype for identifying potential mosquito breeding-site indicators and prioritizing human inspections.

### Current Limitations

1. **Detection Accuracy:** YOLO predictions may contain false positives and false negatives.

2. **Image Quality:** Poor lighting, blurry images, occlusion, and camera angles may affect detection accuracy.

3. **Species Identification:** The system cannot independently confirm Aedes mosquito species or dengue infection.

4. **Risk Assessment:** Inspection priority is calculated using explainable rules, not a trained dengue outbreak prediction model.

5. **Field Verification:** Real-world inspection and cleanup effectiveness have not yet been independently validated.

6. **Drone Integration:** Drone-route planning is a simulation. Autonomous drone flights and real aerial image capture are not claimed.

7. **Before/After Verification:** Image comparisons provide supporting evidence but cannot independently prove successful cleanup.

8. **Model Evaluation:** Final performance metrics require verification using the saved evaluation results and independent test data.

9. **Deployment:** Complete end-to-end testing and production-level deployment validation remain necessary.

### Future Improvements

- Conduct real-world field testing.
- Evaluate false positives and false negatives.
- Improve detection robustness across different environments.
- Expand and validate the dataset.
- Strengthen role-based authentication and permissions.
- Integrate verified field inspection and cleanup records.

The system is intended to support public-health inspection decisions, not replace professional assessment or field verification.
