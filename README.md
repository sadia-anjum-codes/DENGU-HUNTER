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

**TODO:** Add verified dataset information and attribution.

## 12. AI Tools and Disclosure

The project uses a YOLO-family computer-vision model for object detection.

The final AI disclosure must identify:

- Exact model and checkpoint
- Model training framework
- Dataset sources
- Third-party AI APIs, if used
- AI coding assistants used during development
- AI-generated assets, if any
- Human contributions and verification

**TODO:** Complete the AI disclosure using the team's actual development records.

## 13. Team Members and Contributions

| Team Member | Contribution |
|---|---|
| **Sadia Anjum** | AI/ML model development, dataset preparation, model training and testing |
| **Marjahan Begum Urmi** | Frontend development, backend development and integration |
| **Sadia Afreen Jinia** | GitHub repository management, documentation, project coordination and submission preparation |

## 14. Testing and Verification

The team has reported development and testing activities involving:

- Image detection
- Risk prioritization
- GPS/map visualization
- Role-based interfaces
- Administrator review
- Worker assignment
- Before/after evidence submission
- Case status updates
- Model evaluation
- Role and permission checks

**Verification status:** These activities are team-reported. The final testing report should include reproducible test cases, actual results, screenshots and any known failures. Only independently reproduced or documented successful tests should be marked as passed.

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

- Object detection does not confirm Aedes larvae or dengue infection.
- AI predictions may contain false positives and false negatives.
- Image quality and environmental
