# DENGUE HUNTER
## AI-Powered Breeding Site Detection, Risk Prediction & Verified Mosquito Prevention System

**CoU AI Hackathon 2026**  
**Primary domain:** Healthcare | **Secondary domain:** Smart Cities

> Dengue Hunter is an AI-assisted, location-aware mosquito-breeding-site surveillance and response platform designed for urban and semi-urban communities in Bangladesh.

## Project overview
Dengue prevention in densely populated Bangladeshi communities is challenged by the difficulty of finding and repeatedly inspecting scattered mosquito breeding sites. Potential sites may exist on rooftops, in uncovered water containers, discarded tyres, construction areas and other places that are difficult for inspection teams to monitor continuously. Manual inspection is time-consuming, while public reports may be incomplete, delayed or lack precise location information.

A further problem is that identifying standing water does not confirm the presence of Aedes larvae. Inspection teams therefore need a way to prioritize locations for closer verification. Dengue Hunter addresses this operational gap through AI-assisted, location-aware surveillance and follow-up—not by claiming to diagnose dengue or guarantee outbreak prediction.

## Proposed solution
Users can submit ground-level photos. Drone-style aerial images or available drone footage can be used to inspect rooftops and hard-to-reach areas. A computer-vision model analyzes uploaded images to detect visible standing-water containers and other potential breeding-site indicators.

When GPS data is available, each detection can be linked to a location. An explainable risk score prioritizes cases using visual evidence, site type, environmental context and previous inspection records. The dashboard displays flagged locations on a map and supports an inspection or cleanup task for a responsible field worker. A field worker can upload an after-action photo; the system compares before-and-after evidence and records the case as resolved, unresolved or requiring reinspection. Human verification remains part of the workflow, especially where larvae identification or intervention decisions are needed.

## Core demonstration workflow
1. Upload an aerial or ground-level image.
2. Run actual AI inference.
3. Show potential breeding-site indicators with bounding boxes and confidence scores.
4. Display a map location when coordinates are available; otherwise label the location as simulated.
5. Generate an explainable priority score and create an inspection task.
6. Upload an after-action image and demonstrate before/after case status.

## Technology
The team has supplied the following implementation stack: **Python, OpenCV, HTML, CSS, JavaScript and FastAPI**, with a **YOLO-family model**.

The proposal also describes:
- Computer vision using Python, OpenCV and a YOLO-family object-detection model.
- An explainable, initially rule-based risk-prioritization layer.
- GPS-based map display using Leaflet or an equivalent mapping library.
- FastAPI or Streamlit as the application layer; the team has identified FastAPI for its current stack.
- Properly credited public or self-collected images, with dataset limitations documented.
- Available aerial imagery/video or a clearly labelled drone simulation; no new drone hardware is required.

**Not yet specified by the team:** exact YOLO version/checkpoint, dataset name and source, map library/provider, dependency versions, API routes, and measured evaluation results. These are intentionally left for verification rather than invented.

## Team
| Member | Contribution |
|---|---|
| **Sadia Anjum** | AI/ML model development, dataset preparation, model training and testing |
| **Marjahan Begum Urmi** | Frontend development, backend development and integration |
| **Sadia Afreen Jinia** | GitHub repository management, documentation, project coordination and submission preparation |

See [`docs/TEAM_CONTRIBUTIONS.md`](docs/TEAM_CONTRIBUTIONS.md).

## Repository documentation
- [Project overview](docs/PROJECT_OVERVIEW.md)
- [Problem statement](docs/PROBLEM_STATEMENT.md)
- [System architecture](docs/SYSTEM_ARCHITECTURE.md)
- [Workflow](docs/WORKFLOW.md)
- [Features and status](docs/FEATURES.md)
- [Technology stack](docs/TECHNOLOGY_STACK.md)
- [Team contributions](docs/TEAM_CONTRIBUTIONS.md)
- [AI disclosure](docs/AI_DISCLOSURE.md)
- [Dataset information](docs/DATASET_INFORMATION.md)
- [Model evaluation](docs/MODEL_EVALUATION.md)
- [Local setup](docs/LOCAL_SETUP.md)
- [API documentation](docs/API_DOCUMENTATION.md)
- [Testing report](docs/TESTING_REPORT.md)
- [Limitations and future scope](docs/LIMITATIONS_AND_FUTURE_SCOPE.md)
- [Demo script](docs/DEMO_SCRIPT.md)
- [References](docs/REFERENCES.md)
- [Final submission checklist](docs/FINAL_SUBMISSION_CHECKLIST.md)

## Current status
**Project status supplied by the team: In progress.** The feature list below reflects intended project scope, not a claim that all features are currently working. Update the status only after testing the actual application.

| Feature | Current status |
|---|---|
| Image detection | In progress — verify live inference |
| Risk prediction/prioritization | In progress — verify actual scoring logic |
| Map | In progress — verify map and GPS/simulated-location handling |
| Inspection task | In progress — verify task creation and tracking |
| Before/after verification | In progress — verify image comparison and status flow |
| Full local prototype | In progress — complete end-to-end test required |

## Expected impact and measurement
Dengue Hunter aims to help community response teams and local authorities find and prioritize potential mosquito breeding sites more efficiently. Suggested measures are object-detection precision and recall on a held-out image set, false-alert rate, location completeness, time from image submission to task creation, and the proportion of intervention cases with follow-up evidence.

These measures evaluate the software workflow; they do not prove a reduction in dengue cases. Any public-health impact would require a longer field evaluation with local health authorities and reliable epidemiological data.

## Responsible use and limitations
- The system identifies potential breeding-site indicators; it does not diagnose dengue.
- It does not confirm Aedes larvae from distant aerial images or predict an individual's infection.
- Larvae confirmation and chemical-control decisions require suitable close-up evidence and qualified human/public-health oversight.
- The hackathon prototype will not perform autonomous chemical spraying.
- Drone flight may be represented through available aerial footage or a clearly labelled simulation; no new drone hardware is required.
- Real drone operation in communities must follow applicable aviation, privacy and safety requirements.

## Disclosures and credits
All AI models, APIs, datasets, libraries, coding assistants and third-party assets used must be disclosed. See [AI disclosure](docs/AI_DISCLOSURE.md), [dataset information](docs/DATASET_INFORMATION.md) and [references](docs/REFERENCES.md). Complete every marked placeholder before final submission.

## Links
- **Public GitHub repository:** [Insert repository URL]
- **Demo video (if used):** [Insert URL]
- **Live deployment (if available):** [Insert URL or write “Not deployed”]
- **Dataset source:** [Insert verified URL]

## Proposal
The original team proposal is included as [`docs/Dengue_Hunter_Hackathon_Proposal.docx`](docs/Dengue_Hunter_Hackathon_Proposal.docx). This README is based on that proposal and the project details supplied by the team.
