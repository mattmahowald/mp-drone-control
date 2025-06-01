# TODO.md — Hand Gesture Recognition (CS231n Project)

## Project Timeline

- **Proposal Submitted:** April 25
- **Milestone Due:** May 16
- **Final Report Due:** June 4
- **Poster + Code Submission:** June 11

## Out of Scope for Milestone

- Drone integration or control
- Deployment to hardware beyond a basic demo

---

## 1. Download & Create Template

### Tasks

- [ ] Download CVPR 2017 paper template (required for Milestone & Final Report)
  - [ ] Link: https://cvpr2017.thecvf.com/files/cvpr2017AuthorKit.zip
- [ ] Set up Overleaf project or local LaTeX environment
- [ ] Create a `report/` folder to store drafts
- [ ] Write initial headers and outline for milestone report
  - [ ] Introduction
  - [ ] Problem Statement
  - [ ] Technical Approach
  - [ ] Preliminary Results

---

## 2. GPT-Aided Literature Review

### Tasks

- [ ] Prompt GPT for recent real-time gesture recognition papers
- [ ] Manually read and summarize:
  - [ ] MediaPipe Hands + documentation
  - [ ] MobileNetV2 and EfficientNet
  - [ ] Relevant arXiv papers on gesture recognition
- [ ] Collect references and insert into `lit_review.bib`
- [ ] Write the "Related Work" section for milestone paper

---

## 3. Get Dataset Processed and Repo into GitHub

### Tasks

- [x] Set up project GitHub repo with standard Python project structure
  - [x] Created Poetry-based Python project structure
  - [x] Added comprehensive README.md
  - [x] Added Makefile for common development tasks
  - [x] Set up development environment with Poetry
- [x] Download ASL Hands dataset (or write script to extract landmarks)
- [x] Create Python pipeline to:
  - [x] Load raw keypoints
  - [x] Label samples by hand signal
  - [x] Normalize and augment data (optional)
- [x] Split dataset into train/val/test splits (see Task 6)
- [x] Commit preprocessing pipeline and dataset loader

---

## 4. Determine Hand Signals to Use

### Tasks

- [ ] TBD

---

## 5. Draft a Model Recognizing Hand Signals

### Tasks

- [x] Implement a basic pipeline:
  - [x] Use MediaPipe to extract 21 hand landmarks (x,y,z)
  - [x] Build classifier on top of flattened keypoint vectors
  - [x] Start with simple MLP
- [x] Implement MobileNetV2 or EfficientNet classifier (on keypoints or image patches)
- [x] Train baseline model
- [x] Log metrics: accuracy, precision, recall, F1

---

## 6. Split Dataset and Document Results

### Tasks

- [x] Ensure representative stratified splits (train / val / test)
  - [ ] Validate across participants if custom dataset is collected
- [ ] Document class balance and preprocessing steps
- [ ] Write data section of milestone report

---

## 7. Get Code Detecting Hand Signals via Live Video

### Tasks

- [x] Use MediaPipe in real-time video capture (OpenCV)
- [x] Integrate trained model inference
  - [x] Real-time prediction with visualization (overlay gesture label on screen)
- [x] Measure latency and FPS
- [x] Write scripts to demo prediction pipeline

---

## Evaluation & Reporting

### Quantitative

- [ ] Confusion matrix
- [ ] Accuracy, Precision, Recall, F1-score
- [ ] FPS (real-time inference)

### Qualitative

- [ ] Screenshots or short video demos
- [ ] Failure case analysis

---

## Model Size Comparison & Performance Discussion

### Experimental Design

- [ ] Define "small" and "large" model architectures (parameter count, type, resource constraints)
- [ ] Document model definitions in the report methods section

### Evaluation Criteria

- [ ] Select and document performance metrics (accuracy, F1, precision, recall, inference time, model size, memory usage)
- [ ] Ensure identical dataset splits and preprocessing for both models
- [ ] Plan and document statistical significance testing (e.g., paired t-test, confidence intervals)

### Implementation & Training

- [x] Implement both small and large models in `src/mp_drone_control/models/`
- [x] Use consistent training hyperparameters and procedures
- [x] Set random seeds and document all settings for reproducibility

### Evaluation & Analysis

- [x] Evaluate both models on the same test set and collect all metrics
- [ ] Compare inference speed, resource usage, and model file size
- [ ] Generate confusion matrices and error analysis (if relevant)
- [ ] Perform statistical analysis and report confidence intervals

### Reporting

- [x] Write a dedicated section in the report comparing small and large models
- [ ] Present results in tables/plots with clear discussion of trade-offs
- [ ] Discuss where the small model matches or outperforms the large model, and any limitations
- [ ] Document all code, seeds, and environment details for reproducibility

### Performance Discussion

- [x] Write a performance discussion section interpreting results, trade-offs, and implications for deployment or future work

### Additional Tasks

- [ ] Export both models to ONNX for mobile deployment
- [ ] Benchmark both models on a mobile device or emulator

# NOTE: Training and evaluation are now automated via scripts/train.py and scripts/evaluate.py. Model comparison summary is written to models/model_comparison.txt.

---

## Final Report Checklist (Due June 4)

- [ ] Title, Author(s)
- [ ] Abstract
- [ ] Introduction
- [ ] Related Work
- [ ] Data
- [ ] Methods
- [ ] Experiments
- [ ] Conclusion
- [ ] References
- [ ] Appendix (optional): additional figures or tables
- [ ] Supplementary: code + demo materials (.zip)

---

## Notes

- Cite all third-party code, including MediaPipe and MobileNet sources
- Update this file weekly with progress (or move to `docs/` directory with changelog)
- The data split logic is now reusable and tested in `src/mp_drone_control/data/preprocess.py`.
