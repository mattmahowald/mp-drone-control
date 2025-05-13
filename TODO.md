# 🧠 TODO.md — Hand Gesture Recognition (CS231n Project)

## 📅 Project Timeline

- **Proposal Submitted:** ✅ April 25
- **Milestone Due:** 📍 May 16
- **Final Report Due:** 📍 June 4
- **Poster + Code Submission:** 📍 June 11

## ✅ Out of Scope

- ❌ Drone integration or control
- ❌ Non-visual modalities (e.g. voice or text-based commands)
- ❌ Deployment to hardware beyond a basic demo

---

## 🗂️ 1. Download & Create Template

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

## 📚 2. GPT-Aided Literature Review

### Tasks

- [ ] Prompt GPT for recent real-time gesture recognition papers
- [ ] Manually read and summarize:
  - [ ] MediaPipe Hands + documentation
  - [ ] MobileNetV2 and EfficientNet
  - [ ] Relevant arXiv papers on gesture recognition
- [ ] Collect references and insert into `lit_review.bib`
- [ ] Write the "Related Work" section for milestone paper

---

## 📦 3. Get Dataset Processed and Repo into GitHub

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
- [ ] Split dataset into train/val/test splits (see Task 6)
- [ ] Commit preprocessing pipeline and dataset loader

---

## ✋ 4. Determine Hand Signals to Use

### Tasks

- [ ] Define 5–8 unique hand gestures (not for drone control)
  - e.g., "thumbs up", "peace", "stop", "fist", "open palm", "OK", etc.
- [ ] Sketch or source images for each
- [ ] Write clear labels for annotation
- [ ] Add documentation on gesture class definitions

---

## 🧠 5. Draft a Model Recognizing Hand Signals

### Tasks

- [ ] Implement a basic pipeline:
  - [ ] Use MediaPipe to extract 21 hand landmarks (x,y,z)
  - [ ] Build classifier on top of flattened keypoint vectors
  - [ ] Start with simple MLP
- [ ] Implement MobileNetV2 or EfficientNet classifier (on keypoints or image patches)
- [ ] Train baseline model
- [ ] Log metrics: accuracy, precision, recall, F1

---

## 🧪 6. Split Dataset and Document Results

### Tasks

- [ ] Ensure representative stratified splits (train / val / test)
  - [ ] Validate across participants if custom dataset is collected
- [ ] Document class balance and preprocessing steps
- [ ] Write data section of milestone report

---

## 📹 7. Get Code Detecting Hand Signals via Live Video

### Tasks

- [ ] Use MediaPipe in real-time video capture (OpenCV)
- [ ] Integrate trained model inference
  - [ ] Real-time prediction with visualization (overlay gesture label on screen)
- [ ] Measure latency and FPS
- [ ] Write scripts to demo prediction pipeline
- [ ] (Optional) Export to TensorFlow Lite

---

## 📈 Evaluation & Reporting

### Quantitative

- [ ] Confusion matrix
- [ ] Accuracy, Precision, Recall, F1-score
- [ ] FPS (real-time inference)

### Qualitative

- [ ] Screenshots or short video demos
- [ ] Failure case analysis

---

## 🧾 Final Report Checklist (Due June 4)

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

## 📌 Notes

- Cite all third-party code, including MediaPipe and MobileNet sources
- Update this file weekly with progress (or move to `docs/` directory with changelog)
