# Skin Disease AI Decision Support – Project Summary (PBL Submission)

Audience: 
- Primary: Professor of AI / Academic Evaluator (demonstrates applied ML lifecycle & rigor)
- Secondary: Non‑AI / Non‑IT Stakeholders (explains purpose, value, and safe limitations in plain language)

---
## 1. What Is This Application?
A research-grade prototype that classifies dermatoscopic skin lesion images into 7 clinically relevant categories (e.g., Melanoma, Benign Nevus) and provides visual explanations (Grad‑CAM heatmaps) plus calibration quality indicators. It is NOT a medical device; it is an educational / exploratory system.

Plain Language: You upload a close-up skin lesion image (from a medical dermatoscope). The system predicts which type it might be and highlights the region it used to decide, helping a future doctor-in-the-loop make a more informed judgment.

---
## 2. How It Runs (High-Level Flow)
1. Image Loaded → resized to 224×224 and normalized (ImageNet stats).
2. Model (EfficientNet_B0 backbone) produces class logits.
3. Softmax converts logits to probabilities (calibrated quality assessed via ECE + Brier score).
4. Grad-CAM generates an attention heatmap overlay for interpretability.
5. Output bundle: Top class prediction, per-class probabilities, heatmap, latency metrics (for profiling only in this prototype).

---
## 3. Resources & Environment
| Resource | Detail |
|----------|--------|
| Dataset | HAM10000 (10k dermatoscopic images) |
| Hardware (training) | Single GPU (RTX 4060 Laptop GPU) |
| Framework | PyTorch (2.7.1+cu118) |
| Image Size | 224×224 |
| Batch Size (prototype) | 32 |
| Precision | Mixed (AMP) |
| Storage Artifacts | `models/`, `runs/`, `docs/images/` |

---
## 4. Tech Stack
| Layer | Tool | Rationale |
|-------|------|-----------|
| Core DL | PyTorch | Flexible + AMP support |
| Vision Models | torchvision EfficientNet_B0 | Strong accuracy/compute trade-off |
| Data Augmentation | (basic transforms; Albumentations planned) | Improve generalization |
| Visualization | Matplotlib / Seaborn | Reproducible figures |
| Explainability | Custom Grad-CAM | Clinical interpretability baseline |
| Calibration Metrics | Custom ECE + Brier script | Reliability assessment |
| Automation | GitHub Actions | CI reproducibility |

---
## 5. Algorithms & Evolution Path
| Stage | Approach | Reason | Potential Next Step |
|-------|----------|--------|---------------------|
| Initial Baseline | Pretrained EfficientNet_B0 + linear head | Small, efficient, good start | Hyperparameter tuning / longer training |
| Explainability | Grad-CAM over final conv layer | Fast, widely accepted in medical imaging | Add SHAP, LIME, counterfactuals |
| Calibration | Raw softmax probs + post-hoc ECE/Brier evaluation | Measure trustworthiness | Apply temperature scaling |
| Performance Profiling | Latency + throughput script | Determine clinical feasibility | Batch inference, ONNX / TensorRT |
| Reliability Diagram | Bin-based ECE visualization | Visual miscalibration inspection | Adaptive binning / isotonic regression |

Why EfficientNet_B0: Balanced accuracy vs. parameter count; quick iteration in educational setting. 
Why not start with ViT / Large CNN: Data scale (10k images) insufficient to unlock transformer gains without heavy augmentation; risk of overfitting and longer training cycles.

---
## 6. Rationale for Choices
| Decision | Alternative | Why Chosen |
|----------|------------|------------|
| EfficientNet_B0 | ResNet50 / ViT-B/16 | Similar or better accuracy with fewer FLOPs |
| Grad-CAM | SHAP (CNN), LIME | Faster, first-step interpretability (foundation for expansion) |
| Mixed Precision | Full FP32 | 2–3× faster, lower memory; no accuracy drop observed |
| Lesion-level Splitting | Random image split | Prevents data leakage (same lesion in train & val) |
| Macro F1 Tracking | Accuracy only | Captures minority class performance important for clinical risk |
| ECE + Brier | Accuracy/confusion only | Probabilities must be reliable for clinical triage |

---
## 7. Current Prototype Results (2-Epoch Indicative Run)
| Metric | Value |
|--------|-------|
| Validation Accuracy | 0.8104 |
| Macro F1 | 0.6654 |
| ECE | 0.0450 |
| Brier Score | 0.2764 |
| Mean Latency (s) | 0.00904 |
| Throughput (img/s) | 110.59 |

### Key Class (Melanoma) Metrics
| Metric | Value |
|--------|-------|
| Precision | 0.6250 |
| Recall (Sensitivity) | 0.4012 |
| F1 | 0.4887 |
| Support (val images) | 324 |

Interpretation: Overall accuracy is promising early, but melanoma recall is still low—longer training + class rebalancing required before any clinical consideration.

### Per-Class Snapshot
| Class | Precision | Recall | F1 |
|-------|----------|--------|----|
| Melanocytic_nevi | 0.8879 | 0.9337 | 0.9102 |
| Melanoma | 0.6250 | 0.4012 | 0.4887 |
| Benign_keratosis | 0.5798 | 0.6450 | 0.6106 |
| Basal_cell_carcinoma | 0.6769 | 0.6286 | 0.6519 |
| Actinic_keratoses | 0.5581 | 0.5333 | 0.5455 |
| Vascular_lesions | 0.7917 | 0.8636 | 0.8261 |
| Dermatofibroma | 0.8333 | 0.5000 | 0.6250 |

---
## 8. Performance Comparison (Future Plan Table Example)
| Aspect | Current Prototype | Target (Post Full Training) |
|--------|-------------------|-----------------------------|
| Macro F1 | 0.6654 | ≥0.78 |
| Melanoma Recall | 0.4012 | ≥0.85 (triage mode) |
| ECE | 0.0450 | <0.035 (after temperature scaling) |
| Mean Latency | 9 ms | <8 ms (optional optimizations) |
| Throughput | 110 img/s | 150 img/s (batching + export) |

---
## 9. Technical Challenges & Mitigations
| Challenge | Impact | Mitigation |
|-----------|--------|-----------|
| Class Imbalance (nv dominates) | Depressed minority recall | Monitor macro F1; plan class-balanced/focal loss |
| Early Overconfidence | Potential mis-triage | Added calibration metrics + reliability diagram |
| Data Leakage Risk | Inflated metrics | Lesion-level partitioning |
| Small Epoch Run (demo constraints) | Unstable curves | Added smoothing + flagged as prototype |
| Explainability Scope | Limited trust foundation | Roadmap includes SHAP + uncertainty |
| GPU Memory Boundaries | Limits batch scaling | Mixed precision + efficient backbone |

---
## 10. Real-World Use Cases (Future State)
| Use Case | Description | Value |
|----------|-------------|-------|
| Primary Care Triage | Pre-screen lesions before dermatologist referral | Earlier melanoma escalation |
| Teledermatology Queue Prioritization | Sort remote submissions by risk | Faster high-risk review |
| Resident Training Tool | Heatmap-based feedback | Accelerated competency |
| Population Monitoring | Aggregate lesion class trends | Public health surveillance |

---
## 11. Potential Real-World Impact (After Validation)
| Dimension | Positive Outcome |
|----------|------------------|
| Clinical Outcomes | Earlier detection → improved survival |
| Resource Allocation | Reduced unnecessary biopsies |
| Accessibility | Augments regions with dermatologist shortages |
| Education | Visual guidance supports trainees |
| Trust & Transparency | Calibrated + explainable predictions |

---
## 12. Limitations (Honest Disclosure)
- Not trained to convergence (only 2 epochs shown).
- Dataset lacks broad skin tone diversity (risk of fairness gaps).
- No regulatory processes initiated—NOT for clinical decisions.
- No uncertainty estimation yet (only point probabilities).
- Grad-CAM alone insufficient for audit-grade interpretability.

---
## 13. Future Plans
| Phase | Focus | Key Actions |
|-------|-------|-------------|
| Phase 1 | Robust Training | 30–50 epochs, augmentation, class-balanced strategies |
| Phase 1 | Calibration | Temperature scaling, compare isotonic regression |
| Phase 2 | Explainability Expansion | SHAP, LIME, counterfactual prototypes |
| Phase 2 | Fairness & Bias | Stratified metrics by demographic attributes |
| Phase 3 | Multimodal Fusion | Add age, sex, lesion site meta-features |
| Phase 3 | Uncertainty | MC Dropout / Deep Ensembles |
| Phase 4 | Deployment Prep | Containerization, monitoring, model registry |
| Phase 4 | Clinical Study | Prospective validation & regulatory pathway |

---
## 14. Conclusion
This PBL project demonstrates an end-to-end AI dermatology decision support prototype emphasizing: reproducibility, interpretability, calibration awareness, and structured roadmap thinking. It is **a learning scaffold**—not yet a clinical instrument. The groundwork (clean metrics pipeline, artifact-driven README automation, calibration baseline) positions the system for credible academic extension and eventual translational exploration.

---
## 15. Bibliography / References
1. Tschandl P. et al. HAM10000 Dataset. *Scientific Data* (2018).  
2. Tan M., Le Q. EfficientNet: Rethinking Model Scaling for CNNs. *ICML* (2019).  
3. Selvaraju R.R. et al. Grad-CAM: Visual Explanations from Deep Networks. *ICCV* (2017).  
4. Guo C. et al. On Calibration of Modern Neural Networks. *ICML* (2017).  
5. He K. et al. Deep Residual Learning for Image Recognition. *CVPR* (2016) – comparative baseline context.  
6. Dosovitskiy A. et al. An Image Is Worth 16x16 Words (ViT). *ICLR* (2021).  
7. Lin T.-Y. et al. Focal Loss for Dense Object Detection. *ICCV* (2017) – motivating loss for class imbalance.  

---
## 16. Attribution & Academic Use
Prepared for Project-Based Learning (PBL) assessment; all medical interpretations are illustrative only. Dataset under CC BY-NC 4.0. Code is research/educational use—no warranty.

---
*End of Summary*
