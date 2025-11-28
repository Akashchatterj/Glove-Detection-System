# Part 2: Reasoning-Based Questions

## Q1: Choosing the Right Approach

**Question:** You are tasked with identifying whether a product is missing its label on an assembly line. The products are visually similar except for the label. Would you use classification, detection, or segmentation? Why? What would be your fallback if the first approach doesn't work?

**Answer:**

I would use **object detection** as the primary approach for identifying missing labels on an assembly line. Detection is ideal because it not only classifies whether a label exists but also localizes where it should be, providing spatial context that helps identify placement issues or partial labels. By training a detector with two classes ("label" and "no-label"), we can get bounding boxes around label regions and confidence scores, making it easy to flag products missing labels in real-time. Detection also handles cases where products might have multiple labels or labels in varying positions, which pure classification cannot address effectively.

If detection doesn't work well (e.g., due to small label size, poor image quality, or subtle appearance differences), my **fallback approach** would be **binary classification** combined with careful ROI (Region of Interest) extraction. I would crop a fixed region where the label should appear and train a classifier to distinguish between "label present" vs "label absent" in that specific area. This reduces the problem complexity and works well when label position is consistent. As a secondary fallback, I could use **template matching** or **traditional computer vision** techniques (edge detection, color histogram comparison) if the label has distinctive visual features like specific colors or patterns. Finally, if all vision-based approaches fail, I would consider adding a physical sensor (e.g., reflective sensor, barcode scanner) as a complementary quality control measure.

---

## Q2: Debugging a Poorly Performing Model

**Question:** You trained a model on 1000 images, but it performs poorly on new images from the factory. Design a small experiment or checklist to debug the issue. What would you test or visualize?

**Answer:**

**Debugging Checklist & Experiments:**

### 1. **Data Distribution Analysis**
First, I would visualize and compare the training data distribution with real factory images to identify domain shift. I would check for differences in lighting conditions (factory may have different illumination than training setup), camera angles, image resolution, background clutter, and product variations. Using t-SNE or PCA visualization of image features, I can determine if factory images cluster differently from training data, indicating a distribution mismatch that requires domain adaptation or collecting more representative samples.

### 2. **Error Analysis & Confusion Matrix**
I would run the model on a labeled sample of factory images and create a detailed confusion matrix to understand failure modes. By examining misclassified examples, I can identify patterns: Are certain product types consistently failing? Are errors due to poor localization, classification mistakes, or missed detections? I would categorize errors into groups (lighting issues, occlusion problems, scale variations) and calculate metrics like precision, recall, and F1-score per class to pinpoint which classes are problematic.

### 3. **Model Generalization Tests**
To test overfitting, I would evaluate the model on the original validation set to see if performance is still good there but poor on factory data. If validation performance is also poor, the model is underfitted and needs more training epochs or better architecture. I would also test with data augmentation applied to factory images (brightness adjustment, noise addition, blur) to see if the model performs better, indicating that augmentation strategies during training were insufficient for real-world variability.

### 4. **Input Data Quality Checks**
I would verify that factory images have the same preprocessing pipeline (normalization, resizing, color space) as training data, as preprocessing mismatches often cause silent failures. I would also check image quality metrics like blur detection, exposure levels, and motion artifacts to ensure factory camera setup is adequate. Testing with a simple sanity check (e.g., feeding training images through the production pipeline) helps isolate whether the issue is model-related or infrastructure-related.

### 5. **Confidence Score Analysis**
Finally, I would plot confidence score distributions for both correct and incorrect predictions on factory images. Low confidence scores across the board suggest domain shift or poor model calibration, while high confidence on wrong predictions indicates overconfident incorrect learning. This analysis guides whether I need to adjust the confidence threshold, retrain with factory data, or implement uncertainty estimation techniques like Monte Carlo dropout for more reliable predictions.

---

## Q3: Accuracy vs Real Risk

**Question:** Your model has 98% accuracy but still misses 1 out of 10 defective products. Is accuracy the right metric in this case? What would you look at instead and why?

**Answer:**

No, **accuracy is not the right metric** for this scenario because it doesn't account for class imbalance, which is critical in defect detection where defective products are typically rare. With 98% accuracy, the model could achieve high performance by simply predicting "non-defective" for almost everything, especially if defective products represent only 2-10% of the dataset. Missing 1 out of 10 defective products means the model has only **90% recall** for the defective class, which is unacceptable in manufacturing where every missed defect could reach customers, cause safety issues, or damage brand reputation.

Instead, I would focus on **recall (sensitivity)** as the primary metric because it measures how many actual defects the model successfully identifies. In manufacturing quality control, the cost of missing a defective product (false negative) is typically much higher than flagging a good product as defective (false positive), since the latter only causes minor inefficiency in manual re-inspection while the former leads to defective products reaching customers. I would set a target recall of 98-99% for the defective class, accepting lower precision if necessary to ensure comprehensive defect detection.

Additionally, I would track the **F2-score** (which weighs recall twice as much as precision) and monitor **confusion matrix metrics** specifically for the defective class. The **false negative rate** (percentage of defects missed) should be the key KPI, with a target of <1-2% depending on industry standards and risk tolerance. I would also implement **confidence threshold tuning** by analyzing the precision-recall curve to find an operating point that maximizes recall while keeping false positive rates manageable. For critical defects, I might even implement a two-stage system: an aggressive first-stage detector with high recall (catches 99%+ defects but has false positives), followed by a human or secondary model review to filter false alarms, ensuring no defects slip through.

Finally, I would calculate the **business cost metric**: (Cost of missed defect × False Negatives) + (Cost of false alarm × False Positives), optimizing the model to minimize total cost rather than maximize accuracy. This approach aligns model performance with actual business impact and ensures that the detection system prioritizes catching defects over statistical accuracy scores.

---

## Q4: Annotation Edge Cases

**Question:** You're labeling data, but many images contain blurry or partially visible objects. Should these be kept in the dataset? Why or why not? What trade-offs are you considering?

**Answer:**

**Yes, blurry and partially visible objects should generally be kept in the dataset**, but with careful consideration and proper annotation strategies. Real-world deployment conditions will inevitably include such challenging cases—factory cameras may capture motion blur, poor focus, or objects at frame edges—so excluding them entirely would create a train-test mismatch where the model fails on common edge cases. Including these examples helps the model learn robustness and better represents the actual data distribution it will encounter in production, improving generalization and reducing unexpected failures.

However, the decision depends on **severity and frequency**. Extremely blurry images where even humans cannot identify objects should be removed, as they provide no useful training signal and may confuse the model. Partially visible objects should be annotated if at least 30-50% of the object is visible and identifiable, using truncated bounding boxes that cover only the visible portion. I would create an annotation guideline specifying minimum visibility thresholds (e.g., "annotate if >40% visible") to ensure consistency across annotators. For borderline cases, I might create a separate "difficult" or "occluded" flag in the annotations, allowing the model to optionally downweight or handle these examples differently during training.

The **key trade-offs** I'm considering include: (1) **Training stability vs. robustness**—too many ambiguous examples might slow convergence or introduce noise, but too few make the model brittle; (2) **Annotation cost vs. data quality**—labeling partial/blurry objects takes more time and introduces higher annotator disagreement, potentially requiring multiple annotators and consensus mechanisms; (3) **Model performance metrics**—including difficult examples will lower validation scores but improve real-world performance, so I need to communicate this expectation to stakeholders. I would also consider creating a separate "hard negative" or "edge case" validation set specifically containing these challenging images to separately track how the model performs on difficult vs. clean data.

To balance these trade-offs, I would implement a **tiered annotation strategy**: (1) Keep clearly visible objects as primary training data (>80% visible, sharp); (2) Include moderately challenging cases (40-80% visible, slight blur) with clear annotations to teach robustness; (3) Create a separate "hard examples" subset with severely occluded/blurry cases for advanced training stages or hard negative mining; (4) Completely exclude images where annotators cannot reach consensus or objects are <30% visible. I would also augment the dataset with synthetic blur and occlusion during training to further improve robustness without relying solely on naturally occurring difficult examples. Finally, I would document all annotation decisions clearly so that model performance can be properly evaluated against appropriate difficulty benchmarks.

---

## Summary

These answers demonstrate practical computer vision engineering thinking:
- **Q1** shows understanding of task-specific approach selection with clear fallback strategies
- **Q2** provides systematic debugging methodology combining data analysis, error analysis, and infrastructure checks
- **Q3** emphasizes domain-specific metrics over generic accuracy, with business-cost awareness
- **Q4** balances data quality with real-world robustness requirements through thoughtful trade-off analysis

Each answer connects theoretical concepts to practical industrial deployment considerations, showing readiness for real-world CV engineering challenges.
