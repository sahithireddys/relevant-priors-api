# Experiments

## Goal

The task is to decide, for each previous examination of the same patient, whether the prior should be shown to the radiologist while reading the current examination. The endpoint must return exactly one prediction for every prior study in every submitted case.

## Approach

I implemented a deterministic, low-latency model instead of calling an external LLM. This avoids timeout risk on the hidden evaluation, where a request may include many cases and many prior examinations.

The model uses two layers:

1. A public-split calibrated lookup table keyed by normalized `(current_study_description, prior_study_description)`. If a pair was observed in the public split, the model uses the majority public label for that pair.
2. A rule-based fallback for unseen pairs.

The fallback extracts three kinds of features from study descriptions:

1. **Modality**: MRI, CT, XR, ultrasound, nuclear medicine, mammography, fluoroscopy, IR, or other.
2. **Anatomy/body region**: brain/head, spine regions, chest/lung, cardiac, abdomen, pelvis, breast, extremities, vascular, and whole-body studies.
3. **Description similarity and recency**: exact description matches, token overlap, and study-date gap.

## Baselines

### Baseline 1: return all priors as relevant

This has excellent recall but poor precision. It may perform decently if the dataset is highly imbalanced toward relevant priors, but it would show many unrelated studies such as knee radiographs for a head CT.

### Baseline 2: exact same study description only

This has high precision but misses clinically useful priors where descriptions differ slightly, such as:

- `CT HEAD WITHOUT CNTRST` vs. `MRI BRAIN STROKE LIMITED WITHOUT CONTRAST`
- `CT ABDOMEN/PELVIS` vs. `US RIGHT UPPER QUADRANT`
- `XR CHEST 1 VIEW` vs. `CT CHEST`

### Final implemented model: calibrated pair lookup + anatomy/modality fallback

For seen public description pairs, the final implementation uses the majority label from public calibration. For unseen pairs, it predicts a prior as relevant when:

- the current and prior descriptions are exact or highly similar matches;
- they share the same or related anatomy and have compatible modalities;
- they share/relate anatomically and the prior is recent;
- the prior is a broad whole-body/PET-style comparison relevant to body imaging.

## What worked

- Grouping descriptions into canonical anatomy buckets is more robust than raw string matching.
- Treating CT and MRI as compatible cross-sectional modalities improves recall.
- Allowing ultrasound, CT, and MRI to cross-match for abdomen/pelvis/vascular studies captures many clinically relevant comparisons.
- Returning every prediction in a single bulk pass keeps latency low and avoids timeout risk.
- Logging request ID, case count, prior count, and elapsed time makes evaluator debugging easier.

## What failed or was avoided

- One LLM call per prior was avoided because the challenge explicitly warns this can timeout.
- Exact string matching alone is too brittle because radiology descriptions contain abbreviations and spelling variants such as `CNTRST`.
- A very broad always-true rule would likely reduce private-split performance by marking unrelated anatomy as relevant.

## Expected limitations

- The model is deterministic and does not learn from public labels unless the public JSON is used for additional tuning.
- Some ambiguous studies may be misclassified when the description lacks clear anatomy.
- Subspecialty-specific relevance can be more nuanced than modality/anatomy matching.
- The model does not use report text, indication, diagnosis, or ordering context because those fields are not included.

## Next improvements

1. Tune thresholds and anatomy rules against the 996-case public evaluation split.
2. Add confusion analysis by modality/anatomy pair to identify common false positives and false negatives.
3. Train a lightweight classifier using features such as modality pair, anatomy pair, token similarity, recency, laterality, and procedure keywords.
4. Add abbreviation normalization for more radiology-specific variants.
5. Add optional caching keyed by `(current_study_id, prior_study_id)` if the evaluator repeats cases.
6. Consider a batched LLM or embedding-based reranker only after establishing strict timeout controls and caching.


## Public evaluation result

After fixing the evaluator to read the challenge file's separate `truth` array, I evaluated on the uploaded public split:

- Cases: 996
- Labeled previous examinations: 27,614
- Rule-only baseline accuracy: 0.8519
- Final public-calibrated model accuracy: 0.9883

The public-calibrated result is in-sample on the public split, so it should not be interpreted as a guaranteed hidden-split score. The fallback rule model is still included for unseen private description pairs.

## Private-split considerations

The private split may include description pairs not seen in the public JSON. For those cases, the endpoint falls back to deterministic radiology-style anatomy/modality rules. This keeps inference fast and prevents the endpoint from depending on network calls or per-prior LLM requests.
