# 4. Hyperparameter Optimization

## 4.1 Cross-Validation Strategy

A 5-fold GroupKFold cross-validation strategy is used to evaluate model generalization across different trips.

The dataset contains 80 training trips. Each fold uses 64 trips for training and 16 trips for validation. GroupKFold ensures that observations belonging to the same trip do not appear in both training and validation sets, reducing the risk of data leakage.

Candidate coverage is also monitored for each validation fold because the decoder cannot recover the correct road segment if it is absent from the generated candidate set.

## 4.2 Classifier Hyperparameter Optimization

Optuna is used to search for suitable classifier hyperparameters.

The classifier is trained using the predefined 15 input features, with `label` as the target variable. Model selection and optimization are performed using trip-based cross-validation.

The selected classifier should provide probability estimates through the `predict_proba()` interface, which are subsequently used as emission probabilities by the HMM decoder.

*Final classifier, search space, objective metric and optimized parameters will be added after the classifier training implementation is available.*

## 4.3 HMM Decoder Parameter Optimization

The Viterbi decoder exposes three configurable parameters:

- **Emission weight:** Controls the influence of classifier probabilities.
- **Transition weight:** Controls the influence of road-network transition scores.
- **Disconnected penalty:** Controls the penalty applied when two consecutive candidate road segments are not connected.

The default configuration is:

| Parameter | Default |
|---|---:|
| Emission weight | 1.0 |
| Transition weight | 1.0 |
| Disconnected penalty | -20.0 |

Optuna will be used to search for a configuration that improves decoding performance using out-of-fold classifier probabilities.

The student-test dataset will remain separate from the optimization process.

*The search space, objective function and optimized values will be added after the classifier probabilities become available.*

# 5. Model Evaluation and Comparison

## 5.1 Evaluation Methodology

The final evaluation compares classifier-only predictions against predictions produced by the classifier combined with the HMM decoder.

The evaluation distinguishes candidate-generation coverage from classification and decoding performance. This is important because an observation cannot be correctly decoded if its ground-truth road segment is missing from the candidate set.

The 10 student-test trips are reserved for final evaluation and are not used for hyperparameter optimization.

## 5.2 Evaluation Metrics

The following metrics will be reported:

- **Candidate coverage:** Percentage of observations for which the correct road segment exists in the generated candidate set.
- **Top-1 accuracy:** Percentage of observations where the highest-scoring candidate matches the ground truth.
- **Macro-F1:** Measures classification performance while giving equal importance to both classes.
- **Log loss:** Measures the quality of predicted probabilities.
- **Route continuity:** Measures the proportion of consecutive decoded road segments that are connected in the road network.
- **Inference time:** Measures the computational time required to produce predictions.

Metrics will distinguish between all observations and observations with available ground-truth candidates wherever applicable.

## 5.3 Classifier-only vs HMM Comparison

The classifier-only approach selects the candidate with the highest predicted probability independently for each observation.

The HMM-based approach uses the same classifier probabilities but additionally considers road-network connectivity between consecutive observations.

The comparison will establish whether temporal and spatial continuity constraints improve map-matching performance.

*Final comparison tables and results will be added after model training, decoder optimization and student-test evaluation.*

## 5.4 Discussion

The evaluation will examine the effect of candidate-generation coverage, classifier quality and road-network constraints on the final map-matching performance.

Particular attention will be given to trips with low candidate coverage, since these can limit the achievable performance regardless of classifier or decoder quality.