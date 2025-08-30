## Determining the Uncertainty Threshold

This section describes how we set the uncertainty threshold. We propose two validation-driven strategies:

1) **Entropy-quartile method**  
   From the validation set, compute the **average predictive entropy** at each confidence threshold.  
   Use the quartiles (**Q1**, **Q2/median**, **Q3**) of these averages as candidate uncertainty thresholds.

2) **Data-coverage method**  
   From the validation set, compute the **case count** at each confidence threshold.  
   Define **coverage** as `case_count / total_validation_samples`.  
   Select the threshold whose coverage is closest to the target levels (**30%**, **50%**, **70%**).  
   We report results for all three targets.

**Notes**
- Thresholds are chosen **on the validation set only** and then kept fixed for test/inference.
- Ensure the **same preprocessing/normalization** is used during threshold selection and evaluation.
