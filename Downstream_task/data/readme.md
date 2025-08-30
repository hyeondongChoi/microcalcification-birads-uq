# Downstream Task Dataset (Microcalcificiation)

**Data availability.** Due to hospital policy and patient privacy, the datasets cannot be released.

## Training & Validation
- We use `torchvision.datasets.ImageFolder`.
- Expected directory layout (example):

```
CNUH_data/
├─ train/
│  ├─ grade_3/
│  │  └─ *.png
│  └─ grade_4/
│     └─ *.png
└─ val/
   ├─ grade_3/
   │  └─ *.png
   └─ grade_4/
      └─ *.png
```

## Test / Inference
- To apply the proposed **BI-RADS grade classification** algorithm, a **custom dataset** is required.
- An example of the expected file structure (manifest) is provided in **`test_inference.csv`**.
