# Prediction Stability Analysis Report

## Executive Summary

- **Model**: Logistic Regression (C=0.01) on v2 features
- **Test set size**: 418 passengers
- **Bubble passengers** (P between 0.3-0.7): **128** (30.6%)
- **Tight bubble** (P between 0.4-0.6): **70** (16.7%)
- **Bootstrap unstable** (flip in >0% of 100 resamples): **86** passengers
- **Highly unstable** (flip in >10% of resamples): **43** passengers
- **Predictions match logreg_v2.csv**: False

## 1. Probability Distribution

| Probability Range | Count | % of Test Set |
|:---|---:|---:|
| 0.0 - 0.1 | 5 | 1.2% |
| 0.1 - 0.2 | 173 | 41.4% |
| 0.2 - 0.3 | 31 | 7.4% |
| 0.3 - 0.4 | 16 | 3.8% |
| 0.4 - 0.5 | 20 | 4.8% |
| 0.5 - 0.6 | 50 | 12.0% |
| 0.6 - 0.7 | 42 | 10.0% |
| 0.7 - 0.8 | 37 | 8.9% |
| 0.8 - 0.9 | 34 | 8.1% |
| 0.9 - 1.0 | 10 | 2.4% |

## 2. Bubble Passengers by Subgroup

| Sex | Pclass | Total N | N in Bubble | % Bubble | Mean P(Surv) | Pred Surv Rate |
|:---|---:|---:|---:|---:|---:|---:|
| female | 1 | 50 | 2 | 4% | 0.839 | 1.000 |
| female | 2 | 30 | 0 | 0% | 0.771 | 1.000 |
| female | 3 | 72 | 68 | 94% | 0.576 | 0.931 |
| male | 1 | 57 | 41 | 72% | 0.407 | 0.228 |
| male | 2 | 63 | 4 | 6% | 0.200 | 0.032 |
| male | 3 | 146 | 13 | 9% | 0.169 | 0.075 |

**Key finding**: Of the 128 bubble passengers, **68** (53%) are 3rd-class females.

## 3. Complete Bubble Passenger List

| PID | P(Surv) | Pred | Sex | Pcl | Age | Fare | Title | FamSz | FlipRate | Flipped by 12a |
|---:|---:|---:|:---|---:|---:|---:|:---|---:|---:|:---|
| 947 | 0.301 | 0 | male | 3 | 10 | 29.1 | Master | 5 | 0% |  |
| 1024 | 0.302 | 0 | female | 3 | ? | 25.5 | Mrs | 4 | 0% |  |
| 912 | 0.312 | 0 | male | 1 | 55 | 59.4 | Mr | 1 | 0% |  |
| 915 | 0.313 | 0 | male | 1 | 21 | 61.4 | Mr | 1 | 0% |  |
| 1219 | 0.325 | 0 | male | 1 | 46 | 79.2 | Mr | 0 | 0% |  |
| 1264 | 0.325 | 0 | male | 1 | 49 | 0.0 | Mr | 0 | 0% |  |
| 1257 | 0.328 | 0 | female | 3 | ? | 69.5 | Mrs | 10 | 0% |  |
| 1227 | 0.340 | 0 | male | 1 | 30 | 26.0 | Mr | 0 | 0% |  |
| 920 | 0.343 | 0 | male | 1 | 41 | 30.5 | Mr | 0 | 0% |  |
| 1107 | 0.350 | 0 | male | 1 | 42 | 42.5 | Mr | 0 | 0% |  |
| 1270 | 0.355 | 0 | male | 1 | 55 | 50.0 | Mr | 0 | 0% |  |
| 942 | 0.360 | 0 | male | 1 | 24 | 60.0 | Mr | 1 | 0% |  |
| 1032 | 0.360 | 0 | female | 3 | 10 | 46.9 | Miss | 7 | 0% |  |
| 1179 | 0.374 | 0 | male | 1 | 24 | 82.3 | Mr | 1 | 0% |  |
| 1200 | 0.381 | 0 | male | 1 | 55 | 93.5 | Mr | 2 | 0% |  |
| 1282 | 0.382 | 0 | male | 1 | 23 | 93.5 | Mr | 0 | 0% |  |
| 1198 | 0.419 | 0 | male | 1 | 30 | 151.6 | Mr | 3 | 4% |  |
| 1284 | 0.431 | 0 | male | 3 | 13 | 20.2 | Master | 2 | 10% | YES |
| 960 | 0.432 | 0 | male | 1 | 31 | 28.5 | Mr | 0 | 10% |  |
| 1223 | 0.432 | 0 | male | 1 | 39 | 29.7 | Mr | 0 | 10% |  |
| 938 | 0.432 | 0 | male | 1 | 45 | 29.7 | Mr | 0 | 10% |  |
| 1058 | 0.446 | 0 | male | 1 | 48 | 50.5 | Mr | 0 | 12% |  |
| 1069 | 0.449 | 0 | male | 1 | 54 | 55.4 | Mr | 1 | 11% |  |
| 1247 | 0.449 | 0 | male | 1 | 50 | 26.0 | Mr | 0 | 11% |  |
| 1050 | 0.449 | 0 | male | 1 | 42 | 26.6 | Mr | 0 | 11% |  |
| 933 | 0.449 | 0 | male | 1 | ? | 26.6 | Mr | 0 | 11% |  |
| 926 | 0.450 | 0 | male | 1 | 30 | 57.8 | Mr | 1 | 12% |  |
| 1297 | 0.451 | 0 | male | 2 | 20 | 13.9 | Mr | 0 | 22% |  |
| 1193 | 0.451 | 0 | male | 2 | ? | 15.0 | Mr | 0 | 22% |  |
| 1126 | 0.459 | 0 | male | 1 | 39 | 71.3 | Mr | 1 | 14% |  |
| 1162 | 0.463 | 0 | male | 1 | 46 | 75.2 | Mr | 0 | 16% |  |
| 1010 | 0.463 | 0 | male | 1 | 36 | 75.2 | Mr | 0 | 16% |  |
| 1185 | 0.463 | 0 | male | 1 | 53 | 81.9 | Rare | 2 | 26% |  |
| 973 | 0.465 | 0 | male | 1 | 67 | 221.8 | Mr | 1 | 22% |  |
| 1137 | 0.465 | 0 | male | 1 | 41 | 51.9 | Mr | 1 | 22% |  |
| 1038 | 0.466 | 0 | male | 1 | ? | 51.9 | Mr | 0 | 20% |  |
| 1144 | 0.503 | 1 | male | 1 | 27 | 136.8 | Mr | 1 | 56% | YES |
| 1208 | 0.510 | 1 | male | 1 | 57 | 146.5 | Mr | 1 | 50% | YES |
| 913 | 0.515 | 1 | male | 3 | 9 | 3.2 | Master | 1 | 32% |  |
| 1199 | 0.519 | 1 | male | 3 | 1 | 9.3 | Master | 1 | 31% |  |
| 1173 | 0.522 | 1 | male | 3 | 1 | 13.8 | Master | 2 | 27% |  |
| 1093 | 0.523 | 1 | male | 3 | 0 | 14.4 | Master | 2 | 27% |  |
| 1084 | 0.523 | 1 | male | 3 | 12 | 14.5 | Master | 2 | 27% |  |
| 1236 | 0.523 | 1 | male | 3 | ? | 14.5 | Master | 2 | 27% |  |
| 1023 | 0.523 | 1 | male | 1 | 53 | 28.5 | Rare | 0 | 39% | YES |
| 1136 | 0.529 | 1 | male | 3 | ? | 23.4 | Master | 3 | 25% |  |
| 910 | 0.539 | 1 | female | 3 | 27 | 7.9 | Miss | 1 | 11% | YES |
| 1268 | 0.540 | 1 | female | 3 | 22 | 8.7 | Miss | 2 | 10% | YES |
| 1237 | 0.540 | 1 | female | 3 | 16 | 7.7 | Miss | 0 | 14% | YES |
| 1304 | 0.540 | 1 | female | 3 | 28 | 7.8 | Miss | 0 | 14% | YES |
| 1089 | 0.540 | 1 | female | 3 | 18 | 7.8 | Miss | 0 | 14% | YES |
| 1049 | 0.540 | 1 | female | 3 | 23 | 7.9 | Miss | 0 | 14% | YES |
| 990 | 0.540 | 1 | female | 3 | 20 | 7.9 | Miss | 0 | 14% | YES |
| 964 | 0.540 | 1 | female | 3 | 29 | 7.9 | Miss | 0 | 14% | YES |
| 1030 | 0.540 | 1 | female | 3 | 23 | 8.1 | Miss | 0 | 14% | YES |
| 1160 | 0.540 | 1 | female | 3 | ? | 8.1 | Miss | 0 | 14% | YES |
| 979 | 0.540 | 1 | female | 3 | 18 | 8.1 | Miss | 0 | 14% | YES |
| 928 | 0.540 | 1 | female | 3 | ? | 8.1 | Miss | 0 | 14% | YES |
| 1172 | 0.541 | 1 | female | 3 | 23 | 8.7 | Miss | 0 | 14% | YES |
| 929 | 0.541 | 1 | female | 3 | 21 | 8.7 | Miss | 0 | 14% | YES |
| 1061 | 0.541 | 1 | female | 3 | 22 | 9.0 | Miss | 0 | 14% | YES |
| 1296 | 0.544 | 1 | male | 1 | 43 | 27.7 | Mr | 1 | 18% | YES |
| 1017 | 0.545 | 1 | female | 3 | 17 | 16.1 | Miss | 1 | 10% | YES |
| 965 | 0.545 | 1 | male | 1 | 28 | 27.7 | Mr | 0 | 19% | YES |
| 1299 | 0.553 | 1 | male | 1 | 50 | 211.5 | Mr | 2 | 17% | YES |
| 967 | 0.554 | 1 | male | 1 | 32 | 211.5 | Mr | 0 | 20% | YES |
| 1259 | 0.561 | 1 | female | 3 | 22 | 39.7 | Miss | 0 | 2% |  |
| 1128 | 0.576 | 1 | male | 1 | 64 | 75.2 | Mr | 1 | 10% | YES |
| 1073 | 0.581 | 1 | male | 1 | 37 | 83.2 | Mr | 2 | 9% | YES |
| 893 | 0.582 | 1 | female | 3 | 47 | 7.0 | Mrs | 1 | 1% | YES |
| 1091 | 0.584 | 1 | female | 3 | ? | 8.1 | Mrs | 0 | 1% |  |
| 1045 | 0.586 | 1 | female | 3 | 36 | 12.2 | Mrs | 2 | 1% | YES |
| 896 | 0.586 | 1 | female | 3 | 22 | 12.3 | Mrs | 2 | 1% | YES |
| 1051 | 0.587 | 1 | female | 3 | 26 | 13.8 | Mrs | 2 | 0% | YES |
| 982 | 0.587 | 1 | female | 3 | 22 | 13.9 | Mrs | 1 | 0% | YES |
| 1201 | 0.587 | 1 | female | 3 | 45 | 14.1 | Mrs | 1 | 0% | YES |
| 1251 | 0.588 | 1 | female | 3 | 30 | 15.6 | Mrs | 1 | 0% | YES |
| 1274 | 0.588 | 1 | female | 3 | ? | 14.5 | Mrs | 0 | 1% |  |
| 941 | 0.588 | 1 | female | 3 | 36 | 15.9 | Mrs | 2 | 0% | YES |
| 1275 | 0.588 | 1 | female | 3 | 19 | 16.1 | Mrs | 1 | 0% | YES |
| 924 | 0.591 | 1 | female | 3 | 33 | 20.6 | Mrs | 3 | 0% | YES |
| 1057 | 0.592 | 1 | female | 3 | 26 | 22.0 | Mrs | 2 | 0% | YES |
| 925 | 0.593 | 1 | female | 3 | ? | 23.4 | Mrs | 3 | 0% | YES |
| 1183 | 0.600 | 1 | female | 3 | 30 | 7.0 | Miss | 0 | 4% |  |
| 1005 | 0.600 | 1 | female | 3 | 18 | 7.3 | Miss | 0 | 4% |  |
| 898 | 0.600 | 1 | female | 3 | 30 | 7.6 | Miss | 0 | 4% |  |
| 1300 | 0.600 | 1 | female | 3 | ? | 7.7 | Miss | 0 | 4% |  |
| 955 | 0.600 | 1 | female | 3 | 22 | 7.7 | Miss | 0 | 4% |  |
| 1207 | 0.600 | 1 | female | 3 | 17 | 7.7 | Miss | 0 | 4% |  |
| 1052 | 0.600 | 1 | female | 3 | ? | 7.7 | Miss | 0 | 4% |  |
| 1302 | 0.600 | 1 | female | 3 | ? | 7.8 | Miss | 0 | 4% |  |
| 1205 | 0.600 | 1 | female | 3 | 37 | 7.8 | Miss | 0 | 4% |  |
| 980 | 0.600 | 1 | female | 3 | ? | 7.8 | Miss | 0 | 4% |  |
| 1119 | 0.600 | 1 | female | 3 | ? | 7.8 | Miss | 0 | 4% |  |
| 1196 | 0.600 | 1 | female | 3 | ? | 7.8 | Miss | 0 | 4% |  |
| 971 | 0.600 | 1 | female | 3 | 24 | 7.8 | Miss | 0 | 4% |  |
| 1174 | 0.600 | 1 | female | 3 | ? | 7.8 | Miss | 0 | 4% |  |
| 1098 | 0.600 | 1 | female | 3 | 35 | 7.8 | Miss | 0 | 4% |  |
| 962 | 0.600 | 1 | female | 3 | 24 | 7.8 | Miss | 0 | 4% |  |
| 1003 | 0.600 | 1 | female | 3 | ? | 7.8 | Miss | 0 | 4% |  |
| 978 | 0.600 | 1 | female | 3 | 27 | 7.9 | Miss | 0 | 4% |  |
| 1108 | 0.600 | 1 | female | 3 | ? | 7.9 | Miss | 0 | 4% |  |
| 958 | 0.600 | 1 | female | 3 | 18 | 7.9 | Miss | 0 | 4% |  |
| 1165 | 0.604 | 1 | female | 3 | ? | 15.5 | Miss | 1 | 4% |  |
| 1092 | 0.605 | 1 | female | 3 | ? | 15.5 | Miss | 0 | 4% |  |
| 1019 | 0.609 | 1 | female | 3 | ? | 23.2 | Miss | 2 | 2% |  |
| 1231 | 0.612 | 1 | male | 3 | ? | 7.2 | Master | 0 | 1% |  |
| 981 | 0.612 | 1 | male | 2 | 2 | 23.0 | Master | 2 | 1% |  |
| 1134 | 0.614 | 1 | male | 1 | 45 | 134.5 | Mr | 2 | 1% |  |
| 972 | 0.616 | 1 | male | 3 | 6 | 15.2 | Master | 2 | 0% |  |
| 1053 | 0.616 | 1 | male | 3 | 7 | 15.2 | Master | 2 | 0% |  |
| 1086 | 0.618 | 1 | male | 2 | 8 | 32.5 | Master | 2 | 1% |  |
| 1309 | 0.621 | 1 | male | 3 | ? | 22.4 | Master | 2 | 0% |  |
| 1155 | 0.635 | 1 | female | 3 | 1 | 12.2 | Miss | 2 | 0% |  |
| 1301 | 0.636 | 1 | female | 3 | 3 | 13.8 | Miss | 2 | 0% |  |
| 1176 | 0.640 | 1 | female | 3 | 2 | 20.2 | Miss | 2 | 0% |  |
| 1246 | 0.640 | 1 | female | 3 | 0 | 20.6 | Miss | 3 | 0% |  |
| 1094 | 0.651 | 1 | male | 1 | 47 | 227.5 | Rare | 1 | 2% |  |
| 1123 | 0.667 | 1 | female | 1 | 21 | 26.6 | Miss | 0 | 0% |  |
| 945 | 0.669 | 1 | female | 1 | 28 | 263.0 | Miss | 5 | 0% |  |
| 911 | 0.672 | 1 | female | 3 | 45 | 7.2 | Mrs | 0 | 0% |  |
| 1239 | 0.672 | 1 | female | 3 | 38 | 7.2 | Mrs | 0 | 0% |  |
| 900 | 0.672 | 1 | female | 3 | 18 | 7.2 | Mrs | 0 | 0% |  |
| 996 | 0.672 | 1 | female | 3 | 16 | 8.5 | Mrs | 2 | 0% |  |
| 1141 | 0.675 | 1 | female | 3 | ? | 14.5 | Mrs | 1 | 0% |  |
| 1117 | 0.676 | 1 | female | 3 | ? | 15.2 | Mrs | 2 | 0% |  |
| 1225 | 0.676 | 1 | female | 3 | 19 | 15.7 | Mrs | 2 | 0% |  |
| 956 | 0.693 | 1 | male | 1 | 13 | 262.4 | Master | 4 | 0% | YES |

## 4. Bootstrap Stability Analysis

Retrained on 100 bootstrap resamples of the training data.

| Flip Rate Threshold | Passengers | % of Test |
|:---|---:|---:|
| > 0% | 86 | 20.6% |
| > 1% | 77 | 18.4% |
| > 5% | 51 | 12.2% |
| > 10% | 43 | 10.3% |
| > 20% | 15 | 3.6% |
| > 30% | 5 | 1.2% |
| > 50% | 1 | 0.2% |

### Most Unstable Passengers (by bootstrap flip rate)

| PID | P(Surv) | Pred | FlipRate | BootStd | Sex | Pcl | Age | Title | Flipped 12a |
|---:|---:|---:|---:|---:|:---|---:|---:|:---|:---|
| 1144 | 0.503 | 1 | 56% | 0.047 | male | 1 | 27 | Mr | YES |
| 1208 | 0.510 | 1 | 50% | 0.047 | male | 1 | 57 | Mr | YES |
| 1023 | 0.523 | 1 | 39% | 0.067 | male | 1 | 53 | Rare | YES |
| 913 | 0.515 | 1 | 32% | 0.047 | male | 3 | 9 | Master |  |
| 1199 | 0.519 | 1 | 31% | 0.047 | male | 3 | 1 | Master |  |
| 1084 | 0.523 | 1 | 27% | 0.047 | male | 3 | 12 | Master |  |
| 1093 | 0.523 | 1 | 27% | 0.047 | male | 3 | 0 | Master |  |
| 1173 | 0.522 | 1 | 27% | 0.047 | male | 3 | 1 | Master |  |
| 1236 | 0.523 | 1 | 27% | 0.047 | male | 3 | ? | Master |  |
| 1185 | 0.463 | 0 | 26% | 0.066 | male | 1 | 53 | Rare |  |
| 1136 | 0.529 | 1 | 25% | 0.047 | male | 3 | ? | Master |  |
| 973 | 0.465 | 0 | 22% | 0.055 | male | 1 | 67 | Mr |  |
| 1137 | 0.465 | 0 | 22% | 0.050 | male | 1 | 41 | Mr |  |
| 1193 | 0.451 | 0 | 22% | 0.064 | male | 2 | ? | Mr |  |
| 1297 | 0.451 | 0 | 22% | 0.064 | male | 2 | 20 | Mr |  |
| 967 | 0.554 | 1 | 20% | 0.048 | male | 1 | 32 | Mr | YES |
| 1038 | 0.466 | 0 | 20% | 0.049 | male | 1 | ? | Mr |  |
| 965 | 0.545 | 1 | 19% | 0.052 | male | 1 | 28 | Mr | YES |
| 1296 | 0.544 | 1 | 18% | 0.053 | male | 1 | 43 | Mr | YES |
| 1299 | 0.553 | 1 | 17% | 0.051 | male | 1 | 50 | Mr | YES |
| 1010 | 0.463 | 0 | 16% | 0.044 | male | 1 | 36 | Mr |  |
| 1162 | 0.463 | 0 | 16% | 0.044 | male | 1 | 46 | Mr |  |
| 928 | 0.540 | 1 | 14% | 0.035 | female | 3 | ? | Miss | YES |
| 929 | 0.541 | 1 | 14% | 0.035 | female | 3 | 21 | Miss | YES |
| 964 | 0.540 | 1 | 14% | 0.035 | female | 3 | 29 | Miss | YES |

### Instability by Subgroup

| Sex | Pclass | N | Any Flip | Flip >10% | Mean Flip Rate |
|:---|---:|---:|---:|---:|---:|
| female | 1 | 50 | 0 | 0 | 0.0% |
| female | 2 | 30 | 0 | 0 | 0.0% |
| female | 3 | 72 | 45 | 14 | 4.3% |
| male | 1 | 57 | 28 | 20 | 8.4% |
| male | 2 | 63 | 4 | 2 | 0.7% |
| male | 3 | 146 | 9 | 7 | 1.4% |

## 5. Threshold Sensitivity Analysis

| Threshold | Predicted Survived | Change from 0.50 |
|---:|---:|---:|
| 0.40 | 193 | +20 |
| 0.42 | 192 | +19 |
| 0.44 | 188 | +15 |
| 0.45 | 183 | +10 |
| 0.46 | 179 | +6 |
| 0.48 | 173 | +0 |
| 0.50 | 173 | +0 (baseline) |
| 0.52 | 169 | +4 |
| 0.54 | 160 | +13 |
| 0.55 | 145 | +28 |
| 0.56 | 143 | +30 |
| 0.58 | 141 | +32 |
| 0.60 | 123 | +50 |

## 6. Cross-Reference with 12a Segmented Model

The 12a segmented model changed **39** predictions vs v2.
It scored 0.7608 on Kaggle vs v2's 0.7727 (worse by ~5 correct predictions).

- Flips in bubble zone (P 0.3-0.7): **39**
- Flips OUTSIDE bubble zone: **0**

- Flipped survived -> died: **34**
- Flipped died -> survived: **5**

### 12a Flipped Passengers

| PID | P(Surv) | v2 Pred | 12a Pred | Sex | Pcl | Title | FlipRate | In Bubble? |
|---:|---:|---:|---:|:---|---:|:---|---:|:---|
| 1284 | 0.431 | 0 | 1 | male | 3 | Master | 10% | YES |
| 1144 | 0.503 | 1 | 0 | male | 1 | Mr | 56% | YES |
| 1208 | 0.510 | 1 | 0 | male | 1 | Mr | 50% | YES |
| 1023 | 0.523 | 1 | 0 | male | 1 | Rare | 39% | YES |
| 910 | 0.539 | 1 | 0 | female | 3 | Miss | 11% | YES |
| 1268 | 0.540 | 1 | 0 | female | 3 | Miss | 10% | YES |
| 1237 | 0.540 | 1 | 0 | female | 3 | Miss | 14% | YES |
| 1304 | 0.540 | 1 | 0 | female | 3 | Miss | 14% | YES |
| 1089 | 0.540 | 1 | 0 | female | 3 | Miss | 14% | YES |
| 990 | 0.540 | 1 | 0 | female | 3 | Miss | 14% | YES |
| 1049 | 0.540 | 1 | 0 | female | 3 | Miss | 14% | YES |
| 964 | 0.540 | 1 | 0 | female | 3 | Miss | 14% | YES |
| 979 | 0.540 | 1 | 0 | female | 3 | Miss | 14% | YES |
| 928 | 0.540 | 1 | 0 | female | 3 | Miss | 14% | YES |
| 1030 | 0.540 | 1 | 0 | female | 3 | Miss | 14% | YES |
| 1160 | 0.540 | 1 | 0 | female | 3 | Miss | 14% | YES |
| 1172 | 0.541 | 1 | 0 | female | 3 | Miss | 14% | YES |
| 929 | 0.541 | 1 | 0 | female | 3 | Miss | 14% | YES |
| 1061 | 0.541 | 1 | 0 | female | 3 | Miss | 14% | YES |
| 1296 | 0.544 | 0 | 1 | male | 1 | Mr | 18% | YES |
| 1017 | 0.545 | 1 | 0 | female | 3 | Miss | 10% | YES |
| 965 | 0.545 | 0 | 1 | male | 1 | Mr | 19% | YES |
| 1299 | 0.553 | 1 | 0 | male | 1 | Mr | 17% | YES |
| 967 | 0.554 | 1 | 0 | male | 1 | Mr | 20% | YES |
| 1128 | 0.576 | 0 | 1 | male | 1 | Mr | 10% | YES |
| 1073 | 0.581 | 0 | 1 | male | 1 | Mr | 9% | YES |
| 893 | 0.582 | 1 | 0 | female | 3 | Mrs | 1% | YES |
| 1045 | 0.586 | 1 | 0 | female | 3 | Mrs | 1% | YES |
| 896 | 0.586 | 1 | 0 | female | 3 | Mrs | 1% | YES |
| 1051 | 0.587 | 1 | 0 | female | 3 | Mrs | 0% | YES |
| 982 | 0.587 | 1 | 0 | female | 3 | Mrs | 0% | YES |
| 1201 | 0.587 | 1 | 0 | female | 3 | Mrs | 0% | YES |
| 1251 | 0.588 | 1 | 0 | female | 3 | Mrs | 0% | YES |
| 941 | 0.588 | 1 | 0 | female | 3 | Mrs | 0% | YES |
| 1275 | 0.588 | 1 | 0 | female | 3 | Mrs | 0% | YES |
| 924 | 0.591 | 1 | 0 | female | 3 | Mrs | 0% | YES |
| 1057 | 0.592 | 1 | 0 | female | 3 | Mrs | 0% | YES |
| 925 | 0.593 | 1 | 0 | female | 3 | Mrs | 0% | YES |
| 956 | 0.693 | 1 | 0 | male | 1 | Master | 0% | YES |

## 7. Key Insights

### Why 12a Scored Worse

The 12a model aggressively flipped 3rd-class females from survived to died.
Since 12a scored worse on Kaggle, the v2 model's more optimistic predictions
for 3rd-class females appear closer to the test set truth.

### Where Improvement Points Can Come From

- **4** passengers have P within 2% of decision boundary (0.48-0.52)
- **38** passengers have P within 5% of decision boundary (0.45-0.55)
- **128** passengers in broad bubble zone (0.3-0.7)

### Actionable Takeaway

1. **Male 1st class borderline cases**: Safer targets for improvement than female 3rd class
2. **Don't flip 3rd-class females to died**: The 12a experiment proved this hurts
3. **Threshold tuning**: Small threshold changes could flip a few borderline cases
4. **Any improvement must change <15 test predictions** to be trustworthy