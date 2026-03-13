# Error Analysis Report: v2 Logistic Regression

**Model**: LogisticRegression(C=0.01), 5-fold StratifiedKFold (random_state=42)
**Overall OOF accuracy**: 0.8305 (151 errors / 891 passengers)

## 1. Error Type Breakdown

| Type | Count | % of Errors | Description |
|:---|---:|---:|:---|
| False Positive | 64 | 42.4% | Predicted survived, actually died |
| False Negative | 87 | 57.6% | Predicted died, actually survived |

## 2. Error Rates by Sex x Pclass

| Group | Total | Errors | Error Rate | FP | FN | Survival Rate |
|:---|---:|---:|---:|---:|---:|---:|
| male Pclass=1 | 122 | 41 | 33.6% | 7 | 34 | 36.9% |
| male Pclass=2 | 108 | 8 | 7.4% | 0 | 8 | 15.7% |
| male Pclass=3 | 347 | 41 | 11.8% | 0 | 41 | 13.5% |
| female Pclass=1 | 94 | 3 | 3.2% | 3 | 0 | 96.8% |
| female Pclass=2 | 76 | 7 | 9.2% | 6 | 1 | 92.1% |
| female Pclass=3 | 144 | 51 | 35.4% | 48 | 3 | 50.0% |

## 3. Error Rates by Title

| Title | Total | Errors | Error Rate | FP | FN |
|:---|---:|---:|---:|---:|---:|
| Mr | 517 | 79 | 15.3% | 5 | 74 |
| Mrs | 126 | 21 | 16.7% | 19 | 2 |
| Miss | 185 | 40 | 21.6% | 38 | 2 |
| Master | 40 | 5 | 12.5% | 0 | 5 |
| Rare | 23 | 6 | 26.1% | 2 | 4 |

## 4. Probability Distribution of Misclassified Passengers

| Probability Range | Errors | % of Errors |
|:---|---:|---:|
| [0.1-0.2) | 41 | 27.2% |
| [0.2-0.3) | 10 | 6.6% |
| [0.3-0.4) | 10 | 6.6% |
| [0.4-0.5) | 26 | 17.2% |
| [0.5-0.6) | 39 | 25.8% |
| [0.6-0.7) | 15 | 9.9% |
| [0.7-0.8) | 6 | 4.0% |
| [0.8-0.9) | 4 | 2.6% |

- **Near boundary (0.3-0.7)**: 90 (59.6%)
- **Confident errors (<0.3 or >0.7)**: 61 (40.4%)

## 5. Most Confident Errors (Top 20)

| PID | True | Pred | Prob | Sex | Pcl | Age | Fare | Title | FamSz | Cabin | Emb |
|---:|---:|---:|---:|:---|---:|---:|---:|:---|---:|:---|:---|
| 773 | 0 | 1 | 0.885 | female | 2 | 57 | 10.50 | Mrs | 0 | E77 | S |
| 298 | 0 | 1 | 0.884 | female | 1 | 2 | 151.55 | Miss | 3 | C22 C2 | S |
| 570 | 1 | 0 | 0.121 | male | 3 | 32 | 7.85 | Mr | 0 | - | S |
| 745 | 1 | 0 | 0.121 | male | 3 | 31 | 7.92 | Mr | 0 | - | S |
| 205 | 1 | 0 | 0.121 | male | 3 | 18 | 8.05 | Mr | 0 | - | S |
| 805 | 1 | 0 | 0.122 | male | 3 | 27 | 6.97 | Mr | 0 | - | S |
| 392 | 1 | 0 | 0.122 | male | 3 | 21 | 7.80 | Mr | 0 | - | S |
| 580 | 1 | 0 | 0.122 | male | 3 | 32 | 7.92 | Mr | 0 | - | S |
| 339 | 1 | 0 | 0.122 | male | 3 | 45 | 8.05 | Mr | 0 | - | S |
| 221 | 1 | 0 | 0.122 | male | 3 | 16 | 8.05 | Mr | 0 | - | S |
| 445 | 1 | 0 | 0.122 | male | 3 | ? | 8.11 | Mr | 0 | - | S |
| 822 | 1 | 0 | 0.122 | male | 3 | 27 | 8.66 | Mr | 0 | - | S |
| 665 | 1 | 0 | 0.123 | male | 3 | 20 | 7.92 | Mr | 1 | - | S |
| 268 | 1 | 0 | 0.126 | male | 3 | 25 | 7.78 | Mr | 1 | - | S |
| 128 | 1 | 0 | 0.129 | male | 3 | 24 | 7.14 | Mr | 0 | - | S |
| 272 | 1 | 0 | 0.129 | male | 3 | 25 | 0.00 | Mr | 0 | - | S |
| 147 | 1 | 0 | 0.129 | male | 3 | 27 | 7.80 | Mr | 0 | - | S |
| 401 | 1 | 0 | 0.129 | male | 3 | 39 | 7.92 | Mr | 0 | - | S |
| 415 | 1 | 0 | 0.129 | male | 3 | 44 | 7.92 | Mr | 0 | - | S |
| 284 | 1 | 0 | 0.129 | male | 3 | 19 | 8.05 | Mr | 0 | - | S |

## 6. Passengers Barely Correct (prob 0.35-0.65, correct)

**Count**: 124

| Sex x Pclass | Count |
|:---|---:|
| male Pclass=1 | 45 |
| male Pclass=2 | 6 |
| male Pclass=3 | 5 |
| female Pclass=1 | 1 |
| female Pclass=2 | 1 |
| female Pclass=3 | 66 |

## 7. Error Rate by Fold

| Fold | Errors | Total | Error Rate |
|---:|---:|---:|---:|
| 1 | 30 | 179 | 16.8% |
| 2 | 31 | 178 | 17.4% |
| 3 | 29 | 178 | 16.3% |
| 4 | 32 | 178 | 18.0% |
| 5 | 29 | 178 | 16.3% |

## 8. False Positive Analysis (Predicted Survived, Actually Died)

**Total FPs**: 64

- By Sex: {'female': 57, 'male': 7}
- By Pclass: {3: 48, 1: 10, 2: 6}
- By Title: {'Miss': 38, 'Mrs': 19, 'Mr': 5, 'Rare': 2}
- Median age: 26.0
- Median fare: 13.70
- HasCabin rate: 0.20

## 9. False Negative Analysis (Predicted Died, Actually Survived)

**Total FNs**: 87

- By Sex: {'male': 83, 'female': 4}
- By Pclass: {3: 44, 1: 34, 2: 9}
- By Title: {'Mr': 74, 'Master': 5, 'Rare': 4, 'Mrs': 2, 'Miss': 2}
- Median age: 31.0
- Median fare: 23.25
- HasCabin rate: 0.34

## 10. Special Subgroup Analysis

### Women who died: 81
- Model predicted survived (FP): 57 (70.4%)
- By Pclass: {3: 72, 2: 6, 1: 3}

### Men who survived: 109
- Model predicted died (FN): 83 (76.1%)
- By Pclass: {3: 47, 1: 45, 2: 17}

### 3rd Class Women: 144
- Survival rate: 50.0%
- Error rate: 35.4%
- FP: 48, FN: 3

### 1st/2nd Class Men: 230
- Survival rate: 27.0%
- Error rate: 21.3%
- FP: 7, FN: 42

### Children (age <= 12): 69
- Survival rate: 58.0%
- Error rate: 14.5%
- FP: 4, FN: 6

## 11. Error Rate by Family Size

| FamSize | Total | Errors | Error Rate | Survival Rate |
|---:|---:|---:|---:|---:|
| 0 | 537 | 88 | 16.4% | 30.4% |
| 1 | 161 | 28 | 17.4% | 55.3% |
| 2 | 102 | 25 | 24.5% | 57.8% |
| 3 | 29 | 5 | 17.2% | 72.4% |
| 4 | 15 | 0 | 0.0% | 20.0% |
| 5 | 22 | 1 | 4.5% | 13.6% |
| 6 | 12 | 4 | 33.3% | 33.3% |
| 7 | 6 | 0 | 0.0% | 0.0% |
| 10 | 7 | 0 | 0.0% | 0.0% |

## 12. Error Rate by Fare Quartile

| Quartile | Fare Range | Errors | Total | Error Rate |
|:---|:---|---:|---:|---:|
| Q1(low) | [0.0-7.9] | 27 | 223 | 12.1% |
| Q2 | [7.9-14.5] | 47 | 224 | 21.0% |
| Q3 | [14.5-31.0] | 46 | 222 | 20.7% |
| Q4(high) | [31.3-512.3] | 31 | 222 | 14.0% |

## 13. All False Positive Women (Predicted Survived, Actually Died)

| PID | Prob | Pcl | Age | Fare | Title | FamSz | Cabin | Emb | Name |
|---:|---:|---:|---:|---:|:---|---:|:---|:---|:---|
| 773 | 0.885 | 2 | 57 | 10.50 | Mrs | 0 | E77 | S | Mack, Mrs. (Mary) |
| 298 | 0.884 | 1 | 2 | 151.55 | Miss | 3 | C22 C2 | S | Allison, Miss. Helen Loraine |
| 499 | 0.847 | 1 | 25 | 151.55 | Mrs | 3 | C22 C2 | S | Allison, Mrs. Hudson J C (Bessie Waldo D |
| 178 | 0.825 | 1 | 50 | 28.71 | Miss | 0 | C49 | C | Isham, Miss. Ann Elizabeth |
| 313 | 0.749 | 2 | 26 | 26.00 | Mrs | 2 | - | S | Lahtinen, Mrs. William (Anna Sylfven) |
| 855 | 0.749 | 2 | 44 | 26.00 | Mrs | 1 | - | S | Carter, Mrs. Ernest Courtenay (Lilian Hu |
| 206 | 0.737 | 3 | 2 | 10.46 | Miss | 1 | G6 | S | Strom, Miss. Telma Matilda |
| 42 | 0.717 | 2 | 27 | 21.00 | Mrs | 1 | - | S | Turpin, Mrs. William John Robert (Doroth |
| 853 | 0.714 | 3 | 9 | 15.25 | Miss | 2 | - | C | Boulos, Miss. Nourelain |
| 358 | 0.706 | 2 | 38 | 13.00 | Miss | 0 | - | S | Funk, Miss. Annie Clemmer |
| 252 | 0.697 | 3 | 29 | 10.46 | Mrs | 2 | G6 | S | Strom, Mrs. Wilhelm (Elna Matilda Persso |
| 200 | 0.697 | 2 | 24 | 13.00 | Miss | 0 | - | S | Yrois, Miss. Henriette ("Mrs Harbeck") |
| 363 | 0.680 | 3 | 45 | 14.45 | Mrs | 1 | - | C | Barbara, Mrs. (Catherine David) |
| 579 | 0.676 | 3 | ? | 14.46 | Mrs | 1 | - | C | Caram, Mrs. Joseph (Maria Elias) |
| 658 | 0.669 | 3 | 32 | 15.50 | Mrs | 2 | - | Q | Bourke, Mrs. John (Catherine) |
| 141 | 0.655 | 3 | ? | 15.25 | Mrs | 2 | - | C | Boulos, Mrs. Joseph (Sultana) |
| 112 | 0.633 | 3 | 14 | 14.45 | Miss | 1 | - | C | Zabour, Miss. Hileni |
| 420 | 0.632 | 3 | 10 | 24.15 | Miss | 2 | - | S | Van Impe, Miss. Catharina |
| 115 | 0.627 | 3 | 17 | 14.46 | Miss | 0 | - | C | Attalah, Miss. Malake |
| 594 | 0.627 | 3 | ? | 7.75 | Miss | 2 | - | Q | Bourke, Miss. Mary |
| 703 | 0.623 | 3 | 18 | 14.45 | Miss | 1 | - | C | Barbara, Miss. Saiide |
| 241 | 0.623 | 3 | ? | 14.45 | Miss | 1 | - | C | Zabour, Miss. Thamine |
| 502 | 0.616 | 3 | 21 | 7.75 | Miss | 0 | - | Q | Canavan, Miss. Mary |
| 19 | 0.602 | 3 | 31 | 18.00 | Mrs | 1 | - | S | Vander Planke, Mrs. Julius (Emelia Maria |
| 800 | 0.599 | 3 | 30 | 24.15 | Mrs | 2 | - | S | Van Impe, Mrs. Jean Baptiste (Rosalie Pa |
| 618 | 0.594 | 3 | 26 | 16.10 | Mrs | 1 | - | S | Lobb, Mrs. William Arthur (Cordelia K St |
| 265 | 0.586 | 3 | ? | 7.75 | Miss | 0 | - | Q | Henry, Miss. Delia |
| 503 | 0.586 | 3 | ? | 7.63 | Miss | 0 | - | Q | O'Sullivan, Miss. Bridget Mary |
| 416 | 0.585 | 3 | ? | 8.05 | Mrs | 0 | - | S | Meek, Mrs. Thomas (Annie Louise Rowley) |
| 655 | 0.583 | 3 | 18 | 6.75 | Miss | 0 | - | Q | Hegarty, Miss. Hanora "Nora" |
| 41 | 0.580 | 3 | 40 | 9.47 | Mrs | 1 | - | S | Ahlin, Mrs. Johan (Johanna Persdotter La |
| 50 | 0.574 | 3 | 18 | 17.80 | Mrs | 1 | - | S | Arnold-Franchi, Mrs. Josef (Josefine Fra |
| 889 | 0.566 | 3 | ? | 23.45 | Miss | 3 | - | S | Johnston, Miss. Catherine Helen "Carrie" |
| 681 | 0.564 | 3 | ? | 8.14 | Miss | 0 | - | Q | Peters, Miss. Katie |
| 768 | 0.564 | 3 | 30 | 7.75 | Miss | 0 | - | Q | Mangan, Miss. Mary |
| 255 | 0.562 | 3 | 41 | 20.21 | Mrs | 2 | - | S | Rosblom, Mrs. Viktor (Helena Wilhelmina) |
| 39 | 0.562 | 3 | 18 | 18.00 | Miss | 2 | - | S | Vander Planke, Miss. Augusta Maria |
| 133 | 0.559 | 3 | 47 | 14.50 | Mrs | 1 | - | S | Robins, Mrs. Alexander A (Grace Charity  |
| 424 | 0.559 | 3 | 28 | 14.40 | Mrs | 2 | - | S | Danbom, Mrs. Ernst Gilbert (Anna Sigrid  |
| 535 | 0.545 | 3 | 30 | 8.66 | Miss | 0 | - | S | Cacic, Miss. Marija |
| 15 | 0.545 | 3 | 14 | 7.85 | Miss | 0 | - | S | Vestrom, Miss. Hulda Amanda Adolfina |
| 397 | 0.545 | 3 | 31 | 7.85 | Miss | 0 | - | S | Olsson, Miss. Elina |
| 114 | 0.542 | 3 | 20 | 9.82 | Miss | 1 | - | S | Jussila, Miss. Katriina |
| 730 | 0.541 | 3 | 25 | 7.92 | Miss | 1 | - | S | Ilmakangas, Miss. Pieta Sofia |
| 883 | 0.539 | 3 | 22 | 10.52 | Miss | 0 | - | S | Dahlberg, Miss. Gerda Ulrika |
| 475 | 0.539 | 3 | 22 | 9.84 | Miss | 0 | - | S | Strandberg, Miss. Ida Sofia |
| 504 | 0.538 | 3 | 37 | 9.59 | Miss | 0 | - | S | Laitinen, Miss. Kristina Sofia |
| 405 | 0.538 | 3 | 20 | 8.66 | Miss | 0 | - | S | Oreskovic, Miss. Marija |
| 247 | 0.537 | 3 | 25 | 7.78 | Miss | 0 | - | S | Lindahl, Miss. Agda Thorilda Viktoria |
| 277 | 0.537 | 3 | 45 | 7.75 | Miss | 0 | - | S | Lindblom, Miss. Augusta Charlotta |
| 294 | 0.533 | 3 | 24 | 8.85 | Miss | 0 | - | S | Haas, Miss. Aloisia |
| 236 | 0.532 | 3 | ? | 7.55 | Miss | 0 | - | S | Harknett, Miss. Alice Phoebe |
| 403 | 0.532 | 3 | 21 | 9.82 | Miss | 1 | - | S | Jussila, Miss. Mari Aina |
| 817 | 0.528 | 3 | 23 | 7.92 | Miss | 0 | - | S | Heininen, Miss. Wendla Maria |
| 565 | 0.523 | 3 | ? | 8.05 | Miss | 0 | - | S | Meanwell, Miss. (Marion Ogden) |
| 101 | 0.523 | 3 | 28 | 7.90 | Miss | 0 | - | S | Petranec, Miss. Matilda |
| 808 | 0.523 | 3 | 18 | 7.78 | Miss | 0 | - | S | Pettersson, Miss. Ellen Natalia |

## 14. All False Negative Men (Predicted Died, Actually Survived)

| PID | Prob | Pcl | Age | Fare | Title | FamSz | Cabin | Emb | Name |
|---:|---:|---:|---:|---:|:---|---:|:---|:---|:---|
| 570 | 0.121 | 3 | 32 | 7.85 | Mr | 0 | - | S | Jonsson, Mr. Carl |
| 745 | 0.121 | 3 | 31 | 7.92 | Mr | 0 | - | S | Stranden, Mr. Juho |
| 205 | 0.121 | 3 | 18 | 8.05 | Mr | 0 | - | S | Cohen, Mr. Gurshon "Gus" |
| 805 | 0.122 | 3 | 27 | 6.97 | Mr | 0 | - | S | Hedman, Mr. Oskar Arvid |
| 392 | 0.122 | 3 | 21 | 7.80 | Mr | 0 | - | S | Jansson, Mr. Carl Olof |
| 580 | 0.122 | 3 | 32 | 7.92 | Mr | 0 | - | S | Jussila, Mr. Eiriik |
| 339 | 0.122 | 3 | 45 | 8.05 | Mr | 0 | - | S | Dahl, Mr. Karl Edwart |
| 221 | 0.122 | 3 | 16 | 8.05 | Mr | 0 | - | S | Sunderland, Mr. Victor Francis |
| 445 | 0.122 | 3 | ? | 8.11 | Mr | 0 | - | S | Johannesen-Bratthammer, Mr. Bernt |
| 822 | 0.122 | 3 | 27 | 8.66 | Mr | 0 | - | S | Lulic, Mr. Nikola |
| 665 | 0.123 | 3 | 20 | 7.92 | Mr | 1 | - | S | Lindqvist, Mr. Eino William |
| 268 | 0.126 | 3 | 25 | 7.78 | Mr | 1 | - | S | Persson, Mr. Ernst Ulrik |
| 128 | 0.129 | 3 | 24 | 7.14 | Mr | 0 | - | S | Madsen, Mr. Fridtjof Arne |
| 272 | 0.129 | 3 | 25 | 0.00 | Mr | 0 | - | S | Tornquist, Mr. William Henry |
| 147 | 0.129 | 3 | 27 | 7.80 | Mr | 0 | - | S | Andersson, Mr. August Edvard ("Wennerstr |
| 415 | 0.129 | 3 | 44 | 7.92 | Mr | 0 | - | S | Sundman, Mr. Johan Julian |
| 401 | 0.129 | 3 | 39 | 7.92 | Mr | 0 | - | S | Niskanen, Mr. Juha |
| 284 | 0.129 | 3 | 19 | 8.05 | Mr | 0 | - | S | Dorking, Mr. Edward Arthur |
| 287 | 0.129 | 3 | 30 | 9.50 | Mr | 0 | - | S | de Mulder, Mr. Theodore |
| 108 | 0.131 | 3 | ? | 7.78 | Mr | 0 | - | S | Moss, Mr. Albert Johan |
| 82 | 0.132 | 3 | 29 | 9.50 | Mr | 0 | - | S | Sheerlinck, Mr. Jan Baptist |
| 644 | 0.134 | 3 | ? | 56.50 | Mr | 0 | - | S | Foo, Mr. Choong |
| 75 | 0.134 | 3 | 32 | 56.50 | Mr | 0 | - | S | Bing, Mr. Lee |
| 510 | 0.136 | 3 | 26 | 56.50 | Mr | 0 | - | S | Lang, Mr. Fang |
| 511 | 0.137 | 3 | 29 | 7.75 | Mr | 0 | - | Q | Daly, Mr. Eugene Patrick |
| 693 | 0.143 | 3 | ? | 56.50 | Mr | 0 | - | S | Lam, Mr. Ali |
| 839 | 0.143 | 3 | 32 | 56.50 | Mr | 0 | - | S | Chip, Mr. Chang |
| 829 | 0.153 | 3 | ? | 7.75 | Mr | 0 | - | Q | McCormack, Mr. Thomas Joseph |
| 554 | 0.158 | 3 | 22 | 7.22 | Mr | 0 | - | C | Leeni, Mr. Fahim ("Philip Zenni") |
| 18 | 0.162 | 2 | ? | 13.00 | Mr | 0 | - | S | Williams, Mr. Charles Eugene |
| 674 | 0.162 | 2 | 31 | 13.00 | Mr | 0 | - | S | Wilhelms, Mr. Charles |
| 37 | 0.164 | 3 | ? | 7.23 | Mr | 0 | - | C | Mamee, Mr. Hanna |
| 456 | 0.164 | 3 | 29 | 7.90 | Mr | 0 | - | C | Jalsevac, Mr. Ivan |
| 302 | 0.169 | 3 | ? | 23.25 | Mr | 2 | - | Q | McCoy, Mr. Bernard |
| 571 | 0.175 | 2 | 62 | 10.50 | Mr | 0 | - | S | Harris, Mr. George |
| 227 | 0.175 | 2 | 19 | 10.50 | Mr | 0 | - | S | Mellors, Mr. William John |
| 289 | 0.175 | 2 | 42 | 13.00 | Mr | 0 | - | S | Hosono, Mr. Masabumi |
| 623 | 0.180 | 3 | 20 | 15.74 | Mr | 2 | - | C | Nakid, Mr. Sahid |
| 208 | 0.181 | 3 | 26 | 18.79 | Mr | 0 | - | C | Albimona, Mr. Nassef Cassem |
| 544 | 0.182 | 2 | 32 | 26.00 | Mr | 1 | - | S | Beane, Mr. Edward |
| 763 | 0.184 | 3 | 20 | 7.23 | Mr | 0 | - | C | Barah, Mr. Hanna Assi |
| 188 | 0.223 | 1 | 45 | 26.55 | Mr | 0 | - | S | Romaine, Mr. Charles Hallace ("Mr C Rolm |
| 448 | 0.223 | 1 | 34 | 26.55 | Mr | 0 | - | S | Seward, Mr. Frederic Kimber |
| 508 | 0.230 | 1 | ? | 26.55 | Mr | 0 | - | S | Bradley, Mr. George ("George Arthur Bray |
| 608 | 0.231 | 1 | 27 | 30.50 | Mr | 0 | - | S | Daniel, Mr. Robert Williams |
| 548 | 0.234 | 2 | ? | 13.86 | Mr | 0 | - | C | Padro y Manent, Mr. Julian |
| 262 | 0.255 | 3 | 3 | 31.39 | Master | 6 | - | S | Asplund, Master. Edvin Rojj Felix |
| 430 | 0.267 | 3 | 32 | 8.05 | Mr | 0 | E10 | S | Pickard, Mr. Berk (Berk Trembisky) |
| 605 | 0.289 | 1 | 35 | 26.55 | Mr | 0 | - | C | Homer, Mr. Harry ("Mr E Haven") |
| 431 | 0.337 | 1 | 28 | 26.55 | Mr | 0 | C52 | S | Bjornstrom-Steffansson, Mr. Mauritz Haka |
| 24 | 0.342 | 1 | 28 | 35.50 | Mr | 0 | A6 | S | Sloper, Mr. William Thompson |
| 631 | 0.343 | 1 | 80 | 30.00 | Mr | 0 | A23 | S | Barkworth, Mr. Algernon Henry Wilson |
| 56 | 0.345 | 1 | ? | 35.50 | Mr | 0 | C52 | S | Woolner, Mr. Hugh |
| 299 | 0.349 | 1 | ? | 30.50 | Mr | 0 | C106 | S | Saalfeld, Mr. Adolphe |
| 713 | 0.365 | 1 | 48 | 52.00 | Mr | 1 | C126 | S | Taylor, Mr. Elmer Zebley |
| 225 | 0.367 | 1 | 38 | 90.00 | Mr | 1 | C93 | S | Hoyt, Mr. Frederick Maxfield |
| 691 | 0.369 | 1 | 31 | 57.00 | Mr | 1 | B20 | S | Dick, Mr. Albert Adrian |
| 22 | 0.371 | 2 | 34 | 13.00 | Mr | 0 | D56 | S | Beesley, Mr. Lawrence |
| 661 | 0.397 | 1 | 50 | 133.65 | Rare | 2 | - | S | Frauenthal, Dr. Henry William |
| 210 | 0.414 | 1 | 40 | 31.00 | Mr | 0 | A31 | C | Blank, Mr. Henry |
| 450 | 0.420 | 1 | 52 | 30.50 | Rare | 0 | C104 | S | Peuchen, Major. Arthur Godfrey |
| 391 | 0.422 | 1 | 36 | 120.00 | Mr | 3 | B96 B9 | S | Carter, Mr. William Ernest |
| 708 | 0.423 | 1 | 42 | 26.29 | Mr | 0 | E24 | S | Calderhead, Mr. Edward Pennington |
| 840 | 0.430 | 1 | ? | 29.70 | Mr | 0 | C47 | C | Marechal, Mr. Pierre |
| 890 | 0.434 | 1 | 26 | 30.00 | Mr | 0 | C148 | C | Behr, Mr. Karl Howell |
| 858 | 0.437 | 1 | 51 | 26.55 | Mr | 0 | E17 | S | Daly, Mr. Peter Denis  |
| 461 | 0.437 | 1 | 48 | 26.55 | Mr | 0 | E12 | S | Anderson, Mr. Harry |
| 513 | 0.439 | 1 | 36 | 26.29 | Mr | 0 | E25 | S | McGough, Mr. James Robert |
| 741 | 0.442 | 1 | ? | 30.00 | Mr | 0 | D45 | S | Hawksford, Mr. Walter James |
| 702 | 0.445 | 1 | 35 | 26.29 | Mr | 0 | E24 | S | Silverthorne, Mr. Spencer Victor |
| 573 | 0.445 | 1 | 36 | 26.39 | Mr | 0 | E25 | S | Flynn, Mr. John Irwin ("Irving") |
| 725 | 0.447 | 1 | 27 | 53.10 | Mr | 1 | E8 | S | Chambers, Mr. Norman Campbell |
| 249 | 0.455 | 1 | 37 | 52.55 | Mr | 2 | D35 | S | Beckwith, Mr. Richard Leonard |
| 622 | 0.457 | 1 | 42 | 52.55 | Mr | 1 | D19 | S | Kimball, Mr. Edwin Nelson Jr |
| 454 | 0.461 | 1 | 49 | 89.10 | Mr | 1 | C92 | C | Goldenberg, Mr. Samuel L |
| 588 | 0.468 | 1 | 60 | 79.20 | Mr | 2 | B41 | C | Frolicher-Stehli, Mr. Maxmillian |
| 633 | 0.479 | 1 | 32 | 30.50 | Rare | 0 | B50 | C | Stahelin-Maeglin, Dr. Max |
| 870 | 0.483 | 3 | 4 | 11.13 | Master | 2 | - | S | Johnson, Master. Harold Theodor |
| 551 | 0.487 | 1 | 17 | 110.88 | Mr | 2 | C70 | C | Thayer, Mr. John Borland Jr |
| 349 | 0.489 | 3 | 3 | 15.90 | Master | 2 | - | S | Coutts, Master. William Loch "William" |
| 600 | 0.489 | 1 | 49 | 56.93 | Rare | 1 | A20 | C | Duff Gordon, Sir. Cosmo Edmund ("Mr Morg |
| 166 | 0.490 | 3 | 9 | 20.52 | Master | 2 | - | S | Goldsmith, Master. Frank John William "F |
| 789 | 0.492 | 3 | 1 | 20.57 | Master | 3 | - | S | Dean, Master. Bertram Vere |

## 15. Actionable Findings Summary

**Total errors**: 151/891 (FP: 64, FN: 87)

### Error Concentration by Sex x Pclass

| Group | Errors | Error Rate | % of All Errors |
|:---|---:|---:|---:|
| female_Pclass3 | 51 | 35.4% | 33.8% |
| male_Pclass1 | 41 | 33.6% | 27.2% |
| male_Pclass3 | 41 | 11.8% | 27.2% |
| male_Pclass2 | 8 | 7.4% | 5.3% |
| female_Pclass2 | 7 | 9.2% | 4.6% |
| female_Pclass1 | 3 | 3.2% | 2.0% |

**Largest error source**: female_Pclass3 with 51 errors
If we could fix all errors in this group: +5.7pp accuracy

### Swing Passengers (prob 0.4-0.6)

- Total: 158
- Correct: 93
- Wrong: 65
- Accuracy in swing zone: 58.9%

| Sex x Pclass | Count | Accuracy |
|:---|---:|---:|
| male Pclass=1 | 60 | 56.7% |
| male Pclass=2 | 5 | 100.0% |
| male Pclass=3 | 9 | 55.6% |
| female Pclass=1 | 1 | 100.0% |
| female Pclass=2 | 2 | 50.0% |
| female Pclass=3 | 81 | 58.0% |
