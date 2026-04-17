\# CDK2 QSAR Model



Predicting CDK2 inhibitor potency from molecular structure using machine learning on ChEMBL bioactivity data.



\## Key Result



![SAR Summary](https://raw.githubusercontent.com/sandvikerik-spec/cdk2-qsar/main/figures/sar_summary.png)


XGBoost model trained on 2,001 CDK2 inhibitors across 857 unique Murcko scaffolds.  

Scaffold-split test set: \*\*R²=0.39, RMSE=0.96 pIC50 units\*\*.



\*\*Key SAR Finding:\*\* SHAP analysis identified an aniline NH (hinge-binding donor to Leu83) as the near-universal CDK2 pharmacophore (present in \~93% of active compounds), with sulfonamide-containing compounds showing \*\*6.8x higher median potency\*\* than non-sulfonamide compounds.



\## Approach



\### Data

\- Source: ChEMBL CHEMBL301 (Cyclin-dependent kinase 2, single protein)

\- Filters: biochemical IC50 only, standard relation "=", nM units

\- Final dataset: 2,001 compounds, pIC50 range 3.46–9.52, 857 unique Murcko scaffolds



\### Featurization

\- 1024-bit ECFP4 Morgan fingerprints (radius=2)

\- 10 RDKit physicochemical descriptors (MolWt, LogP, TPSA, HBD, HBA, RotBonds, Rings, ArRings, FSP3, Heteroatoms)

\- Total: 1,034 features per compound



\### Split Strategy

Scaffold-based train/test split (80/20) using Murcko scaffolds — entire scaffold families assigned to either train or test, never split across both. This tests generalization to genuinely new chemotypes rather than close analogs of training compounds.



\### Models

\- Random Forest (n=200 trees)

\- XGBoost (n=300 trees, lr=0.05)

\- SHAP TreeExplainer for interpretability



\## Results



| Model | R² (vs 1:1) | R² (regression) | Pearson r | Slope | RMSE |
|-------|-------------|-----------------|-----------|-------|------|
| Random Forest | 0.362 | 0.405 | 0.637 | 0.354 | 0.980 |
| XGBoost | 0.389 | 0.419 | 0.647 | 0.411 | 0.959 |

![Regression Analysis](https://raw.githubusercontent.com/sandvikerik-spec/cdk2-qsar/main/figures/regression_analysis.png)

### Interpreting the metrics

**R² vs 1:1 (0.39)** — strict accuracy against perfect prediction. Penalizes the systematic compression of predictions toward the mean (slope=0.41).

**R² regression (0.42)** — variance in activity explained by the linear trend in predictions. Closer to the model's true explanatory power.

**Pearson r (0.65)** — rank correlation between predicted and measured potency. The most relevant metric for drug discovery triage: the model correctly rank-orders compounds with enough reliability to enrich potent hits ~2-3x vs random selection when prioritizing the top 20% of a virtual library.

**Slope (0.41)** — prediction compression toward the mean. A known property of ensemble tree models on scaffold-split data — the model hedges toward the center of the training distribution for novel chemotypes. Useful for ranking, not absolute potency prediction.


![Predicted vs Actual](https://raw.githubusercontent.com/sandvikerik-spec/cdk2-qsar/main/figures/predicted_vs_actual.png)


\## SAR Findings



![SHAP Summary](https://raw.githubusercontent.com/sandvikerik-spec/cdk2-qsar/main/figures/shap_summary.png)



Top features by mean absolute SHAP value:



| Feature | Type | Chemical meaning |

|---------|------|-----------------|

| MolWt | PhysChem | Size required to fill ATP pocket |

| TPSA | PhysChem | Polar surface for hinge/DFG H-bonds |

| bit\_128 | ECFP4 | Aniline NH — universal hinge binder (Leu83) |

| bit\_350 | ECFP4 | Sulfonamide NH — secondary H-bond donor |

| bit\_319 | ECFP4 | Sulfonamide on specific ring system |



\### Sulfonamide potency advantage



![Sulfonamide Activity](https://raw.githubusercontent.com/sandvikerik-spec/cdk2-qsar/main/figures/bit350_sulfonamide_activity.png)



Compounds containing sulfonamide substructures (n=411) show mean pIC50 of 7.24 vs 6.41 for non-sulfonamide compounds — a \*\*6.8x improvement in IC50\*\*.



\### Pharmacophoric stratification



![SAR Summary](https://raw.githubusercontent.com/sandvikerik-spec/cdk2-qsar/main/figures/sar_summary.png)


| Group | n | Mean pIC50 |

|-------|---|-----------|

| Aniline + Sulfonamide | \~200 | \~7.5 |

| Aniline only | \~1,200 | \~6.8 |

| Neither | \~376 | \~6.0 |



\## Why R²=0.39 is the honest result



Random train/test splits on this dataset yield R²\~0.70 by leaking structural information — close analogs of test compounds appear in training. The scaffold split is the realistic drug discovery test: can the model generalize to new chemotypes? Published QSAR benchmarks on ChEMBL kinase datasets with scaffold splits typically report R²=0.35–0.55. This model is within that range and its predictions are useful for compound triage and rank-ordering, not precise potency prediction.



\## Requirements



```bash

pip install -r requirements.txt

```



\## Usage



```bash

jupyter notebook notebook/cdk2\_qsar\_model.ipynb

```



\## Author



Erik Sandvik — Senior Scientist, Drug Discovery  

Built as part of a computational drug discovery portfolio demonstrating integration of cheminformatics, ML, and HTS domain expertise.

