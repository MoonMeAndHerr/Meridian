# MERIDIAN: Mapping Skill Mismatches & Nowcasting Economic Vulnerability in Pahang

[![Competition](https://img.shields.io/badge/Competition-DAX_Challenge_2026-blue.svg)](https://ump.edu.my)
[![Track](https://img.shields.io/badge/Domain-Economy_(DOSM)-teal.svg)]()
[![Target](https://img.shields.io/badge/Focus-Negeri_Pahang-gold.svg)]()

## Overview

Pahang's headline unemployment rate appears remarkably stable (~1.9% as of early 2026). However, this metric masks a critical "Illusion of Employment." According to the Multidimensional Poverty Index (MPI), **52.9% of poverty in Pahang is driven purely by income deprivation**. At the same time, an estimated 40% of tertiary-educated workers in the state are trapped in jobs below their skill level. 

The problem isn't a lack of jobs; it is a severe **Skill-Related Underemployment (SRU)** crisis. 

**MERIDIAN** is a machine-learning dashboard designed to solve this exact bottleneck. Rather than proposing broad, blanket employment schemes that simply create more low-wage jobs, MERIDIAN is engineered to **map specific skill-to-job mismatches across Pahang’s key industries**. By doing so, it allows state authorities to execute surgical, targeted policy interventions exactly where they are needed most.

### The AI Engine (Two-Stage Pipeline)
1. **Unsupervised Spatial & Sectoral Clustering (K-Means):** Groups local economic sectors and demographic data to identify which specific industries are suffering the worst skill-to-job mismatches.
2. **Supervised Regression Nowcasting (XGBoost):** Statistically downscales macro-indicators to predict real-time, district-level purchasing power deficits. 
3. **Interactive Policy Simulator:** An interactive console empowering UPEN Pahang to test preemptive "What-If" investment scenarios. State planners can visualize how targeting specific SME grants and high-tech industrial matching will absorb underemployed graduates and reduce vulnerability down to the district level.

---

### Core Objectives
* **Isolate the Anomaly:** Prove why headline unemployment is an insufficient metric for Pahang's current economic reality.
* **Map the Mismatch:** Pinpoint exactly which local industries are failing to absorb tertiary-educated talent.
* **Targeted Interventions:** Shift state policy away from broad job creation toward targeted capital allocation, ensuring public funds directly address the income-to-living-cost gap.
