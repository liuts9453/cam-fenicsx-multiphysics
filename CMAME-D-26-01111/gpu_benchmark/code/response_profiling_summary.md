# Heterogeneous-Computing Performance Bottleneck Analysis (latest logs, $1^3 \to 30^3$)

These data were prepared to address CMAME reviewer questions about the crossover point and computational bottlenecks, including local Newton iterations, matrix exponentials, and automatic-differentiation overhead. All values were extracted from the full long-running benchmark logs.

## 1. Profiling Data Table

| Mesh size | Platform | Total time | **First-call/JIT cost** | **Pure constitutive compute<br>(average per call)** | **Communication<br>(H2D+D2H total)** | **FEM assembly and solve<br>(total)** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1^3** | CPU | 15.01 s | 6.29 s | **0.0009 s** | 0.0673 s | 8.32 s |
| **1^3** | **GPU** | 32.01 s | 12.82 s | **0.0031 s** | 0.1173 s | 17.86 s |
| | | | | | | |
| **3^3** | CPU | 25.36 s | 6.33 s | **0.0125 s** | 0.1099 s | 13.27 s |
| **3^3** | **GPU** | 32.59 s | 12.75 s | **0.0050 s** | 0.1421 s | 17.48 s |
| | | | | | | |
| **5^3** | CPU | 49.83 s | 6.40 s | **0.0402 s** | 0.0980 s | 26.79 s |
| **5^3** | **GPU** | 43.67 s | 13.36 s | **0.0123 s** | 0.1539 s | 25.13 s |
| | | | | | | |
| **7^3** | CPU | 107.42 s | 6.55 s | **0.0875 s** | 0.1339 s | 63.11 s |
| **7^3** | **GPU** | 62.00 s | 13.04 s | **0.0274 s** | 0.2081 s | 37.02 s |
| | | | | | | |
| **10^3** | CPU | 301.98 s | 7.19 s | **0.2914 s** | 0.1225 s | 178.70 s |
| **10^3** | **GPU** | 105.03 s | 13.46 s | **0.0741 s** | 0.3470 s | 61.77 s |
| | | | | | | |
| **20^3** | CPU | 3214.96 s | 15.61 s | **3.2606 s** | 1.1655 s | 1796.12 s |
| **20^3** | **GPU** | 748.52 s | 19.79 s | **0.6622 s** | 2.1353 s | 441.87 s |
| | | | | | | |
| **30^3** | CPU | 12207.38 s | 41.66 s | **11.3012 s** | 21.9208 s | 6809.63 s |
| **30^3** | **GPU** | 3258.90 s | 24.16 s | **2.7874 s** | 6.2744 s | 1912.82 s |

*Note: small discrepancies can appear because of system-call timing and rounding. The intended decomposition is: total time is approximately first-call/JIT cost plus total pure constitutive compute time plus total communication time plus assembly/solve time.*

---
