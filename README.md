# Knowledge-Graph-Constrained Generative Probabilistic Planning (KG-GPP)

Reproducibility package for:

**Trustworthy AI and Risk-Aware Planning for Global Supply Chains: A Probabilistic Framework with Knowledge Graph Constraints**

## Repository Structure

```text
project_root/
├── code/
│   └── run_r3_experiments.py
├── data/
│   └── README.md
├── results/                 # generated experiment outputs
├── logs/
│   └── run_metadata.json
├── requirements-core.txt
├── requirements-manuscript.txt
└── README.md
```

## 1. Clone the Repository

```bash
git clone <GITHUB_REPOSITORY_URL>
cd KG_GPP_EJOR_GitHub
```

Replace `<GITHUB_REPOSITORY_URL>` with the permanent public repository URL after
the repository has been created.

## 2. Install Dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-core.txt
```

The complete environment reported for the manuscript is listed in
`requirements-manuscript.txt`.

## 3. Download and Place the DataCo Dataset

Download the public DataCo Supply Chain dataset from:

<https://data.mendeley.com/datasets/8gx2fvg2k6/3>

Place the downloaded file here:

```text
project_root/
└── data/
    └── DataCoSupplyChainDataset.csv
```

The raw dataset is not committed to this repository. DataCo does not contain
ground-truth labels for ideological risk, trust, or supplier bias; trust and
disparity are operational proxies defined in
`results/dataco_proxy_definitions.csv`.

## 4. Run the Experiments

From the repository root, run the benchmark, robustness, and complexity analyses:

```bash
python code/run_r3_experiments.py \
  --data-path data/DataCoSupplyChainDataset.csv \
  --output-dir .
```

Aggregate outputs are written to `results/`, and run configuration is written to
`logs/run_metadata.json`. Individual outcomes for every evaluated instance, policy,
and scenario are stored in `results/instance_results.csv.gz`.

## 5. Reproducibility Configuration

```text
Random seed:       20260602
Training rows:     60,000
Evaluation rows:   25,000
Training epochs:   160
Batch size:        2,048
Learning rate:     0.012
```

These values can be overridden using:

```bash
python code/run_r3_experiments.py --help
```

The experiment is simulator-based and uses the documented DataCo preprocessing,
digital-twin assumptions, and operational proxy definitions.

## 6. Inspect Individual Results

```python
import pandas as pd

results = pd.read_csv("results/instance_results.csv.gz")
print(results.head())
```

The files in `results/` provide the machine-readable outputs needed to verify the
computational results reported in the manuscript.
