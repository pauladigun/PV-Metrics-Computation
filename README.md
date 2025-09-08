# Climate Change Impacts on Solar Photovoltaic Potential Analysis

[![DOI](https://img.shields.io/badge/DOI-10.xxxx%2Fxxxxxx-blue)](https://doi.org/10.xxxx/xxxxxx)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Overview

This repository contains the computational framework for analyzing climate change impacts on global solar photovoltaic (PV) energy potential using CMIP6 multi-model ensemble projections. The code implements advanced methodologies for decomposing atmospheric drivers, extreme event attribution, and variability analysis of solar energy resources under different emission scenarios.

**Associated Publication**: "Climate Change Influence on Solar Photovoltaic Energy Production and Its Associated Drivers in CMIP6 Ensemble Projections" by Adigun et al.

## Key Features

- **Multi-model ensemble analysis** using 32 CMIP6 climate models
- **Atmospheric forcing decomposition** separating aerosol-radiation interactions (ARI) from cloud radiative effects (CRE)
- **Extreme event attribution framework** for solar energy potential
- **Coefficient of variation analysis** excluding extreme weather conditions
- **High-performance computing** with parallel processing capabilities
- **CF-compliant NetCDF output** with comprehensive metadata

## Repository Structure

```
├── pvp_analysis_framework.py    # Main PV potential computation engine
├── extreme_metrics_analysis.py  # Extreme events probability analysis
├── cv_analysis.py              # Coefficient of variation analysis
├── data/                       # Data directory (create locally)
│   ├── historical/            # Historical climate data
│   ├── scenarios/             # Future scenario data (SSP126, SSP245, etc.)
│   └── output/                # Analysis results
├── docs/                      # Documentation
├── examples/                  # Usage examples
└── requirements.txt           # Python dependencies
```

## Installation

### Prerequisites

- Python 3.8 or higher
- Git
- Anaconda/Miniconda (recommended)

### Setup Environment

```bash
# Clone the repository
git clone https://github.com/pauladigun/PV-Metrics-Computation.git
cd PV-Metrics-Computation

# Create conda environment
conda create -n pvp_analysis python=3.9
conda activate pvp_analysis

# Install dependencies
pip install -r requirements.txt
```

### Required Python Packages

```
numpy>=1.20.0
xarray>=0.19.0
netCDF4>=1.5.7
dask>=2021.8.1
pandas>=1.3.0
scipy>=1.7.0
matplotlib>=3.4.0
cartopy>=0.19.0
```

## Data Requirements

### Input Data Structure

The framework expects climate data in NetCDF format with the following variables:

**Required Variables:**
- `tas`: Near-surface air temperature (°C)
- `rsds`: Surface downwelling shortwave radiation (W/m²)
- `rsdscs`: Surface downwelling shortwave radiation under clear-sky conditions (W/m²)
- `clt`: Total cloud cover (%)

**File Naming Convention:**
```
{variable}_{frequency}_{model}_{scenario}_{period}.nc

Examples:
- tas_mon_CMIP6-MME_historical_1980-2014.nc
- rsds_mon_CMIP6-MME_ssp126_2015-2100.nc
```

### Data Sources

- **CMIP6 Data**: Available from [Earth System Grid Federation (ESGF)](https://esgf-node.llnl.gov/projects/cmip6/)
- **WFDE5 Data**: Available from [Copernicus Climate Data Store](https://climate.copernicus.eu/climate-reanalysis)

## Usage

### Basic PV Potential Calculation

```python
from pvp_analysis_framework import PVComputeEngine, PVSystemParameters, ClimateDataHandler

# Initialize components
params = PVSystemParameters(
    ref_temperature=25.0,
    thermal_coefficient=-0.005
)
engine = PVComputeEngine(params)
handler = ClimateDataHandler()

# Load climate data
temp_data, rad_data = handler.load_climate_data(
    'tas_mon_CMIP6-MME_ssp126_2015-2100.nc',
    'rsds_mon_CMIP6-MME_ssp126_2015-2100.nc'
)

# Compute PV metrics
results = engine.compute_pv_potential(
    temp_data.values,
    rad_data.values,
    use_parallel=True
)

# Save results
handler.save_results(
    results,
    'PVP_ssp126_analysis.nc',
    temp_data,
    metadata={'scenario': 'SSP1-2.6', 'period': '2015-2100'}
)
```

### Extreme Events Analysis

```python
from extreme_metrics_analysis import ExtremeMetricsAnalyzer

# Initialize analyzer
analyzer = ExtremeMetricsAnalyzer(data_dir='./data')

# Analyze multiple scenarios
scenarios = ['126', '245', '370', '585']
results = analyzer.analyze_scenarios(
    scenarios=scenarios,
    historical_period=('1980', '2014'),
    future_period=('2065', '2100')
)

# Results contain probability of extreme high/low PV potential events
```

### Coefficient of Variation Analysis

```python
from cv_analysis import CVAnalyzer

# Initialize analyzer
analyzer = CVAnalyzer(data_dir='./data')

# Analyze variability changes after removing extremes
results = analyzer.analyze_scenarios(['126', '245', '370', '585'])

# Results show CV changes for each climate variable
```

## Methodology

### PV Potential Calculation

The framework implements the energy rating method for PV potential calculation:

```
PVP(t) = Pr(t) × Rs(t) / R_STC

Where:
- Pr(t): Performance ratio accounting for temperature effects
- Rs(t): Solar irradiance
- R_STC: Standard test condition irradiance (1000 W/m²)
```

**Performance Ratio:**
```
Pr(t) = 1 + γ × (T_cell - T_STC)

Where:
- γ: Temperature coefficient (-0.005 °C⁻¹ for monocrystalline silicon)
- T_cell: Cell temperature
- T_STC: Standard test temperature (25°C)
```

**Cell Temperature Model:**
```
T_cell = c₁ + c₂×T + c₃×I - c₄×WS

Coefficients for monocrystalline silicon:
- c₁ = 4.3°C
- c₂ = 0.943
- c₃ = 0.028°C·m²·W⁻¹  
- c₄ = 1.528°C·m·s⁻¹
```

### Atmospheric Forcing Decomposition

The framework separates total PV potential changes into:

1. **Aerosol-Radiation Interactions (ARI)**: Direct effects using clear-sky radiation
2. **Cloud Radiative Effects (CRE)**: Residual between total and ARI effects

```
CRE = ΔPVP_total - ΔPVP_ARI
```

### Extreme Event Attribution

Implements climate attribution methodology using:

- **Risk Ratio (RR)**: `RR = P_future / P_historical`
- **Fraction of Attributable Risk (FAR)**: `FAR = (P_future - P_historical) / P_future`

## Output Files

### NetCDF Variables

| Variable | Units | Description |
|----------|-------|-------------|
| `pv_potential` | W/m² | Photovoltaic power potential |
| `cell_temperature` | °C | PV cell temperature |
| `performance_ratio` | - | Temperature-dependent performance ratio |
| `efficiency_factor` | % | PV system efficiency factor |

### Analysis Results

- **Extreme probabilities**: Probability of extreme high/low PV potential events
- **CV changes**: Coefficient of variation changes after removing extremes
- **Attribution metrics**: Risk ratios and attributable risk fractions

## Configuration

### System Parameters

Modify `PVSystemParameters` for different PV technologies:

```python
# Monocrystalline Silicon (default)
params = PVSystemParameters(
    ref_temperature=25.0,
    thermal_coefficient=-0.005
)

# Thin-film technologies
params = PVSystemParameters(
    ref_temperature=25.0,
    thermal_coefficient=-0.002
)
```

### Analysis Parameters

Configure analysis settings in respective modules:

```python
# Extreme analysis thresholds
high_quantile = 0.9  # 90th percentile for high extremes
low_quantile = 0.1   # 10th percentile for low extremes

# CV analysis percentiles
low_p = 0.05   # 5th percentile
high_p = 0.95  # 95th percentile
```

## Performance Optimization

### Parallel Processing

Enable parallel computation for large datasets:

```python
# Automatic parallel processing for datasets > 1000 points
results = engine.compute_pv_potential(
    temp_data, rad_data, use_parallel=True
)

# Manual chunk size control
chunks = engine._prepare_computation_chunks(data, chunk_size=500)
```


## Validation

The framework includes validation against WFDE5 observational data:

- **RSDS bias**: ±15 W/m² (≤14%)
- **Temperature bias**: ±5°C (≤14%)
- **PVP bias**: ±25 W/m² (≤11%)

## Citation

If you use this code in your research, please cite:

```bibtex
@article{adigun2025climate,
  title={Climate Change Influence on Solar Photovoltaic Energy Production and Its Associated Drivers in CMIP6 Ensemble Projections},
  author={Adigun, Paul and Koji, Dairaku and Ogunrinde, Akinwale T. and Xue, Xian},
  journal={[Journal Name]},
  year={2025},
  doi={10.xxxx/xxxxxx}
}
```


For questions and support:

- **Issues**: [GitHub Issues](https://github.com/pauladigun/PV-Metrics-Computation/issues)
- **Email**: adigunmet133492@futa.edu.ng
- **Documentation**: [Wiki](https://github.com/pauladigun/PV-Metrics-Computation/wiki)



**Keywords**: Climate Change, Solar Energy, Photovoltaic Potential, CMIP6, Extreme Events, Atmospheric Forcing, Python, NetCDF
