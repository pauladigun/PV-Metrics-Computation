Climate Analysis Tools
A collection of Python tools for analyzing climate data and photovoltaic potential under different scenarios.
Overview
This repository contains three main analysis scripts:

extreme_metrics_analysis.py: Analyzes probability of extreme events
cv_analysis.py: Analyzes coefficient of variation changes excluding extremes
pv_analysis.py: Computes photovoltaic potential metrics

Installation
Requirements
Copynumpy>=1.20.0
xarray>=0.19.0
netCDF4>=1.5.7
scipy>=1.7.0
pandas>=1.3.0
dask>=2021.8.1
Install requirements:
bashCopypip install -r requirements.txt
Data Requirements
Each script expects NetCDF files in the following format:

Historical data: historical.nc
Future scenarios: PVP_{scenario}_2015_to_2100.nc

Where scenario is one of: '126', '245', '370', '585'



Required variables in NetCDF files:

PVP (Photovoltaic Potential)
tas (Temperature)
rsds (Solar Radiation)
clt (Cloud Cover)

Script Descriptions
1. extreme_metrics_analysis.py
Calculates probability of extreme events in future scenarios compared to historical period.
pythonCopyfrom extreme_metrics_analysis import ExtremeMetricsAnalyzer

analyzer = ExtremeMetricsAnalyzer()
results = analyzer.analyze_scenarios(['126', '245', '370', '585'])
Key Features:

Computes probability of exceeding historical thresholds
Handles multiple scenarios
Provides global statistics

2. cv_analysis.py
Analyzes changes in variability by computing coefficient of variation with/without extreme events.
pythonCopyfrom cv_analysis import CVAnalyzer

analyzer = CVAnalyzer()
results, stats = analyzer.main()
Key Features:

Removes extreme events based on percentiles
Calculates CV changes
Supports multiple variables (rsds, tas, clt)

3. pv_analysis.py
Computes photovoltaic potential and related metrics.
pythonCopyfrom pv_analysis import PVComputeEngine, PVSystemParameters

params = PVSystemParameters()
engine = PVComputeEngine(params)
results = engine.compute_pv_potential(ambient_temp, solar_radiation)
Key Features:

High-performance computation
Parallel processing capabilities
Comprehensive error handling

Output Files
Each script generates several output files:
extreme_metrics_analysis.py

extreme_prob_high_ssp{scenario}.nc: Probability of high extremes
extreme_prob_low_ssp{scenario}.nc: Probability of low extremes

cv_analysis.py

cv_{variable}_ssp{scenario}.nc: CV changes for each variable and scenario

pv_analysis.py

PVP_{scenario}.nc: Computed PV potential metrics

Example Usage
Complete analysis workflow:
pythonCopy# 1. First compute PV potential
from pv_analysis import PVComputeEngine
engine = PVComputeEngine()
pv_results = engine.compute_pv_potential(temp_data, radiation_data)

# 2. Analyze extremes
from extreme_metrics_analysis import ExtremeMetricsAnalyzer
extreme_analyzer = ExtremeMetricsAnalyzer()
extreme_results = extreme_analyzer.analyze_scenarios(['126'])

# 3. Analyze variability
from cv_analysis import CVAnalyzer
cv_analyzer = CVAnalyzer()
cv_results = cv_analyzer.analyze_scenarios(['126'])
