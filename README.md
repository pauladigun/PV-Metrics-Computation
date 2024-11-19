Code for analyzing Climate change influence on Solar Photovoltaic Energy Production and Its Associated Drivers in CMIP6 Ensemble Projections.
Scripts
1. cv_analysis.py
Main script for coefficient of variation analysis.
pythonCopyfrom cv_analysis import CVAnalyzer

analyzer = CVAnalyzer()
results = analyzer.analyze_scenarios(['126', '245', '370', '585'])
Calculates:

CV changes excluding extreme events
Multi-variable analysis (RSDS, TAS, CLT)
Area-weighted global statistics

2. extreme_metrics.py
Extreme event probability analysis.
pythonCopyfrom extreme_metrics import ExtremeMetricsAnalyzer

analyzer = ExtremeMetricsAnalyzer()
results = analyzer.analyze_scenarios(['126', '245', '370', '585'])

3. pv_analysis.py
Photovoltaic potential calculations.
pythonCopyfrom pv_analysis import PVComputeEngine

engine = PVComputeEngine()
results = engine.compute_pv_potential(temp_data, rad_data)
Requirements
Copyxarray>=0.19.0
numpy>=1.20.0
netCDF4>=1.5.7
scipy>=1.7.0
Data Format
Expects NetCDF files:

historical.nc: Historical data
PVP_{scenario}_2015_to_2100.nc: Future scenarios

Usage
pythonCopy# Example workflow
analyzer = CVAnalyzer()
results = analyzer.analyze_scenarios(['126', '245'])
stats = analyzer.calc_global_stats(results)
