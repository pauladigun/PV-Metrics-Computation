#!/usr/bin/env python3
"""
cv_analysis.py
=============

Calculate climate variability changes excluding extreme events.


"""

import xarray as xr
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class Params:
    """Analysis parameters."""
    low_p: float = 0.05
    high_p: float = 0.95
    hist_period: Tuple[str, str] = ('1980', '2014')
    fut_period: Tuple[str, str] = ('2051', '2100')
    vars: Tuple[str, ...] = ('rsds', 'tas', 'clt')

class CVAnalyzer:
    """Analyze coefficient of variation changes."""
    
    def __init__(self, data_dir: str = '.', params: Optional[Params] = None):
        self.data_dir = Path(data_dir)
        self.params = params or Params()
    
    def load_data(self, file_path: str) -> xr.Dataset:
        """Load and process climate data."""
        ds = xr.open_dataset(self.data_dir / file_path)
        land_mask = ds.PVP.notnull().any(dim='time')
        ds_land = ds.where(land_mask)
        return ds_land.dropna(dim='lat', how='all').dropna(dim='lon', how='all')
    
    @staticmethod
    def calc_cv(data: xr.DataArray) -> xr.DataArray:
        """Calculate coefficient of variation."""
        return data.std(dim='time') / data.mean(dim='time')
    
    def remove_extremes(self, data: xr.Dataset, var: str) -> xr.Dataset:
        """Remove extreme days based on percentiles."""
        low_t = data[var].quantile(self.params.low_p, dim='time')
        high_t = data[var].quantile(self.params.high_p, dim='time')
        mask = (data[var] >= low_t) & (data[var] <= high_t)
        return data.where(mask)
    
    def calc_cv_change(self, data: xr.Dataset, var: str) -> xr.DataArray:
        """Calculate CV change after removing extremes."""
        orig_cv = self.calc_cv(data.PVP)
        filtered_cv = self.calc_cv(self.remove_extremes(data, var).PVP)
        return (filtered_cv - orig_cv) / orig_cv * 100
    
    def analyze_scenarios(self, scenarios: list, mask: Optional[xr.DataArray] = None) -> Dict:
        """Analyze CV changes across scenarios."""
        results = {}
        
        for scenario in scenarios:
            future = self.load_data(f'PVP_{scenario}_2015_to_2100.nc')
            future = future.sel(time=slice(*self.params.fut_period))
            
            scenario_results = {}
            for var in self.params.vars:
                cv_change = self.calc_cv_change(future, var)
                if mask is not None:
                    cv_change = cv_change.where(mask)
                
                # Save results
                cv_change.to_netcdf(f'cv_{var}_ssp{scenario}.nc')
                scenario_results[var] = cv_change
                
                # Log stats
                mean = cv_change.mean().item()
                std = cv_change.std().item()
                print(f"SSP{scenario} - {var}: {mean:.2f}% ± {std:.2f}%")
            
            results[scenario] = scenario_results
        
        return results

def calc_global_stats(results: Dict) -> Dict:
    """Calculate area-weighted global statistics."""
    stats = {}
    
    for scenario, var_results in results.items():
        scenario_stats = {}
        
        for var, cv_change in var_results.items():
            weights = np.cos(np.deg2rad(cv_change.lat))
            weighted = cv_change.weighted(weights)
            
            scenario_stats[var] = {
                'mean': weighted.mean(['lat', 'lon']).item(),
                'std': weighted.std(['lat', 'lon']).item(),
                'range': [cv_change.min().item(), cv_change.max().item()]
            }
        
        stats[scenario] = scenario_stats
    
    return stats

def main():
    """Execute analysis."""
    try:
        analyzer = CVAnalyzer()
        scenarios = ['126', '245', '370', '585']
        
        # Run analysis
        results = analyzer.analyze_scenarios(scenarios)
        stats = calc_global_stats(results)
        
        # Print summary
        print("\nGlobal Statistics:")
        for scenario, var_stats in stats.items():
            print(f"\nSSP{scenario}:")
            for var, metrics in var_stats.items():
                print(f"{var}: {metrics['mean']:.1f}% ± {metrics['std']:.1f}%")
                print(f"Range: [{metrics['range'][0]:.1f}%, {metrics['range'][1]:.1f}%]")
        
        return results, stats
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
