#!/usr/bin/env python3
"""
extreme_metrics_analysis.py
==========================

Calculate extreme metrics for climate analysis with a focus on
photovoltaic potential under different scenarios.


import xarray as xr
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ExtremeMetricsAnalyzer:
    """Analysis of extreme conditions in climate datasets."""
    
    def __init__(self, data_dir: str = '.'):
        """
        Initialize the analyzer.
        
        Args:
            data_dir: Directory containing the NetCDF files
        """
        self.data_dir = Path(data_dir)
    
    def load_dataset(self, file_path: str) -> xr.Dataset:
        """
        Load and preprocess a NetCDF dataset.
        
        Args:
            file_path: Path to the NetCDF file
            
        Returns:
            xr.Dataset: Processed dataset with land-only values
        """
        logger.info(f"Loading dataset: {file_path}")
        ds = xr.open_dataset(self.data_dir / file_path)
        
        # Create land mask based on non-null values
        land_mask = ds.PVP.notnull().any(dim='time')
        
        # Apply mask and drop empty coordinates
        ds_land = ds.where(land_mask)
        ds_land = ds_land.dropna(dim='lat', how='all')
        ds_land = ds_land.dropna(dim='lon', how='all')
        
        return ds_land
    
    def calculate_extremes(self, 
                         historical: xr.DataArray,
                         future: xr.DataArray,
                         high_quantile: float = 0.9,
                         low_quantile: float = 0.1) -> Tuple[xr.DataArray, xr.DataArray]:
        """
        Calculate probability of extreme conditions.
        
        Args:
            historical: Historical data array
            future: Future scenario data array
            high_quantile: Threshold for high extremes (default: 0.9)
            low_quantile: Threshold for low extremes (default: 0.1)
            
        Returns:
            Tuple containing (probability of high extremes, probability of low extremes)
        """
        # Calculate historical thresholds
        logger.info(f"Calculating thresholds: {low_quantile} and {high_quantile}")
        historical_high = historical.quantile(high_quantile, dim='time')
        historical_low = historical.quantile(low_quantile, dim='time')
        
        # Calculate exceedance probabilities
        extreme_high = (future > historical_high).mean(dim='time') * 100
        extreme_low = (future < historical_low).mean(dim='time') * 100
        
        return extreme_high, extreme_low
    
    def analyze_scenarios(self,
                        scenarios: list,
                        historical_period: Tuple[str, str] = ('1980', '2014'),
                        future_period: Tuple[str, str] = ('2051', '2100')) -> Dict[str, Tuple[xr.DataArray, xr.DataArray]]:
        """
        Analyze multiple scenarios and calculate their extreme probabilities.
        
        Args:
            scenarios: List of scenario identifiers (e.g., ['126', '245', '370', '585'])
            historical_period: Start and end years for historical period
            future_period: Start and end years for future period
            
        Returns:
            Dictionary containing results for each scenario
        """
        # Load and process historical data
        historical_ds = self.load_dataset('historical.nc')
        historical = historical_ds.sel(time=slice(*historical_period)).PVP
        
        results = {}
        for scenario in scenarios:
            logger.info(f"Processing scenario SSP{scenario}")
            
            # Load and process future scenario data
            future_ds = self.load_dataset(f'PVP_{scenario}_2015_to_2100.nc')
            future = future_ds.sel(time=slice(*future_period)).PVP
            
            # Calculate extreme probabilities
            extreme_high, extreme_low = self.calculate_extremes(historical, future)
            results[scenario] = (extreme_high, extreme_low)
            
            # Optional: Add basic statistics
            self._log_basic_stats(scenario, extreme_high, extreme_low)
        
        return results
    
    def _log_basic_stats(self, scenario: str, high: xr.DataArray, low: xr.DataArray):
        """Log basic statistics for the results."""
        logger.info(f"\nScenario SSP{scenario} Statistics:")
        logger.info(f"High extremes - Mean: {high.mean().item():.2f}%, Max: {high.max().item():.2f}%")
        logger.info(f"Low extremes - Mean: {low.mean().item():.2f}%, Max: {low.max().item():.2f}%")

def compute_global_metrics(results: Dict[str, Tuple[xr.DataArray, xr.DataArray]]) -> Dict[str, Dict[str, float]]:
    """
    Compute global metrics for all scenarios.
    
    Args:
        results: Dictionary of scenario results
        
    Returns:
        Dictionary of global metrics for each scenario
    """
    global_metrics = {}
    
    for scenario, (high, low) in results.items():
        # Calculate area-weighted means
        weights = np.cos(np.deg2rad(high.lat))
        
        # High extremes metrics
        high_weighted = high.weighted(weights)
        high_mean = high_weighted.mean(['lat', 'lon']).item()
        high_std = high_weighted.std(['lat', 'lon']).item()
        
        # Low extremes metrics
        low_weighted = low.weighted(weights)
        low_mean = low_weighted.mean(['lat', 'lon']).item()
        low_std = low_weighted.std(['lat', 'lon']).item()
        
        global_metrics[scenario] = {
            'high_mean': high_mean,
            'high_std': high_std,
            'low_mean': low_mean,
            'low_std': low_std
        }
    
    return global_metrics

def main():
    """Main execution function."""
    try:
        # Initialize analyzer
        analyzer = ExtremeMetricsAnalyzer()
        
        # Define scenarios and periods
        scenarios = ['126', '245', '370', '585']
        historical_period = ('1980', '2014')
        future_period = ('2051', '2100')
        
        # Analyze all scenarios
        results = analyzer.analyze_scenarios(
            scenarios=scenarios,
            historical_period=historical_period,
            future_period=future_period
        )
        
        # Compute global metrics
        global_metrics = compute_global_metrics(results)
        
        # Log global metrics
        logger.info("\nGlobal Metrics Summary:")
        for scenario, metrics in global_metrics.items():
            logger.info(f"\nSSP{scenario}:")
            logger.info(f"High Extremes - Mean: {metrics['high_mean']:.2f}% ± {metrics['high_std']:.2f}%")
            logger.info(f"Low Extremes - Mean: {metrics['low_mean']:.2f}% ± {metrics['low_std']:.2f}%")
        
        logger.info("Analysis completed successfully")
        return results, global_metrics
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
