#!/usr/bin/env python3
"""
Advanced Photovoltaic Potential Analysis Framework
===============================================

A high-performance implementation for computing photovoltaic potential 
using climate data with parallel processing capabilities.

Required packages:
    - numpy>=1.20.0
    - netCDF4>=1.5.7 
    - xarray>=0.19.0
    - dask>=2021.8.1
    - pandas>=1.3.0
"""

import numpy as np
import xarray as xr
import netCDF4 as nc
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Union
import dask.array as da
import logging
import warnings
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class PVSystemParameters:
    """System parameters for PV calculations."""
    ref_temperature: float = 25.0  # °C
    thermal_coefficient: float = -0.005  # %/°C
    
    # Cell temperature coefficients for monocrystalline silicon
    c1: float = 4.3  # °C
    c2: float = 0.943  # dimensionless
    c3: float = 0.028  # °C m^-2 W^-1
    c4: float = 1.528  # °C m^-1 s^-1
 
    
class PVComputeEngine:
    """High-performance computation engine for PV metrics."""
    
    def __init__(self, params: PVSystemParameters):
        """Initialize computation engine with system parameters."""
        self.params = params
        self._validate_parameters()
        
    def _validate_parameters(self) -> None:
        """Validate input parameters against physical constraints."""
        if not -0.01 <= self.params.thermal_coefficient <= 0:
            warnings.warn("Thermal coefficient outside typical range")
    
    @staticmethod
    def _prepare_computation_chunks(data: np.ndarray, chunk_size: int = 1000) -> list:
        """Prepare data chunks for parallel processing."""
        return np.array_split(data, max(1, len(data) // chunk_size))
    
    def compute_cell_temperature(self, 
                               ambient_temp: Union[np.ndarray, xr.DataArray],
                               solar_radiation: Union[np.ndarray, xr.DataArray],
                               wind_speed: Union[np.ndarray, xr.DataArray],
                               use_parallel: bool = True) -> np.ndarray:
        """
        Compute PV cell temperature using advanced thermal model with wind speed.
        
        Formula: Tcell = c1 + c2 * T + c3 * I - c4 * WS
        
        Args:
            ambient_temp: Ambient temperature data (°C)
            solar_radiation: Solar radiation data (W/m²)
            wind_speed: Wind speed data (m/s)
            use_parallel: Enable parallel processing
            
        Returns:
            np.ndarray: Computed cell temperatures
        """
        if isinstance(ambient_temp, xr.DataArray):
            ambient_temp = ambient_temp.values
        if isinstance(solar_radiation, xr.DataArray):
            solar_radiation = solar_radiation.values
        if isinstance(wind_speed, xr.DataArray):
            wind_speed = wind_speed.values
            
        if use_parallel and ambient_temp.size > 1000:
            return self._parallel_compute_temperature(ambient_temp, solar_radiation, wind_speed)
        
        return self._compute_temperature_core(ambient_temp, solar_radiation, wind_speed)
    
    def _compute_temperature_core(self, 
                                ambient_temp: np.ndarray,
                                solar_radiation: np.ndarray,
                                wind_speed: np.ndarray) -> np.ndarray:
        """
        Core temperature computation implementation using proper equation.
        
        Tcell = c1 + c2 * T + c3 * I - c4 * WS
        
        Where:
        - c1 = 4.3°C
        - c2 = 0.943 (dimensionless)
        - c3 = 0.028°C m^-2 W^-1
        - c4 = 1.528°C m^-1 s^-1
        """
        cell_temp = (self.params.c1 + 
                    self.params.c2 * ambient_temp + 
                    self.params.c3 * solar_radiation - 
                    self.params.c4 * wind_speed)
        
        return cell_temp
    
    def _parallel_compute_temperature(self,
                                    ambient_temp: np.ndarray,
                                    solar_radiation: np.ndarray,
                                    wind_speed: np.ndarray) -> np.ndarray:
        """Parallel implementation of temperature computation."""
        chunks_temp = self._prepare_computation_chunks(ambient_temp)
        chunks_rad = self._prepare_computation_chunks(solar_radiation)
        chunks_wind = self._prepare_computation_chunks(wind_speed)
        
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(lambda x: self._compute_temperature_core(*x),
                                     zip(chunks_temp, chunks_rad, chunks_wind)))
        
        return np.concatenate(results)
    
    def compute_performance_ratio(self, cell_temp: np.ndarray) -> np.ndarray:
        """Compute temperature-dependent performance ratio."""
        return 1 + self.params.thermal_coefficient * (cell_temp - self.params.ref_temperature)
    
    def compute_pv_potential(self,
                           ambient_temp: np.ndarray,
                           solar_radiation: np.ndarray,
                           wind_speed: np.ndarray,
                           use_parallel: bool = True) -> Dict[str, np.ndarray]:
        """
        Compute comprehensive PV metrics.
        
        Args:
            ambient_temp: Ambient temperature data (°C)
            solar_radiation: Solar radiation data (W/m²)
            wind_speed: Wind speed data (m/s)
            use_parallel: Enable parallel processing
        
        Returns:
            Dict containing computed metrics:
                - cell_temperature (°C)
                - performance_ratio (dimensionless)
                - pv_potential (W/m²)
                - efficiency_factor (%)
        """
        cell_temp = self.compute_cell_temperature(ambient_temp, solar_radiation, wind_speed, use_parallel)
        perf_ratio = self.compute_performance_ratio(cell_temp)
        pv_potential = np.multiply(perf_ratio, solar_radiation)
        
        return {
            'cell_temperature': cell_temp,
            'performance_ratio': perf_ratio,
            'pv_potential': pv_potential,
            'efficiency_factor': perf_ratio * 100
        }

class ClimateDataHandler:
    """Advanced handler for climate data processing and I/O operations."""
    
    def __init__(self, base_path: Optional[Path] = None):
        """Initialize data handler with optional base path."""
        self.base_path = Path(base_path) if base_path else Path.cwd()
        
    def load_climate_data(self,
                         temperature_file: str,
                         radiation_file: str,
                         wind_speed_file: str) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
        """
        Load and preprocess climate data files.
        
        Args:
            temperature_file: Path to temperature NetCDF file
            radiation_file: Path to radiation NetCDF file
            wind_speed_file: Path to wind speed NetCDF file
            
        Returns:
            Tuple of (temperature, radiation, wind_speed) as xarray DataArrays
        """
        with xr.open_dataset(self.base_path / temperature_file) as temp_ds, \
             xr.open_dataset(self.base_path / radiation_file) as rad_ds, \
             xr.open_dataset(self.base_path / wind_speed_file) as wind_ds:
            
            # Ensure temporal alignment
            temp_data = temp_ds['tas'].load()
            rad_data = rad_ds['rsds'].load()
            wind_data = wind_ds['sfcWind'].load()  # or 'wspd' depending on your data
            
            # Validate data
            self._validate_climate_data(temp_data, rad_data, wind_data)
            
            return temp_data, rad_data, wind_data
    
    @staticmethod
    def _validate_climate_data(temp: xr.DataArray, rad: xr.DataArray, wind: xr.DataArray) -> None:
        """Validate climate data for consistency and physical bounds."""
        if not (temp.dims == rad.dims == wind.dims):
            raise ValueError("Inconsistent dimensions between climate data variables")
        
        # Physical bounds checking
        if (temp < -90).any() or (temp > 60).any():
            warnings.warn("Temperature values outside physical bounds detected")
        if (rad < 0).any() or (rad > 1500).any():
            warnings.warn("Radiation values outside physical bounds detected")
        if (wind < 0).any() or (wind > 50).any():
            warnings.warn("Wind speed values outside physical bounds detected")
    
    def save_results(self,
                    results: Dict[str, np.ndarray],
                    output_file: str,
                    reference_data: xr.DataArray,
                    metadata: Optional[Dict] = None) -> None:
        """
        Save results to NetCDF file with CF conventions.
        
        Args:
            results: Dictionary of computed metrics
            output_file: Output filename
            reference_data: Reference DataArray for dimensional information
            metadata: Optional metadata dictionary
        """
        ds = xr.Dataset(
            data_vars={
                name: (reference_data.dims, data, self._get_variable_attributes(name))
                for name, data in results.items()
            },
            coords=reference_data.coords
        )
        
        # Add metadata
        ds.attrs.update({
            'title': 'PV Potential Analysis Results',
            'creation_date': datetime.now().isoformat(),
        })
        
        # Save with compression
        ds.to_netcdf(
            self.base_path / output_file,
            encoding={var: {'zlib': True, 'complevel': 5} for var in ds.data_vars}
        )
        
    @staticmethod
    def _get_variable_attributes(var_name: str) -> Dict[str, str]:
        """Get CF-compliant variable attributes."""
        attributes = {
            'cell_temperature': {
                'units': 'degree_Celsius',
                'long_name': 'PV Cell Temperature',
                'standard_name': 'temperature'
            },
            'performance_ratio': {
                'units': '1',
                'long_name': 'Temperature-Dependent Performance Ratio',
                'standard_name': 'performance_ratio'
            },
            'pv_potential': {
                'units': 'W m-2',
                'long_name': 'Photovoltaic Power Potential',
                'standard_name': 'surface_photovoltaic_potential'
            },
            'efficiency_factor': {
                'units': 'percent',
                'long_name': 'PV System Efficiency Factor',
                'standard_name': 'efficiency_percentage'
            }
        }
        return attributes.get(var_name, {})

def main():
    """Main execution function."""
    try:
        # Initialize components
        params = PVSystemParameters()
        engine = PVComputeEngine(params)
        handler = ClimateDataHandler()
        
        # Load data (you'll need to provide the wind speed file)
        temp_data, rad_data, wind_data = handler.load_climate_data(
            'tas_mon_one_ssp126_192_ave_converted.nc',
            'rsds_mon_one_ssp126_192_ave.nc',
            'sfcWind_mon_one_ssp126_192_ave.nc'  # You'll need this file
        )
        
        # Compute metrics
        results = engine.compute_pv_potential(
            temp_data.values,
            rad_data.values,
            wind_data.values,
            use_parallel=True
        )
        
        # Save results
        handler.save_results(
            results,
            'PVP_126_advanced.nc',
            temp_data,
            metadata={'scenario': 'SSP1-2.6'}
        )
        
        logger.info("Analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
