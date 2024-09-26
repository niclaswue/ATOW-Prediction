import pandas as pd
import numpy as np


class AircraftLandingWeightEstimator:
    def __init__(self):
        # BADA reference data (example values, real implementation would need accurate data)
        # Reference: BADA (Base of Aircraft Data) mentioned in Section II.A of the paper
        self.bada_data = {
            "A320": {"V_stall_ref": 51.5, "m_ref": 64000, "MLW": 66000},
            "B767-300ER": {"V_stall_ref": 62.6, "m_ref": 145000, "MLW": 165000},
            "B777-200ER": {"V_stall_ref": 67.6, "m_ref": 208700, "MLW": 213000},
        }
        self.R = 287  # Gas constant for dry air (J/kg·K)
        self.rho_0 = 1.225  # Air density at sea level (kg/m^3)
        self.C_V_min = 1.3  # Constant from BADA, mentioned in Equation (1)

    def estimate_landing_weights(
        self, df, clip_mlw=False, clip_95_mlw=False, adjust_v_ddes=False
    ):
        """
        Estimate landing weights for aircraft based on mode S data.
        Reference: Method described in Section II.A of the paper

        Args:
        df (pd.DataFrame): DataFrame with columns: aircraft_type, true_airspeed, air_pressure, temperature, altitude
        clip_mlw (bool): If True, clip estimates at MLW (Section III, Table 3)
        clip_95_mlw (bool): If True, clip estimates at 95% of MLW (Section III, Table 4)
        adjust_v_ddes (bool): If True, double V_dDES to 10 kt (Section III, Table 5)

        Returns:
        pd.DataFrame: Input DataFrame with additional column 'estimated_landing_weight'
        """
        df = df.copy()

        # Calculate air density (Equation of state for perfect gases, Section II.A)
        df["air_density"] = df["air_pressure"] / (self.R * df["temperature"])

        # Convert true airspeed to calibrated airspeed (Equation 4)
        df["calibrated_airspeed"] = df["true_airspeed"] * np.sqrt(
            df["air_density"] / self.rho_0
        )

        # Calculate V_dDES based on altitude
        df["V_dDES"] = self._calculate_v_ddes(df["altitude"])
        if adjust_v_ddes:
            df["V_dDES"] *= 2  # Double V_dDES as per Section III

        # Estimate landing weight (Equation 3)
        df["estimated_landing_weight"] = df.apply(self._estimate_weight, axis=1)

        # Apply clipping if requested
        if clip_mlw:
            df["estimated_landing_weight"] = df.apply(
                lambda row: min(
                    row["estimated_landing_weight"],
                    self.bada_data[row["aircraft_type"]]["MLW"],
                ),
                axis=1,
            )
        elif clip_95_mlw:
            df["estimated_landing_weight"] = df.apply(
                lambda row: min(
                    row["estimated_landing_weight"],
                    0.95 * self.bada_data[row["aircraft_type"]]["MLW"],
                ),
                axis=1,
            )

        return df

    def _calculate_v_ddes(self, altitude):
        """Calculate V_dDES based on altitude (Section II.A)."""
        conditions = [
            (altitude < 304.8),  # 1000 ft
            (altitude >= 304.8) & (altitude < 457.2),  # 1000-1500 ft
            (altitude >= 457.2) & (altitude < 609.6),  # 1500-2000 ft
            (altitude >= 609.6) & (altitude < 914.4),  # 2000-3000 ft
        ]
        choices = [2.572, 5.144, 10.288, 25.72]  # 5, 10, 20, 50 kt in m/s
        return np.select(conditions, choices, default=25.72)

    def _estimate_weight(self, row):
        """Estimate landing weight for a single aircraft (Equation 3)."""
        bada = self.bada_data[row["aircraft_type"]]
        V_cas = row["calibrated_airspeed"]
        V_dDES = row["V_dDES"]

        m = ((V_cas - V_dDES) / (self.C_V_min * bada["V_stall_ref"])) ** 2 * bada[
            "m_ref"
        ]
        return m


# Usage example with a 3-row DataFrame:
estimator = AircraftLandingWeightEstimator()

# Create a sample DataFrame
data = {
    "aircraft_type": ["A320", "B767-300ER", "B777-200ER"],
    "true_airspeed": [70, 75, 80],  # m/s
    "air_pressure": [101325, 100000, 98000],  # Pa
    "temperature": [288, 285, 283],  # K
    "altitude": [100, 350, 600],  # m
}
df = pd.DataFrame(data)

# Estimate landing weights
result_df = estimator.estimate_landing_weights(df, clip_mlw=True, adjust_v_ddes=True)

print(result_df)
