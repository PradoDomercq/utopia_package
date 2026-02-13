'''
Author: Xiaoyu Zhang (xiaoyu.zhang@aces.su.se)
Date: 2026-02-11 15:26:29
LastEditTime: 2026-02-13 13:07:51
Description: this script is for testing the atmospheric settling velocity,
             and comparing the settling velocity & Reynolds number with measured data.
             (test data extracted from the paper: https://pubs.acs.org/doi/10.1021/acsestair.5c00250)
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import warnings
warnings.filterwarnings('ignore')


# ==================== 1. Original Function from dry_deposition_MS.py ====================

# ---- Constants dependants of Temperature, Pressure, altitud. Have to be adapted to specific regions ------------
muAir = 1.82e-5 #1.789e-5  # [kg.m−1.s−1], dynamic viscosity of the air (Heigth, Temp dependent)
rhoAir = 1.205 #1.1438  # [kg.m-3], density of air (Heigth, Temp dependent)
#!!! 260211 XZ testing: revised muAir and rhoAir to the values at 20°C and ori. vaules are commented behind

g0 = 9.81  # [m.s-2] gravitational acceleration on earth (Heigth dependent)


# Constant for Cunningham correction, here from Jennings, S, 1988
A1 = 1.257
A2 = 0.400
A3 = 1.100
mfpAir = 6.635e-8  # [m] Mean free path on dry air, from Jennings, S, 1988


# Calculate the Reynolds Number (based on settling velocity of Stokes,
# may be optimised with self consistancy)
def ReynoldsNumberFromStokes(d, rho):
    vg = np.multiply(d, d)
    vg = vg * g0 / (18.0 * muAir)
    vg = np.multiply(vg, (rho - rhoAir))

    Rep = np.multiply(vg, rhoAir)
    Rep = np.multiply(Rep, d)
    Rep = np.multiply(Rep, (1.0 / muAir))

    return Rep


# Placeholder function to calculate Reynolds number
def ReynoldsNumberFromVg(d, rho, vg):
    # Placeholder implementation
    # This function should calculate the Reynolds number based on the settling velocity
    Rep = np.multiply(vg, rhoAir)
    Rep = np.multiply(Rep, d)
    Rep = np.multiply(Rep, (1.0 / muAir))

    return Rep


# Calculate the Drag coefficient number
def dragCoefficient(d, rho, Rep):
    # Rep = ReynoldsNumberFromStokes (d, rho)
    # Set Rep to a specific value for test (e.g., 13.4) !!!!!!!!!!! to remoove after test

    # Rep = 13.35
    Cd = np.zeros_like(Rep)

    Rep_conditions = [
        Rep <= 1,
        (1 < Rep) & (Rep <= 1000.0),
        (1000.0 < Rep) & (Rep <= 2.0e5),
        Rep > 2.0e5,
    ]

    Cd_functions = [
        lambda x: 24.0 / x,
        lambda x: 24.0 / x + 5.0 / np.power(x, 0.6) + 0.44,
        0.44,
        0.10,
    ]

    Cd = np.piecewise(Rep, Rep_conditions, Cd_functions)

    # Cd = 24.0 * (1.0 + (0.15 * Rep**0.687)) / Rep
    # Cd = Cd + 0.42 / (1.0 + (42500.0/ (Rep**1.16)))
    return Cd


# Calculate rate constants of dry settling based on Newton regime
# (for big particles, generating a turbulent flow)
def kineticCstdrySettlingNewtonSphere(d, rho, Rep):
    d = np.array(d)
    rho = np.array(rho)

    # Calculate the drag coefficient Cd
    Cd = dragCoefficient(d, rho, Rep)
    # print("Cd sphere=", Cd)
    # Calculate the settling velocity
    v = np.sqrt(4.0 * d * g0 * (rho - rhoAir) / (3.0 * Cd * rhoAir))
    # v = np.sqrt(4.0 * d * g0 * (rho) / (3.0 * Cd * rhoAir))

    # elt = v / shape  # [m.s-1] / [m] The Boundary layer heigth
    elt = v
    return elt


# Placeholder function to calculate settling velocity
def calculate_settling(reynolds_number):
    # Placeholder implementation
    # This function should calculate a new estimate for the settling velocity based on the Reynolds number
    return new_settling_velocity


# Subroutine to calculate settling velocity iteratively
def get_settling(initial_Settling, d, rho, initial_Rep):

    settling_old = initial_Settling

    # Set convergence threshold
    tolerance = 0.001

    # Maximum number of iterations
    max_iterations = 20

    # Initialize iteration counter
    iteration = 0

    # Iterate until convergence or maximum iterations reached
    while iteration < max_iterations:
        # Calculate Reynolds number using current settling velocity
        reynolds = ReynoldsNumberFromVg(d, rho, settling_old)

        # Use Reynolds number to calculate new estimate for settling velocity
        settling_new = kineticCstdrySettlingNewtonSphere(d, rho, reynolds)

        # Check for convergence
        cvg = abs((settling_new - settling_old) / settling_new)
        # print("convergence <", tolerance, "?=", cvg)
        if np.all(cvg < tolerance):
            break

        # Update settling velocity for next iteration
        settling_old = settling_new

        # Increment iteration counter
        iteration += 1

    # Use final settling velocity for further calculations
    final_settling_velocity = settling_new
    reynolds = ReynoldsNumberFromVg(d, rho, final_settling_velocity)

    return final_settling_velocity




# ==================== 2. Read data and process ====================
def load_and_prepare(csv_path):
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    
    # Fuzzy match key column names (adapt to your latest headers)
    def find_col(patterns):
        for col in df.columns:
            col_lower = str(col).lower()
            if any(p.lower() in col_lower for p in patterns):
                return col
        return None
    
    col_map = {
        'diameter': find_col(['diameter', '(mm)']),
        'density': find_col(['density']),
        'vs_meas': find_col(['settling velocity', 'v (m s-1)']),
        're_meas': find_col(['rep']),
        'shape': find_col(['shape']),
        'length': find_col(['l (mm)']),
        'aspect': find_col(['aspect ratio']),
        'vs_sd': find_col(['settling velocity sd'])
    }
    
    print("Automatically detected columns:")
    for k, v in col_map.items():
        print(f"  {k:10} -> {v}")
    
    # Extract main diameter value (handle parentheses)
    def extract_number(val):
        if pd.isna(val):
            return np.nan
        s = str(val).strip()
        match = re.search(r'([-+]?\d*\.?\d+)', s)
        return float(match.group(1)) if match else np.nan
    
    df['diameter_mm'] = df[col_map['diameter']].apply(extract_number)
    df['diameter_m'] = df['diameter_mm'] / 1000.0
    df['density_kgm3'] = pd.to_numeric(df[col_map['density']], errors='coerce') * 1000.0
    df['vs_measured'] = pd.to_numeric(df[col_map['vs_meas']], errors='coerce')
    df['vs_sd'] = pd.to_numeric(df[col_map['vs_sd']], errors='coerce')
    df['Re_measured'] = pd.to_numeric(df[col_map['re_meas']], errors='coerce')
    
    # delete rows with missing critical data
    initial_len = len(df)
    df.dropna(subset=['diameter_m', 'density_kgm3', 'vs_measured', 'Re_measured'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"Valid data: {len(df)} rows (originally {initial_len} rows)")
    
    return df, col_map

# ==================== 3. Calculate settling velocity and Reynolds number ====================
def compute_settling(df):
    vs_calc_list = []
    Re_calc_list = []
    
    for idx, row in df.iterrows():
        d = np.array([row['diameter_m']])
        rho_p = np.array([row['density_kgm3']])
        
        vs_stokes = (d**2 * g0 * (rho_p - rhoAir)) / (18.0 * muAir)
        Re_stokes = ReynoldsNumberFromStokes(d, rho_p)
        
        vs_final = get_settling(vs_stokes, d, rho_p, Re_stokes)
        vs_final = vs_final[0] if isinstance(vs_final, np.ndarray) else vs_final
        
        Re_final = ReynoldsNumberFromVg(d, rho_p, np.array([vs_final]))[0]
        
        vs_calc_list.append(vs_final)
        Re_calc_list.append(Re_final)
    
    df['vs_calc'] = vs_calc_list
    df['Re_calc'] = Re_calc_list
    
    # Relative error
    df['vs_error_%'] = (df['vs_calc'] - df['vs_measured']) / df['vs_measured'] * 100.0
    df['Re_error_%'] = (df['Re_calc'] - df['Re_measured']) / df['Re_measured'] * 100.0
    
    return df

# ==================== 4. Save results to CSV ====================
def save_results(df, col_map):
    output_cols = [
        'Number', 'Microplastics', 'Material',
        col_map['density'], col_map['shape'], 'diameter_mm',
        col_map['length'], col_map['aspect'],
        col_map['vs_meas'], col_map['vs_sd'],
        'vs_calc', 'vs_error_%',
        col_map['re_meas'], 'Re_calc', 'Re_error_%'
    ]
    output_cols = [c for c in output_cols if c in df.columns]
    df_out = df[output_cols].copy()
    df_out.to_csv('settling_velocity_results.csv', index=False, encoding='utf-8-sig')
    print("Result saved to settling_velocity_results.csv")

# ==================== 5. Plotting (Settling Velocity with Error Bars) ====================
def plot_results(df, col_map):
    shape_colors = {
        'sphere': '#1f78b4', 'disk': '#5a9755', 'fragment': '#e31a1c',
        'fiber': '#ff7f00', 'film': '#6a3d9a'
    }
    df['shape_lower'] = df[col_map['shape']].str.lower()
    shapes = df['shape_lower'].unique()
    
    # ----- Figure 1: Measured vs Calculated (1x2) -----
    plt.figure(figsize=(14, 6))
    
    # Subplot 1: Settling Velocity (with error bars)
    plt.subplot(1, 2, 1)
    for shape in shapes:
        subset = df[df['shape_lower'] == shape].sort_values('diameter_mm')
        # Measured values: hollow circles + error bars
        plt.errorbar(subset['diameter_mm'], subset['vs_measured'],
                     yerr=subset['vs_sd'],
                     fmt='o',
                     markerfacecolor='none',
                     markeredgecolor=shape_colors.get(shape, 'gray'),
                     markeredgewidth=1.5,
                     ecolor=shape_colors.get(shape, 'gray'),
                     elinewidth=1,
                     capsize=3,
                     capthick=1,
                     label=f'{shape.title()} (Measured)')
        # Calculated values: crosses
        plt.scatter(subset['diameter_mm'], subset['vs_calc'],
                    color=shape_colors.get(shape, 'gray'),
                    marker='x',
                    label=f'{shape.title()} (Calc)')
    plt.xlabel('Diameter (mm)')
    plt.ylabel('Settling Velocity (m/s)')
    plt.title('Settling Velocity: Measured vs Calculated')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # Subplot 2: Reynolds Number (no error bars)
    plt.subplot(1, 2, 2)
    for shape in shapes:
        subset = df[df['shape_lower'] == shape]
        plt.scatter(subset['diameter_mm'], subset['Re_measured'],
                    edgecolors=shape_colors.get(shape, 'gray'), facecolors='none',
                    label=f'{shape.title()} (Measured)', marker='o')
        plt.scatter(subset['diameter_mm'], subset['Re_calc'],
                    color=shape_colors.get(shape, 'gray'),
                    label=f'{shape.title()} (Calc)', marker='x')
    plt.xlabel('Diameter (mm)')
    plt.ylabel('Reynolds Number')
    plt.title('Reynolds Number: Measured vs Calculated')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('settling_velocity_comparison.png', dpi=300)
    plt.show()
    
    # ----- Figure 2: Error scatter plots (1x2, no error bars) -----
    plt.figure(figsize=(14, 6))
    
    # Settling velocity error
    plt.subplot(1, 2, 1)
    for shape in shapes:
        subset = df[df['shape_lower'] == shape].dropna(subset=['vs_error_%'])
        plt.scatter(subset['diameter_mm'], subset['vs_error_%'],
                    color=shape_colors.get(shape, 'gray'),
                    label=shape.title(), s=80)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    plt.xlabel('Diameter (mm)')
    plt.ylabel('Error (%)')
    plt.title('Settling Velocity Error')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xscale('log')
    plt.ylim(-100, 200)
    plt.legend()
    
    # Reynolds number error
    plt.subplot(1, 2, 2)
    for shape in shapes:
        subset = df[df['shape_lower'] == shape].dropna(subset=['Re_error_%'])
        plt.scatter(subset['diameter_mm'], subset['Re_error_%'],
                    color=shape_colors.get(shape, 'gray'),
                    label=shape.title(), s=80)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    plt.xlabel('Diameter (mm)')
    plt.ylabel('Error (%)')
    plt.title('Reynolds Number Error')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xscale('log')
    plt.ylim(-100, 200)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('settling_velocity_errors.png', dpi=300)
    plt.show()

# ==================== 6. Main Program ====================
if __name__ == '__main__':
    df, col_map = load_and_prepare('settlingdata_re.csv')
    df = compute_settling(df)
    save_results(df, col_map)
    plot_results(df, col_map)