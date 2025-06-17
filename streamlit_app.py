import streamlit as st
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import pandas as pd
import plotly.graph_objects as go
import time

# --- 1. Fuzzy Logic System Definition (Based on Paper) ---

def create_fuzzy_system():
    # Antecedents (Inputs)
    # CO2: 0 - 2000 ppm
    co2 = ctrl.Antecedent(np.arange(0, 2001, 1), 'co2')
    # PM10: 0 - 500 ug/m3
    pm10 = ctrl.Antecedent(np.arange(0, 501, 1), 'pm10')

    # Consequent (Output) - Fan Interval: 0 - 200 seconds
    fan_interval = ctrl.Consequent(np.arange(0, 201, 1), 'fan_interval')

    # Membership Functions for CO2 (trapmf)
    # Low: 0 - 1200 ppm
    co2['low'] = fuzz.trapmf(co2.universe, [0, 0, 1000, 1200])
    # Medium: 1000 - 1600 ppm
    co2['medium'] = fuzz.trapmf(co2.universe, [1000, 1200, 1400, 1600])
    # High: 1400 - 2000 ppm
    co2['high'] = fuzz.trapmf(co2.universe, [1400, 1600, 2000, 2000])

    # Membership Functions for PM10 (trapmf)
    # Low: 0 - 200 ug/m3
    pm10['low'] = fuzz.trapmf(pm10.universe, [0, 0, 150, 200])
    # Medium: 150 - 350 ug/m3
    pm10['medium'] = fuzz.trapmf(pm10.universe, [150, 200, 300, 350])
    # High: 300 - 500 ug/m3
    pm10['high'] = fuzz.trapmf(pm10.universe, [300, 350, 500, 500])

    # Membership Functions for Fan Interval (trimf for Fan-Short, Fan-Medium, Fan-High)
    # Fan-Stop implies 0 seconds, not a triangular function
    fan_interval['fan_stop'] = fuzz.trimf(fan_interval.universe, [0, 0, 0]) # Represents 0 duration
    # Fan-Short: 10 - 30 seconds
    fan_interval['fan_short'] = fuzz.trimf(fan_interval.universe, [10, 20, 30])
    # Fan-Medium: 100 - 140 seconds
    fan_interval['fan_medium'] = fuzz.trimf(fan_interval.universe, [100, 120, 140])
    # Fan-High: 120 - 200 seconds
    fan_interval['fan_high'] = fuzz.trimf(fan_interval.universe, [120, 160, 200])

    # --- Fuzzy Rules (from Table 2) ---
    rules = [
        # CO2 Low
        ctrl.Rule(co2['low'] & pm10['low'], fan_interval['fan_stop']),
        ctrl.Rule(co2['low'] & pm10['medium'], fan_interval['fan_short']),
        ctrl.Rule(co2['low'] & pm10['high'], fan_interval['fan_short']),

        # CO2 Medium
        ctrl.Rule(co2['medium'] & pm10['low'], fan_interval['fan_medium']),
        ctrl.Rule(co2['medium'] & pm10['medium'], fan_interval['fan_medium']),
        ctrl.Rule(co2['medium'] & pm10['high'], fan_interval['fan_medium']),

        # CO2 High
        ctrl.Rule(co2['high'] & pm10['low'], fan_interval['fan_high']),
        ctrl.Rule(co2['high'] & pm10['medium'], fan_interval['fan_high']),
        ctrl.Rule(co2['high'] & pm10['high'], fan_interval['fan_high']),
    ]

    # Control System
    fan_ctrl = ctrl.ControlSystem(rules)
    fan_simulation = ctrl.ControlSystemSimulation(fan_ctrl)
    return fan_simulation

# --- 2. AQI Calculation (Simplified for Simulation based on Paper's thresholds) ---

def calculate_co2_aqi(co2_ppm):
    # ASHRAE standard 1000 PPM max comfort
    if co2_ppm <= 800:
        return np.interp(co2_ppm, [0, 800], [0, 50]) # Good
    elif co2_ppm <= 1000:
        return np.interp(co2_ppm, [800, 1000], [50, 100]) # Moderate
    elif co2_ppm <= 1500:
        return np.interp(co2_ppm, [1000, 1500], [100, 200]) # Unhealthy for sensitive groups
    else:
        return np.interp(co2_ppm, [1500, 2000], [200, 300]) # Unhealthy to Very Unhealthy (max sensor 2000)

def calculate_pm10_aqi(pm10_ugm3):
    # US EPA standard 150 ug/m3 (24-hour)
    if pm10_ugm3 <= 50:
        return np.interp(pm10_ugm3, [0, 50], [0, 50]) # Good
    elif pm10_ugm3 <= 100:
        return np.interp(pm10_ugm3, [50, 100], [50, 100]) # Moderate
    elif pm10_ugm3 <= 150:
        return np.interp(pm10_ugm3, [100, 150], [100, 200]) # Unhealthy for sensitive groups
    else:
        return np.interp(pm10_ugm3, [150, 350], [200, 300]) # Unhealthy to Very Unhealthy (max sensor 500)

def is_safe_aqi(co2_ppm, pm10_ugm3):
    return co2_ppm <= 1000 and pm10_ugm3 <= 150

# --- 3. Simulation Engine ---

def run_simulation(initial_co2, initial_pm10, duration_minutes,
                   co2_increase_rate, pm10_increase_rate,
                   co2_reduction_rate_per_sec, pm10_reduction_rate_per_sec,
                   fan_simulation_instance, progress_bar):

    sim_duration_seconds = duration_minutes * 60
    current_co2 = initial_co2
    current_pm10 = initial_pm10
    fan_on_duration = 0 # How long the fan needs to run from current cycle
    fan_active = False
    time_since_last_check = 0 # To track 14-second interval

    # Data to store for plotting
    timestamps = []
    co2_values = []
    pm10_values = []
    fan_status = [] # 0 for off, 1 for on
    co2_aqi_values = []
    pm10_aqi_values = []
    safe_aqi_flags = [] # True if safe, False otherwise

    # Simulation loop (1 second increments)
    for t_sec in range(sim_duration_seconds + 1):
        # Update progress bar
        progress_bar.progress((t_sec / sim_duration_seconds))

        timestamps.append(t_sec)
        co2_values.append(current_co2)
        pm10_values.append(current_pm10)
        co2_aqi_values.append(calculate_co2_aqi(current_co2))
        pm10_aqi_values.append(calculate_pm10_aqi(current_pm10))
        safe_aqi_flags.append(is_safe_aqi(current_co2, current_pm10))


        # If fan is currently active from a previous fuzzy decision
        if fan_on_duration > 0:
            fan_active = True
            current_co2 = max(0, current_co2 - co2_reduction_rate_per_sec)
            current_pm10 = max(0, current_pm10 - pm10_reduction_rate_per_sec)
            fan_on_duration -= 1
        else:
            fan_active = False
            # Pollutants gradually increase when fan is off
            current_co2 += co2_increase_rate
            current_pm10 += pm10_increase_rate

        fan_status.append(1 if fan_active else 0)

        # Ensure values don't go below zero or exceed sensor max (for CO2/PM10 plots)
        current_co2 = min(max(0, current_co2), 2500) # Allowing slightly higher than 2000 for illustration
        current_pm10 = min(max(0, current_pm10), 600) # Allowing slightly higher than 500 for illustration

        time_since_last_check += 1

        # Every 14 seconds, perform fuzzy logic calculation
        if time_since_last_check >= 14:
            # print(f"Time {t_sec}s: Running fuzzy logic. CO2: {current_co2:.2f}, PM10: {current_pm10:.2f}")
            fan_simulation_instance.input['co2'] = current_co2
            fan_simulation_instance.input['pm10'] = current_pm10

            try:
                fan_simulation_instance.compute()
                calculated_interval = fan_simulation_instance.output['fan_interval']
                fan_on_duration = int(round(calculated_interval))
                # print(f"    Fuzzy output: {calculated_interval:.2f} seconds. Fan will run for {fan_on_duration}s.")
            except ValueError as e:
                st.warning(f"Fuzzy logic computation error at time {t_sec}s: {e}. Defaulting fan interval to 0.")
                fan_on_duration = 0 # Default to 0 if computation fails (e.g., input out of range)
            time_since_last_check = 0 # Reset timer

    results_df = pd.DataFrame({
        'Time_Seconds': timestamps,
        'CO2_PPM': co2_values,
        'PM10_UG/M3': pm10_values,
        'Fan_Active': fan_status,
        'CO2_AQI': co2_aqi_values,
        'PM10_AQI': pm10_aqi_values,
        'Is_AQI_Safe': safe_aqi_flags
    })
    return results_df

# --- 4. Streamlit UI ---

st.set_page_config(layout="wide", page_title="Indoor AQI Fuzzy Logic Simulation")

st.title("🏡 Indoor Air Quality Monitoring System Simulation")
st.markdown("---")

st.markdown("""
This application simulates the **Indoor Air Quality Monitoring System with Fuzzy Logic Control Based On IOT** described in the research paper by Fadli Pradityo and Nico Surantha.
It models CO2 and PM10 levels, applies a Mamdani fuzzy logic system to control an exhaust fan, and visualizes the air quality over time.
""")

st.subheader("Simulation Parameters")

col1, col2, col3 = st.columns(3)

with col1:
    initial_co2 = st.slider("Initial CO2 (PPM)", 500, 2000, 1500, 100)
    initial_pm10 = st.slider("Initial PM10 (µg/m³)", 50, 500, 300, 50)

with col2:
    duration_minutes = st.slider("Simulation Duration (minutes)", 5, 120, 30, 5)
    st.markdown("---")
    st.write("Environmental Increase Rates (per second when fan is OFF):")
    co2_increase_rate = st.slider("CO2 Increase Rate (PPM/s)", 0.0, 1.0, 0.5, 0.1)
    pm10_increase_rate = st.slider("PM10 Increase Rate (µg/m³/s)", 0.0, 0.5, 0.1, 0.05)


with col3:
    st.write("Exhaust Fan Reduction Rates (per second when fan is ON):")
    # Based on paper's stated averages to reduce High levels
    # High CO2 (1400-2000 to safe 1000) in 155s => approx (1500-1000)/155 = 3.2 PPM/s
    co2_reduction_rate = st.slider("CO2 Reduction Rate (PPM/s)", 1.0, 10.0, 3.2, 0.5)
    # High PM10 (300-500 to safe 150) in 17s => approx (350-150)/17 = 11.7 µg/m³/s
    pm10_reduction_rate = st.slider("PM10 Reduction Rate (µg/m³/s)", 5.0, 30.0, 11.7, 1.0)


if st.button("Run Simulation", type="primary"):
    st.markdown("---")
    st.subheader("Simulation Results")

    # Create fuzzy system
    fuzzy_system = create_fuzzy_system()

    # Progress bar for simulation
    progress_text = "Simulation in progress. Please wait..."
    my_bar = st.progress(0, text=progress_text)

    # Run simulation
    simulation_data = run_simulation(
        initial_co2, initial_pm10, duration_minutes,
        co2_increase_rate, pm10_increase_rate,
        co2_reduction_rate, pm10_reduction_rate,
        fuzzy_system, my_bar
    )
    my_bar.empty() # Clear the progress bar after completion

    # --- Metrics ---
    total_safe_co2_time = simulation_data[simulation_data['CO2_PPM'] <= 1000].shape[0]
    total_safe_pm10_time = simulation_data[simulation_data['PM10_UG/M3'] <= 150].shape[0]
    total_safe_aqi_time = simulation_data[simulation_data['Is_AQI_Safe']].shape[0]
    total_fan_on_time = simulation_data['Fan_Active'].sum()


    st.markdown(f"""
    **Simulation Summary (Duration: {duration_minutes} minutes)**
    * **Total time CO2 was within safe limits (<1000 PPM):** {total_safe_co2_time} seconds
    * **Total time PM10 was within safe limits (<150 µg/m³):** {total_safe_pm10_time} seconds
    * **Total time both CO2 and PM10 were safe:** {total_safe_aqi_time} seconds
    * **Total time exhaust fan was active:** {total_fan_on_time} seconds
    """)
    st.info("Note: The paper's safe thresholds are CO2 <= 1000 PPM and PM10 <= 150 µg/m³.")


    # --- Plotting Results ---

    st.subheader("Air Quality Trends Over Time")

    # Plot CO2
    fig_co2 = go.Figure()
    fig_co2.add_trace(go.Scatter(x=simulation_data['Time_Seconds'], y=simulation_data['CO2_PPM'],
                                 mode='lines', name='CO2 (PPM)', line=dict(color='red')))
    fig_co2.add_hline(y=1000, line_dash="dash", line_color="green", annotation_text="ASHRAE CO2 Safe Limit (1000 PPM)", annotation_position="top right")
    fig_co2.update_layout(title="CO2 Concentration Over Time",
                          xaxis_title="Time (Seconds)", yaxis_title="CO2 (PPM)",
                          template="plotly_white", height=400)
    st.plotly_chart(fig_co2, use_container_width=True)

    # Plot PM10
    fig_pm10 = go.Figure()
    fig_pm10.add_trace(go.Scatter(x=simulation_data['Time_Seconds'], y=simulation_data['PM10_UG/M3'],
                                  mode='lines', name='PM10 (µg/m³)', line=dict(color='blue')))
    fig_pm10.add_hline(y=150, line_dash="dash", line_color="green", annotation_text="EPA PM10 Safe Limit (150 µg/m³)", annotation_position="top right")
    fig_pm10.update_layout(title="PM10 Concentration Over Time",
                            xaxis_title="Time (Seconds)", yaxis_title="PM10 (µg/m³)",
                            template="plotly_white", height=400)
    st.plotly_chart(fig_pm10, use_container_width=True)

    # Plot Fan Status
    fig_fan = go.Figure()
    fig_fan.add_trace(go.Scatter(x=simulation_data['Time_Seconds'], y=simulation_data['Fan_Active'],
                                 mode='lines', name='Fan Status (0=Off, 1=On)', line=dict(color='purple', shape='hv')))
    fig_fan.update_layout(title="Exhaust Fan Activity Over Time",
                          xaxis_title="Time (Seconds)", yaxis_title="Fan Status",
                          yaxis=dict(tickvals=[0, 1], ticktext=['Off', 'On']),
                          template="plotly_white", height=250)
    st.plotly_chart(fig_fan, use_container_width=True)

    st.subheader("Simulated AQI Over Time")
    fig_aqi = go.Figure()
    fig_aqi.add_trace(go.Scatter(x=simulation_data['Time_Seconds'], y=simulation_data['CO2_AQI'],
                                 mode='lines', name='CO2 AQI', line=dict(color='red')))
    fig_aqi.add_trace(go.Scatter(x=simulation_data['Time_Seconds'], y=simulation_data['PM10_AQI'],
                                 mode='lines', name='PM10 AQI', line=dict(color='blue')))
    fig_aqi.add_hline(y=100, line_dash="dash", line_color="orange", annotation_text="Moderate AQI Threshold", annotation_position="top right")
    fig_aqi.add_hline(y=150, line_dash="dash", line_color="red", annotation_text="Unhealthy for Sensitive Groups Threshold", annotation_position="top right")
    fig_aqi.update_layout(title="Simulated AQI Values",
                          xaxis_title="Time (Seconds)", yaxis_title="AQI Value",
                          template="plotly_white", height=400)
    st.plotly_chart(fig_aqi, use_container_width=True)

    st.subheader("Raw Simulation Data")
    st.dataframe(simulation_data)
