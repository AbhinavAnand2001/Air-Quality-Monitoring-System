# Import necessary libraries for Shiny, fuzzy logic, data handling, and plotting
from shiny import App, ui, render, reactive
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import pandas as pd
import plotly.graph_objects as go
import time # For simulation delay and progress bar

# --- 1. Fuzzy Logic System Definition (Based on Paper) ---
# This function sets up the fuzzy logic rules and membership functions
# based on the provided paper.
def create_fuzzy_system():
    # Define the input variables (antecedents) and their universes of discourse.
    # CO2 sensor range: 0 - 2000 ppm
    co2 = ctrl.Antecedent(np.arange(0, 2001, 1), 'co2')
    # PM10 sensor range: 0 - 500 ug/m3
    pm10 = ctrl.Antecedent(np.arange(0, 501, 1), 'pm10')

    # Define the output variable (consequent) and its universe of discourse.
    # Exhaust fan interval range: 0 - 200 seconds
    fan_interval = ctrl.Consequent(np.arange(0, 201, 1), 'fan_interval')

    # Define membership functions for CO2 using trapezoidal shapes (trapmf).
    # These ranges are directly from the paper's Fig 4.
    co2['low'] = fuzz.trapmf(co2.universe, [0, 0, 1000, 1200])
    co2['medium'] = fuzz.trapmf(co2.universe, [1000, 1200, 1400, 1600])
    co2['high'] = fuzz.trapmf(co2.universe, [1400, 1600, 2000, 2000])

    # Define membership functions for PM10 using trapezoidal shapes (trapmf).
    # These ranges are directly from the paper's Fig 5.
    pm10['low'] = fuzz.trapmf(pm10.universe, [0, 0, 150, 200])
    pm10['medium'] = fuzz.trapmf(pm10.universe, [150, 200, 300, 350])
    pm10['high'] = fuzz.trapmf(pm10.universe, [300, 350, 500, 500])

    # Define membership functions for Fan Interval using triangular shapes (trimf).
    # 'fan_stop' is an edge case representing 0 duration.
    # These ranges are directly from the paper's Fig 6.
    fan_interval['fan_stop'] = fuzz.trimf(fan_interval.universe, [0, 0, 0])
    fan_interval['fan_short'] = fuzz.trimf(fan_interval.universe, [10, 20, 30])
    fan_interval['fan_medium'] = fuzz.trimf(fan_interval.universe, [100, 120, 140])
    fan_interval['fan_high'] = fuzz.trimf(fan_interval.universe, [120, 160, 200])

    # --- Fuzzy Rules (from Paper's Table 2) ---
    # These rules define how the inputs map to the output.
    rules = [
        # If CO2 is Low AND PM10 is Low, then Fan is Stop
        ctrl.Rule(co2['low'] & pm10['low'], fan_interval['fan_stop']),
        # If CO2 is Low AND PM10 is Medium, then Fan is Short
        ctrl.Rule(co2['low'] & pm10['medium'], fan_interval['fan_short']),
        # If CO2 is Low AND PM10 is High, then Fan is Short
        ctrl.Rule(co2['low'] & pm10['high'], fan_interval['fan_short']),

        # If CO2 is Medium AND PM10 is Low, then Fan is Medium
        ctrl.Rule(co2['medium'] & pm10['low'], fan_interval['fan_medium']),
        # If CO2 is Medium AND PM10 is Medium, then Fan is Medium
        ctrl.Rule(co2['medium'] & pm10['medium'], fan_interval['fan_medium']),
        # If CO2 is Medium AND PM10 is High, then Fan is Medium
        ctrl.Rule(co2['medium'] & pm10['high'], fan_interval['fan_medium']),

        # If CO2 is High AND PM10 is Low, then Fan is High
        ctrl.Rule(co2['high'] & pm10['low'], fan_interval['fan_high']),
        # If CO2 is High AND PM10 is Medium, then Fan is High
        ctrl.Rule(co2['high'] & pm10['medium'], fan_interval['fan_high']),
        # If CO2 is High AND PM10 is High, then Fan is High
        ctrl.Rule(co2['high'] & pm10['high'], fan_interval['fan_high']),
    ]

    # Create the control system and simulation object.
    fan_ctrl = ctrl.ControlSystem(rules)
    fan_simulation = ctrl.ControlSystemSimulation(fan_ctrl)
    return fan_simulation

# --- 2. AQI Calculation (Simplified for Simulation based on Paper's thresholds) ---
# These functions convert CO2 and PM10 values to a simplified Air Quality Index (AQI)
# based on the paper's discussion of ASHRAE and EPA standards.
def calculate_co2_aqi(co2_ppm):
    # ASHRAE standard for comfort: 1000 PPM maximum.
    # Simplified linear interpolation for AQI ranges.
    if co2_ppm <= 800:
        return np.interp(co2_ppm, [0, 800], [0, 50]) # Good
    elif co2_ppm <= 1000:
        return np.interp(co2_ppm, [800, 1000], [50, 100]) # Moderate
    elif co2_ppm <= 1500:
        return np.interp(co2_ppm, [1000, 1500], [100, 200]) # Unhealthy for sensitive groups
    else:
        return np.interp(co2_ppm, [1500, 2000], [200, 300]) # Unhealthy to Very Unhealthy (max sensor 2000)

def calculate_pm10_aqi(pm10_ugm3):
    # US EPA standard for PM10: 150 ug/m3 (24-hour average).
    # Simplified linear interpolation for AQI ranges.
    if pm10_ugm3 <= 50:
        return np.interp(pm10_ugm3, [0, 50], [0, 50]) # Good
    elif pm10_ugm3 <= 100:
        return np.interp(pm10_ugm3, [50, 100], [50, 100]) # Moderate
    elif pm10_ugm3 <= 150:
        return np.interp(pm10_ugm3, [100, 150], [100, 200]) # Unhealthy for sensitive groups
    else:
        return np.interp(pm10_ugm3, [150, 350], [200, 300]) # Unhealthy to Very Unhealthy (max sensor 500)

# Checks if both CO2 and PM10 are within the "safe" limits as defined in the paper.
def is_safe_aqi(co2_ppm, pm10_ugm3):
    return co2_ppm <= 1000 and pm10_ugm3 <= 150

# --- 3. Simulation Engine ---
# This function simulates the air quality over time based on inputs
# and the fuzzy logic system, updating the concentrations and fan status.
def run_simulation(initial_co2, initial_pm10, duration_minutes,
                   co2_increase_rate, pm10_increase_rate,
                   co2_reduction_rate_per_sec, pm10_reduction_rate_per_sec,
                   fan_simulation_instance, progress_obj):

    sim_duration_seconds = duration_minutes * 60
    current_co2 = initial_co2
    current_pm10 = initial_pm10
    fan_on_duration = 0 # Remaining time fan needs to run from current cycle
    fan_active = False # Current state of the fan
    time_since_last_check = 0 # Counter for the 14-second fuzzy logic check interval

    # Lists to store simulation data for plotting and analysis
    timestamps = []
    co2_values = []
    pm10_values = []
    fan_status = [] # 0 for off, 1 for on
    co2_aqi_values = []
    pm10_aqi_values = []
    safe_aqi_flags = [] # True if AQI is safe, False otherwise

    # Main simulation loop, running second by second
    for t_sec in range(sim_duration_seconds + 1):
        # Update the progress bar in the Shiny UI
        progress_obj.set(t_sec / sim_duration_seconds, message=f"Simulating... {t_sec}/{sim_duration_seconds}s")

        # Record current values
        timestamps.append(t_sec)
        co2_values.append(current_co2)
        pm10_values.append(current_pm10)
        co2_aqi_values.append(calculate_co2_aqi(current_co2))
        pm10_aqi_values.append(calculate_pm10_aqi(current_pm10))
        safe_aqi_flags.append(is_safe_aqi(current_co2, current_pm10))

        # Check if the fan is currently running based on a previous fuzzy decision
        if fan_on_duration > 0:
            fan_active = True
            # Reduce pollutant concentrations when fan is active
            current_co2 = max(0, current_co2 - co2_reduction_rate_per_sec)
            current_pm10 = max(0, current_pm10 - pm10_reduction_rate_per_sec)
            fan_on_duration -= 1 # Decrement remaining fan active time
        else:
            fan_active = False
            # Pollutants gradually increase when fan is off
            current_co2 += co2_increase_rate
            current_pm10 += pm10_increase_rate

        fan_status.append(1 if fan_active else 0)

        # Ensure concentrations stay within reasonable bounds (e.g., non-negative)
        # Allowing slightly higher than sensor max for visual effect if inputs are high.
        current_co2 = min(max(0, current_co2), 2500)
        current_pm10 = min(max(0, current_pm10), 600)

        time_since_last_check += 1 # Increment time since last fuzzy check

        # Every 14 seconds (as per paper), re-evaluate with fuzzy logic
        if time_since_last_check >= 14:
            fan_simulation_instance.input['co2'] = current_co2
            fan_simulation_instance.input['pm10'] = current_pm10

            try:
                # Compute the fuzzy logic output (fan interval)
                fan_simulation_instance.compute()
                calculated_interval = fan_simulation_instance.output['fan_interval']
                # Round the result to the nearest integer for fan operation time
                fan_on_duration = int(round(calculated_interval))
            except ValueError as e:
                # Handle cases where fuzzy computation might fail (e.g., input out of defined range)
                print(f"Fuzzy logic computation error at time {t_sec}s: {e}. Defaulting fan interval to 0.")
                fan_on_duration = 0 # Default to 0 if computation fails
            time_since_last_check = 0 # Reset the timer for the next check

    # Create a Pandas DataFrame from the collected simulation data
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

# --- 4. Shiny UI (User Interface Definition) ---
# This defines the layout and interactive elements of your web application.
app_ui = ui.page_fluid( # Uses a fluid layout for responsiveness
    ui.h2("🏡 Indoor Air Quality Monitoring System Simulation"),
    ui.markdown("---"),
    ui.markdown("""
        This application simulates the **Indoor Air Quality Monitoring System with Fuzzy Logic Control Based On IOT**
        described in the research paper by Fadli Pradityo and Nico Surantha.
        It models CO2 and PM10 levels, applies a Mamdani fuzzy logic system to control an exhaust fan,
        and visualizes the air quality over time.
    """),

    ui.h3("Simulation Parameters"),
    ui.layout_sidebar( # Creates a sidebar for inputs and main panel for outputs
        ui.panel_sidebar(
            ui.h4("Initial Conditions"),
            ui.input_slider("initial_co2", "Initial CO2 (PPM)", 500, 2000, 1500, step=100),
            ui.input_slider("initial_pm10", "Initial PM10 (µg/m³)", 50, 500, 300, step=50),

            ui.hr(), # Horizontal rule for visual separation
            ui.h4("Environmental Dynamics"),
            ui.input_slider("duration_minutes", "Simulation Duration (minutes)", 5, 120, 30, step=5),
            ui.input_slider("co2_increase_rate", "CO2 Increase Rate (PPM/s, fan OFF)", 0.0, 1.0, 0.5, step=0.1),
            ui.input_slider("pm10_increase_rate", "PM10 Increase Rate (µg/m³/s, fan OFF)", 0.0, 0.5, 0.1, step=0.05),

            ui.hr(),
            ui.h4("Exhaust Fan Efficiency"),
            ui.input_slider("co2_reduction_rate", "CO2 Reduction Rate (PPM/s, fan ON)", 1.0, 10.0, 3.2, step=0.5),
            ui.input_slider("pm10_reduction_rate", "PM10 Reduction Rate (µg/m³/s, fan ON)", 5.0, 30.0, 11.7, step=1.0),

            ui.hr(),
            ui.input_action_button("run_sim_button", "Run Simulation", class_="btn-primary") # Button to trigger simulation
        ),
        ui.panel_main(
            ui.h3("Simulation Results"),
            ui.output_ui("simulation_summary"), # Placeholder for summary text
            ui.output_plot("co2_plot"), # Placeholder for CO2 plot
            ui.output_plot("pm10_plot"), # Placeholder for PM10 plot
            ui.output_plot("fan_plot"), # Placeholder for fan status plot
            ui.output_plot("aqi_plot"), # Placeholder for AQI plot
            ui.h3("Raw Simulation Data"),
            ui.output_data_frame("simulation_data_table") # Placeholder for data table
        )
    )
)

# --- 5. Shiny Server (Backend Logic) ---
# This defines how the app reacts to user inputs and generates outputs.
def server(input, output, session):
    # Reactive value to store the simulation results DataFrame
    sim_data = reactive.Value(pd.DataFrame())

    # Observer for the "Run Simulation" button click
    @reactive.Effect
    @reactive.event(input.run_sim_button) # This decorator makes the code run only when the button is clicked
    def _():
        # Display a progress bar during simulation
        with ui.Progress(min=0, max=1) as p:
            p.set(0.0, message="Initializing simulation...")

            # Create the fuzzy logic system instance
            fuzzy_system = create_fuzzy_system()

            # Run the simulation and store the results in the reactive value
            current_sim_data = run_simulation(
                input.initial_co2(), input.initial_pm10(), input.duration_minutes(),
                input.co2_increase_rate(), input.pm10_increase_rate(),
                input.co2_reduction_rate(), input.pm10_reduction_rate(),
                fuzzy_system, p # Pass the progress object
            )
            sim_data.set(current_sim_data)

    # --- Output Renderers ---

    # Render the simulation summary text
    @render.ui
    def simulation_summary():
        df = sim_data.get()
        if df.empty:
            return ui.p("Click 'Run Simulation' to see results.")

        # Calculate summary metrics from the simulation data
        total_safe_co2_time = df[df['CO2_PPM'] <= 1000].shape[0]
        total_safe_pm10_time = df[df['PM10_UG/M3'] <= 150].shape[0]
        total_safe_aqi_time = df[df['Is_AQI_Safe']].shape[0]
        total_fan_on_time = df['Fan_Active'].sum()
        duration_minutes = input.duration_minutes()

        return ui.markdown(f"""
        **Simulation Summary (Duration: {duration_minutes} minutes)**
        * **Total time CO2 was within safe limits (<1000 PPM):** {total_safe_co2_time} seconds
        * **Total time PM10 was within safe limits (<150 µg/m³):** {total_safe_pm10_time} seconds
        * **Total time both CO2 and PM10 were safe:** {total_safe_aqi_time} seconds
        * **Total time exhaust fan was active:** {total_fan_on_time} seconds
        """)

    # Render the CO2 concentration plot
    @render.plot(alt="CO2 Concentration Over Time")
    def co2_plot():
        df = sim_data.get()
        if df.empty:
            return None

        fig_co2 = go.Figure()
        fig_co2.add_trace(go.Scatter(x=df['Time_Seconds'], y=df['CO2_PPM'],
                                     mode='lines', name='CO2 (PPM)', line=dict(color='red')))
        fig_co2.add_hline(y=1000, line_dash="dash", line_color="green",
                          annotation_text="ASHRAE CO2 Safe Limit (1000 PPM)", annotation_position="top right")
        fig_co2.update_layout(title="CO2 Concentration Over Time",
                              xaxis_title="Time (Seconds)", yaxis_title="CO2 (PPM)",
                              template="plotly_white", height=400)
        return fig_co2

    # Render the PM10 concentration plot
    @render.plot(alt="PM10 Concentration Over Time")
    def pm10_plot():
        df = sim_data.get()
        if df.empty:
            return None

        fig_pm10 = go.Figure()
        fig_pm10.add_trace(go.Scatter(x=df['Time_Seconds'], y=df['PM10_UG/M3'],
                                      mode='lines', name='PM10 (µg/m³)', line=dict(color='blue')))
        fig_pm10.add_hline(y=150, line_dash="dash", line_color="green",
                           annotation_text="EPA PM10 Safe Limit (150 µg/m³)", annotation_position="top right")
        fig_pm10.update_layout(title="PM10 Concentration Over Time",
                                xaxis_title="Time (Seconds)", yaxis_title="PM10 (µg/m³)",
                                template="plotly_white", height=400)
        return fig_pm10

    # Render the Exhaust Fan activity plot
    @render.plot(alt="Exhaust Fan Activity Over Time")
    def fan_plot():
        df = sim_data.get()
        if df.empty:
            return None

        fig_fan = go.Figure()
        fig_fan.add_trace(go.Scatter(x=df['Time_Seconds'], y=df['Fan_Active'],
                                     mode='lines', name='Fan Status (0=Off, 1=On)', line=dict(color='purple', shape='hv')))
        fig_fan.update_layout(title="Exhaust Fan Activity Over Time",
                              xaxis_title="Time (Seconds)", yaxis_title="Fan Status",
                              yaxis=dict(tickvals=[0, 1], ticktext=['Off', 'On']),
                              template="plotly_white", height=250)
        return fig_fan

    # Render the AQI values plot
    @render.plot(alt="Simulated AQI Values")
    def aqi_plot():
        df = sim_data.get()
        if df.empty:
            return None

        fig_aqi = go.Figure()
        fig_aqi.add_trace(go.Scatter(x=df['Time_Seconds'], y=df['CO2_AQI'],
                                     mode='lines', name='CO2 AQI', line=dict(color='red')))
        fig_aqi.add_trace(go.Scatter(x=df['Time_Seconds'], y=df['PM10_AQI'],
                                     mode='lines', name='PM10 AQI', line=dict(color='blue')))
        fig_aqi.add_hline(y=100, line_dash="dash", line_color="orange",
                          annotation_text="Moderate AQI Threshold", annotation_position="top right")
        fig_aqi.add_hline(y=150, line_dash="dash", line_color="red",
                          annotation_text="Unhealthy for Sensitive Groups Threshold", annotation_position="top right")
        fig_aqi.update_layout(title="Simulated AQI Values",
                              xaxis_title="Time (Seconds)", yaxis_title="AQI Value",
                              template="plotly_white", height=400)
        return fig_aqi

    # Render the raw simulation data table
    @render.data_frame
    def simulation_data_table():
        return render.DataGrid(sim_data.get(), row_selection_mode="multiple")


# Create the Shiny application instance
app = App(app_ui, server)
