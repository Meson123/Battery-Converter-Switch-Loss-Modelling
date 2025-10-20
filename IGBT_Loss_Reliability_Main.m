%% IGBT Loss and Reliability Modeling for EV Converters
% Based on Semikron Application Manual, Power Cycle Model AN 21-001, and
% Infineon IKW40N120H3 datasheet

clear all; close all; clc;

%% 1. IGBT Parameters (IKW40N120H3)
IGBT.V_CE0 = 0.95;          % Threshold voltage [V] (from datasheet)
IGBT.r_CE = 0.028;          % On-state resistance [Ohm]
IGBT.V_rated = 1200;        % Voltage rating [V]
IGBT.I_rated = 40;          % Current rating [A]
IGBT.Tj_max = 175;          % Max junction temp [°C]
IGBT.Tj_min = -40;          % Min junction temp [°C]

% Switching energy parameters (normalized to datasheet conditions)
% Datasheet conditions: Vcc=600V, Ic=40A, Tj=25°C, Rg=15Ohm
IGBT.E_on_ref = 2.4e-3;     % Turn-on energy [J] at reference
IGBT.E_off_ref = 1.5e-3;    % Turn-off energy [J] at reference
IGBT.V_ref = 600;           % Reference voltage [V]
IGBT.I_ref = 40;            % Reference current [A]
IGBT.Tj_ref = 25;           % Reference temp [°C]

% Diode parameters
Diode.V_F0 = 1.0;           % Forward voltage drop [V]
Diode.r_F = 0.02;           % Forward resistance [Ohm]
Diode.E_rec_ref = 0.8e-3;   % Recovery energy [J]

%% 2. Thermal Parameters
Thermal.Rth_jc = 0.48;      % Junction-to-case thermal resistance [K/W]
Thermal.Rth_ch = 0.10;      % Case-to-heatsink thermal resistance [K/W]
Thermal.Rth_ha = 0.50;      % Heatsink-to-ambient thermal resistance [K/W]
Thermal.Cth_j = 0.05;       % Junction thermal capacitance [J/K]
Thermal.Cth_c = 0.50;       % Case thermal capacitance [J/K]
Thermal.T_amb = 40;         % Ambient temperature [°C]

%% 3. Operating Conditions (EV Driving Cycle)
Oper.V_dc = 400;            % DC bus voltage [V]
Oper.I_peak = 200;          % Peak current [A]
Oper.f_sw = 10e3;           % Switching frequency [Hz]
Oper.m_index = 0.8;         % Modulation index
Oper.power_factor = 0.95;   % Power factor
Oper.f_output = 100;        % Output frequency [Hz]

%% 4. Power Cycling Parameters (Semikron AN 21-001)
% Power cycling model: Nf = A * (dTj)^-B * exp(Ea/kB*Tj_mean)
PowerCycle.A = 9.34e14;     % Scaling factor
PowerCycle.B = -4.416;      % Temperature swing exponent
PowerCycle.Ea = 0.129;      % Activation energy [eV]
PowerCycle.kB = 8.617e-5;   % Boltzmann constant [eV/K]
PowerCycle.t_on = 1;        % Heating time [s]

%% 5. Degradation Mechanisms
% Bond wire lift-off
Degradation.bond_wire.enabled = true;
Degradation.bond_wire.R_increase_rate = 0.001; % Per cycle

% Solder fatigue
Degradation.solder.enabled = true;
Degradation.solder.Rth_increase_rate = 0.0005; % Per cycle

% Gate oxide degradation
Degradation.gate_oxide.enabled = true;
Degradation.gate_oxide.V_th_drift_rate = 0.0001; % V per cycle

%% 6. Simulation Parameters
Sim.t_mission = 3600;       % Mission profile time [s] (1 hour)
Sim.dt = 1e-5;              % Time step [s]
Sim.cycles_per_year = 1000; % Number of driving cycles per year
Sim.lifetime_years = 10;    % Target lifetime [years]

%% 7. Generate Mission Profile (Simplified EV cycle)
time = 0:Sim.dt:Sim.t_mission;
% Sinusoidal current with modulation
I_load = Oper.I_peak * abs(sin(2*pi*Oper.f_output*time)) .* ...
         (0.5 + 0.5*sin(2*pi*0.01*time)); % Slow modulation envelope

%% 8. Calculate Losses for Single Cycle
fprintf('Calculating losses for single mission profile...\n');

% Preallocate arrays
P_cond_IGBT = zeros(size(time));
P_sw_IGBT = zeros(size(time));
P_cond_Diode = zeros(size(time));
P_sw_Diode = zeros(size(time));
Tj_instant = zeros(size(time));

% Initial conditions
Tj_instant(1) = Thermal.T_amb + 20;
R_CE = IGBT.r_CE;
V_CE0 = IGBT.V_CE0;

for i = 2:length(time)
    % Current magnitude
    I_c = abs(I_load(i));
    
    % Update parameters based on temperature (simplified)
    temp_factor = 1 + 0.005*(Tj_instant(i-1) - 25);
    R_CE_temp = R_CE * temp_factor;
    V_CE0_temp = V_CE0 * (1 + 0.001*(Tj_instant(i-1) - 25));
    
    % Conduction losses (assuming 50% duty for IGBT, 50% for diode)
    P_cond_IGBT(i) = (V_CE0_temp + R_CE_temp * I_c) * I_c * 0.5;
    P_cond_Diode(i) = (Diode.V_F0 + Diode.r_F * I_c) * I_c * 0.5;
    
    % Switching losses (scaled from reference conditions)
    V_factor = Oper.V_dc / IGBT.V_ref;
    I_factor = I_c / IGBT.I_ref;
    Tj_factor = 1 + 0.003*(Tj_instant(i-1) - IGBT.Tj_ref);
    
    E_on = IGBT.E_on_ref * V_factor * I_factor^1.3 * Tj_factor;
    E_off = IGBT.E_off_ref * V_factor * I_factor^0.6 * Tj_factor;
    E_rec = Diode.E_rec_ref * V_factor * I_factor^0.6;
    
    P_sw_IGBT(i) = (E_on + E_off) * Oper.f_sw;
    P_sw_Diode(i) = E_rec * Oper.f_sw;
    
    % Total power dissipation
    P_total = P_cond_IGBT(i) + P_sw_IGBT(i) + P_cond_Diode(i) + P_sw_Diode(i);
    
    % Thermal model (simplified first-order)
    Rth_total = Thermal.Rth_jc + Thermal.Rth_ch + Thermal.Rth_ha;
    tau_thermal = Rth_total * Thermal.Cth_j;
    
    dTj = (P_total * Rth_total - (Tj_instant(i-1) - Thermal.T_amb)) / tau_thermal;
    Tj_instant(i) = Tj_instant(i-1) + dTj * Sim.dt;
end

%% 9. Calculate Temperature Swing Statistics
Tj_mean = mean(Tj_instant);
Tj_max_cycle = max(Tj_instant);
Tj_min_cycle = min(Tj_instant);
dTj = Tj_max_cycle - Tj_min_cycle;

fprintf('Temperature Statistics:\n');
fprintf('  Mean Junction Temp: %.2f °C\n', Tj_mean);
fprintf('  Max Junction Temp: %.2f °C\n', Tj_max_cycle);
fprintf('  Temperature Swing (dTj): %.2f K\n', dTj);

%% 10. Power Cycling Lifetime Calculation (Semikron AN 21-001)
% Nf = A * (dTj)^B * exp(Ea/(kB*Tj_mean))
Tj_mean_K = Tj_mean + 273.15;
Nf = PowerCycle.A * (dTj^PowerCycle.B) * ...
     exp(PowerCycle.Ea / (PowerCycle.kB * Tj_mean_K));

fprintf('\nPower Cycling Lifetime:\n');
fprintf('  Cycles to Failure: %.2e cycles\n', Nf);
fprintf('  Years to Failure: %.2f years\n', Nf/(Sim.cycles_per_year));

%% 11. Loss Breakdown
P_cond_IGBT_avg = mean(P_cond_IGBT);
P_sw_IGBT_avg = mean(P_sw_IGBT);
P_cond_Diode_avg = mean(P_cond_Diode);
P_sw_Diode_avg = mean(P_sw_Diode);
P_total_avg = P_cond_IGBT_avg + P_sw_IGBT_avg + P_cond_Diode_avg + P_sw_Diode_avg;

fprintf('\nAverage Power Losses:\n');
fprintf('  IGBT Conduction: %.2f W (%.1f%%)\n', P_cond_IGBT_avg, 100*P_cond_IGBT_avg/P_total_avg);
fprintf('  IGBT Switching: %.2f W (%.1f%%)\n', P_sw_IGBT_avg, 100*P_sw_IGBT_avg/P_total_avg);
fprintf('  Diode Conduction: %.2f W (%.1f%%)\n', P_cond_Diode_avg, 100*P_cond_Diode_avg/P_total_avg);
fprintf('  Diode Switching: %.2f W (%.1f%%)\n', P_sw_Diode_avg, 100*P_sw_Diode_avg/P_total_avg);
fprintf('  Total Average Loss: %.2f W\n', P_total_avg);

%% 12. Reliability Analysis - Multiple Operating Conditions
fprintf('\n=== RELIABILITY ANALYSIS ===\n');

% Define operating scenarios
scenarios = {
    'Urban Driving', 150, 0.6, 30;
    'Highway Driving', 200, 0.8, 40;
    'Aggressive Driving', 250, 0.95, 50;
    'Continuous High Power', 280, 1.0, 60
};

reliability_results = zeros(size(scenarios, 1), 4);

for s = 1:size(scenarios, 1)
    scenario_name = scenarios{s, 1};
    I_scenario = scenarios{s, 2};
    m_scenario = scenarios{s, 3};
    T_amb_scenario = scenarios{s, 4};
    
    % Simplified calculation for each scenario
    % Scale losses based on current
    P_scale = (I_scenario / Oper.I_peak)^2;
    P_scenario = P_total_avg * P_scale;
    
    % Estimate junction temperature
    Rth_total = Thermal.Rth_jc + Thermal.Rth_ch + Thermal.Rth_ha;
    Tj_mean_scenario = T_amb_scenario + P_scenario * Rth_total;
    
    % Estimate temperature swing (empirical)
    dTj_scenario = dTj * (I_scenario / Oper.I_peak) * m_scenario;
    
    % Calculate lifetime
    Tj_mean_K_scenario = Tj_mean_scenario + 273.15;
    Nf_scenario = PowerCycle.A * (dTj_scenario^PowerCycle.B) * ...
                  exp(PowerCycle.Ea / (PowerCycle.kB * Tj_mean_K_scenario));
    
    lifetime_years = Nf_scenario / Sim.cycles_per_year;
    
    reliability_results(s, :) = [Tj_mean_scenario, dTj_scenario, Nf_scenario, lifetime_years];
    
    fprintf('\n%s:\n', scenario_name);
    fprintf('  Mean Tj: %.1f °C, dTj: %.1f K\n', Tj_mean_scenario, dTj_scenario);
    fprintf('  Lifetime: %.2e cycles (%.1f years)\n', Nf_scenario, lifetime_years);
end

%% 13. Plot Results
figure('Position', [100 100 1200 800]);

% Subplot 1: Losses over time
subplot(3,2,1);
plot(time, P_cond_IGBT, 'LineWidth', 1.5); hold on;
plot(time, P_sw_IGBT, 'LineWidth', 1.5);
plot(time, P_cond_IGBT + P_sw_IGBT, 'k--', 'LineWidth', 2);
xlabel('Time [s]'); ylabel('Power [W]');
title('IGBT Losses');
legend('Conduction', 'Switching', 'Total', 'Location', 'best');
grid on;

% Subplot 2: Junction temperature
subplot(3,2,2);
plot(time, Tj_instant, 'r', 'LineWidth', 1.5);
xlabel('Time [s]'); ylabel('Temperature [°C]');
title('Junction Temperature');
grid on;
yline(IGBT.Tj_max, 'r--', 'Tj_{max}');
yline(Tj_mean, 'b--', 'Tj_{mean}');

% Subplot 3: Loss breakdown pie chart
subplot(3,2,3);
losses = [P_cond_IGBT_avg, P_sw_IGBT_avg, P_cond_Diode_avg, P_sw_Diode_avg];
labels = {'IGBT Cond', 'IGBT Sw', 'Diode Cond', 'Diode Sw'};
pie(losses, labels);
title('Average Loss Distribution');

% Subplot 4: Reliability vs operating conditions
subplot(3,2,4);
bar(reliability_results(:, 4));
set(gca, 'XTickLabel', scenarios(:,1), 'XTickLabelRotation', 45);
ylabel('Lifetime [years]');
title('Lifetime Prediction for Different Scenarios');
grid on;

% Subplot 5: Temperature swing vs lifetime
subplot(3,2,5);
dTj_range = 20:5:100;
Nf_range = PowerCycle.A * (dTj_range.^PowerCycle.B) * ...
           exp(PowerCycle.Ea / (PowerCycle.kB * (Tj_mean + 273.15)));
semilogy(dTj_range, Nf_range/1e6, 'LineWidth', 2);
xlabel('\DeltaT_j [K]'); ylabel('Cycles to Failure [×10^6]');
title('Power Cycling Capability (Semikron Model)');
grid on;
hold on;
plot(dTj, Nf/1e6, 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
legend('Power Cycling Curve', 'Operating Point', 'Location', 'best');

% Subplot 6: Current and temperature profile (zoomed)
subplot(3,2,6);
t_zoom = time(1:10000);
I_zoom = I_load(1:10000);
Tj_zoom = Tj_instant(1:10000);
yyaxis left
plot(t_zoom, I_zoom, 'LineWidth', 1.5);
ylabel('Current [A]');
yyaxis right
plot(t_zoom, Tj_zoom, 'r', 'LineWidth', 1.5);
ylabel('Junction Temp [°C]');
xlabel('Time [s]');
title('Operating Point Detail (First 0.1s)');
grid on;

sgtitle('IGBT Loss and Reliability Analysis', 'FontSize', 14, 'FontWeight', 'bold');

%% 14. Save Results
results.IGBT = IGBT;
results.Thermal = Thermal;
results.Operating = Oper;
results.time = time;
results.losses.P_cond_IGBT = P_cond_IGBT;
results.losses.P_sw_IGBT = P_sw_IGBT;
results.losses.P_cond_Diode = P_cond_Diode;
results.losses.P_sw_Diode = P_sw_Diode;
results.temperature.Tj = Tj_instant;
results.temperature.Tj_mean = Tj_mean;
results.temperature.dTj = dTj;
results.reliability.Nf = Nf;
results.reliability.lifetime_years = Nf/Sim.cycles_per_year;
results.scenarios = reliability_results;

save('IGBT_Loss_Reliability_Results.mat', 'results');
fprintf('\n\nResults saved to IGBT_Loss_Reliability_Results.mat\n');