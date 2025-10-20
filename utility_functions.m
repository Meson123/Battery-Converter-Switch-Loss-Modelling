%% IGBT Loss and Reliability Model - Utility Functions
% Collection of helper functions for the IGBT modeling project

%% Function 1: Rainflow Counting Algorithm
function [cycles, ranges, means] = rainflow_counting(signal)
    % RAINFLOW_COUNTING Performs rainflow cycle counting on temperature signal
    % Used for accurate fatigue damage calculation
    %
    % Inputs:
    %   signal - Temperature or stress time series
    %
    % Outputs:
    %   cycles - Number of cycles for each range
    %   ranges - Temperature/stress ranges
    %   means  - Mean values for each cycle
    
    % Find peaks and valleys
    [peaks, peak_idx] = findpeaks(signal);
    [valleys, valley_idx] = findpeaks(-signal);
    valleys = -valleys;
    
    % Combine and sort
    extrema = [peaks; valleys];
    extrema_idx = [peak_idx; valley_idx];
    [~, sort_idx] = sort(extrema_idx);
    extrema = extrema(sort_idx);
    
    % Rainflow algorithm
    cycles = [];
    ranges = [];
    means = [];
    stack = [];
    
    for i = 1:length(extrema)
        stack = [stack; extrema(i)];
        
        while length(stack) >= 3
            % Check if middle point forms a cycle
            Y = abs(stack(end) - stack(end-1));
            X = abs(stack(end-1) - stack(end-2));
            
            if length(stack) == 3
                condition = Y >= X;
            else
                Z = abs(stack(end-2) - stack(end-3));
                condition = Y >= X && Y >= Z;
            end
            
            if condition
                % Extract cycle
                range_val = X;
                mean_val = (stack(end-1) + stack(end-2)) / 2;
                
                cycles = [cycles; 1];
                ranges = [ranges; range_val];
                means = [means; mean_val];
                
                % Remove from stack
                stack(end-2) = [];
            else
                break;
            end
        end
    end
    
    % Aggregate similar cycles
    if ~isempty(ranges)
        % Bin ranges
        range_bins = linspace(min(ranges), max(ranges), 20);
        [N, edges, bin] = histcounts(ranges, range_bins);
        
        cycles_agg = N';
        ranges_agg = (edges(1:end-1) + edges(2:end))' / 2;
        means_agg = accumarray(bin, means, [], @mean);
        
        % Remove zero bins
        nonzero = cycles_agg > 0;
        cycles = cycles_agg(nonzero);
        ranges = ranges_agg(nonzero);
        means = means_agg(nonzero);
    end
end

%% Function 2: Calculate Total Losses
function [P_cond, P_sw, P_total] = calculate_losses(V_dc, I_C, Tj, f_sw, params)
    % CALCULATE_LOSSES Computes IGBT conduction and switching losses
    %
    % Inputs:
    %   V_dc   - DC bus voltage [V]
    %   I_C    - Collector current [A]
    %   Tj     - Junction temperature [°C]
    %   f_sw   - Switching frequency [Hz]
    %   params - Structure with IGBT parameters
    
    if nargin < 5
        % Default parameters for IKW40N120H3
        params.V_CE0 = 0.95;
        params.r_CE = 0.028;
        params.E_on_ref = 2.4e-3;
        params.E_off_ref = 1.5e-3;
        params.V_ref = 600;
        params.I_ref = 40;
        params.Tj_ref = 25;
    end
    
    % Temperature correction factors
    temp_factor_r = 1 + 0.005 * (Tj - params.Tj_ref);
    temp_factor_V = 1 + 0.001 * (Tj - params.Tj_ref);
    temp_factor_E = 1 + 0.003 * (Tj - params.Tj_ref);
    
    % Temperature-corrected parameters
    V_CE0_temp = params.V_CE0 * temp_factor_V;
    r_CE_temp = params.r_CE * temp_factor_r;
    
    % Conduction loss (assuming 50% duty cycle)
    I_abs = abs(I_C);
    P_cond = (V_CE0_temp + r_CE_temp * I_abs) * I_abs * 0.5;
    
    % Switching loss
    V_factor = V_dc / params.V_ref;
    I_factor = I_abs / params.I_ref;
    
    % Switching energy with non-linear current dependence
    E_on = params.E_on_ref * V_factor * I_factor^1.3 * temp_factor_E;
    E_off = params.E_off_ref * V_factor * I_factor^0.6 * temp_factor_E;
    
    P_sw = (E_on + E_off) * f_sw;
    
    % Total loss
    P_total = P_cond + P_sw;
end

%% Function 3: Power Cycling Lifetime Calculation
function Nf = calculate_lifetime(dTj, Tj_mean, model_type)
    % CALCULATE_LIFETIME Computes power cycling lifetime
    %
    % Inputs:
    %   dTj       - Temperature swing [K]
    %   Tj_mean   - Mean junction temperature [°C]
    %   model_type - 'semikron', 'cips2008', or 'lesit'
    
    if nargin < 3
        model_type = 'semikron';
    end
    
    % Convert to Kelvin
    Tj_mean_K = Tj_mean + 273.15;
    
    % Boltzmann constant
    kB = 8.617e-5;  % eV/K
    
    switch lower(model_type)
        case 'semikron'
            % Semikron AN 21-001 model (most conservative)
            A = 9.34e14;
            B = -4.416;
            Ea = 0.129;  % eV
            
        case 'cips2008'
            % CIPS 2008 model (Bayerer)
            A = 3.0e14;
            B = -5.039;
            Ea = 0.138;
            
        case 'lesit'
            % LESIT model
            A = 1.0e6;
            B = -4.0;
            Ea = 0.12;
            
        otherwise
            error('Unknown model type: %s', model_type);
    end
    
    % Ensure valid inputs
    if dTj < 1
        dTj = 1;
    end
    
    % Calculate lifetime
    Nf = A * (dTj^B) * exp(Ea / (kB * Tj_mean_K));
    
    % Sanity check
    if Nf < 0 || isnan(Nf) || isinf(Nf)
        warning('Invalid lifetime calculated, using default');
        Nf = 1e10;
    end
end

%% Function 4: Thermal Impedance Calculator
function [Zth, tau] = calculate_thermal_impedance(t, R_th, C_th)
    % CALCULATE_THERMAL_IMPEDANCE Computes transient thermal impedance
    %
    % Inputs:
    %   t    - Time vector [s]
    %   R_th - Thermal resistance vector [K/W]
    %   C_th - Thermal capacitance vector [J/K]
    %
    % Outputs:
    %   Zth - Thermal impedance vs time [K/W]
    %   tau - Time constants [s]
    
    % Number of RC stages
    n_stages = length(R_th);
    
    if length(C_th) ~= n_stages
        error('R_th and C_th must have same length');
    end
    
    % Time constants
    tau = R_th .* C_th;
    
    % Calculate impedance
    Zth = zeros(size(t));
    for i = 1:n_stages
        Zth = Zth + R_th(i) * (1 - exp(-t / tau(i)));
    end
end

%% Function 5: Mission Profile Generator
function [time, I_load, P_mech] = generate_mission_profile(cycle_type, duration)
    % GENERATE_MISSION_PROFILE Creates current profile for EV driving cycle
    %
    % Inputs:
    %   cycle_type - 'WLTP', 'NEDC', 'US06', 'custom'
    %   duration   - Total duration [s]
    %
    % Outputs:
    %   time   - Time vector [s]
    %   I_load - Current profile [A]
    %   P_mech - Mechanical power profile [kW]
    
    dt = 0.1;  % 100ms sample time
    time = 0:dt:duration;
    n_points = length(time);
    
    switch lower(cycle_type)
        case 'wltp'
            % WLTP Class 3 cycle (simplified)
            % Phase 1: Low (0-589s)
            % Phase 2: Medium (589-1022s)
            % Phase 3: High (1022-1477s)
            % Phase 4: Extra High (1477-1800s)
            
            phase_bounds = [0, 589, 1022, 1477, 1800];
            phase_currents = [50, 100, 150, 200];  % RMS currents [A]
            
            I_load = zeros(size(time));
            for i = 1:4
                mask = time >= phase_bounds(i) & time < phase_bounds(i+1);
                % Add sinusoidal variation
                I_load(mask) = phase_currents(i) * ...
                    abs(sin(2*pi*50*time(mask))) .* ...
                    (0.8 + 0.2*rand(sum(mask), 1)');
            end
            
            % Repeat if duration > 1800s
            if duration > 1800
                I_load = repmat(I_load(1:18000), 1, ceil(duration/1800));
                I_load = I_load(1:n_points);
            end
            
        case 'nedc'
            % NEDC cycle
            I_peak = 120;
            I_load = I_peak * (0.3 + 0.4*sin(2*pi*0.01*time)) .* ...
                     abs(sin(2*pi*50*time));
            
        case 'us06'
            % US06 aggressive driving
            I_peak = 250;
            I_load = I_peak * abs(sin(2*pi*50*time)) .* ...
                     (0.6 + 0.4*sin(2*pi*0.02*time));
            
        case 'custom'
            % Custom profile: acceleration, cruise, deceleration
            t_accel = duration * 0.3;
            t_cruise = duration * 0.4;
            t_decel = duration * 0.3;
            
            I_max = 200;
            I_cruise = 100;
            
            I_load = zeros(size(time));
            I_load(time <= t_accel) = I_max * (time(time <= t_accel) / t_accel);
            I_load(time > t_accel & time <= t_accel+t_cruise) = I_cruise;
            I_load(time > t_accel+t_cruise) = I_cruise * ...
                (1 - (time(time > t_accel+t_cruise) - t_accel - t_cruise) / t_decel);
            
            % Add fundamental frequency
            I_load = I_load .* abs(sin(2*pi*50*time));
            
        otherwise
            error('Unknown cycle type: %s', cycle_type);
    end
    
    % Calculate mechanical power (simplified)
    % Assume: P = V * I * cos(phi) * efficiency
    V_battery = 400;  % V
    cos_phi = 0.95;
    eta = 0.93;
    P_mech = V_battery * I_load * cos_phi * eta / 1000;  % kW
end

%% Function 6: Parameter Extraction from Datasheet
function params = extract_IGBT_params(datasheet_file)
    % EXTRACT_IGBT_PARAMS Extracts parameters from datasheet curves
    %
    % Input:
    %   datasheet_file - Path to file with datasheet data
    %
    % Output:
    %   params - Structure with all IGBT parameters
    
    % This is a template - actual implementation depends on data format
    
    % Example: Extract V_CE vs I_C curve
    % Load data
    if exist(datasheet_file, 'file')
        data = readmatrix(datasheet_file);
        I_C = data(:, 1);
        V_CE = data(:, 2);
        
        % Linear regression
        p = polyfit(I_C, V_CE, 1);
        params.r_CE = p(1);
        params.V_CE0 = p(2);
    else
        % Default values for IKW40N120H3
        params.V_CE0 = 0.95;
        params.r_CE = 0.028;
        params.E_on_ref = 2.4e-3;
        params.E_off_ref = 1.5e-3;
        params.E_rec_ref = 0.8e-3;
        params.V_ref = 600;
        params.I_ref = 40;
        params.Tj_ref = 25;
        params.Tj_max = 175;
        params.V_rated = 1200;
        params.I_rated = 40;
        params.R_th_jc = 0.48;
        params.R_th_ch = 0.10;
        params.C_th_j = 0.05;
        
        warning('Using default parameters for IKW40N120H3');
    end
end

%% Function 7: Damage Accumulation (Miner's Rule)
function [damage_total, damage_per_cycle] = miner_damage(cycle_data)
    % MINER_DAMAGE Calculates cumulative damage using Miner's rule
    %
    % Input:
    %   cycle_data - Matrix [dTj, Tj_mean, n_cycles]
    %
    % Outputs:
    %   damage_total     - Total accumulated damage
    %   damage_per_cycle - Damage for each cycle type
    
    n_cycle_types = size(cycle_data, 1);
    damage_per_cycle = zeros(n_cycle_types, 1);
    
    for i = 1:n_cycle_types
        dTj = cycle_data(i, 1);
        Tj_mean = cycle_data(i, 2);
        n_cycles = cycle_data(i, 3);
        
        % Calculate cycles to failure for this condition
        Nf = calculate_lifetime(dTj, Tj_mean);
        
        % Damage = n / Nf
        damage_per_cycle(i) = n_cycles / Nf;
    end
    
    % Total damage (Miner's rule)
    damage_total = sum(damage_per_cycle);
end

%% Function 8: Reliability Function (Weibull)
function R = weibull_reliability(t, Nf, beta)
    % WEIBULL_RELIABILITY Calculates reliability using Weibull distribution
    %
    % Inputs:
    %   t    - Time or cycles
    %   Nf   - Characteristic life (scale parameter)
    %   beta - Shape parameter (typically 2-3 for electronics)
    %
    % Output:
    %   R - Reliability (probability of survival)
    
    if nargin < 3
        beta = 2.5;  % Default for power electronics
    end
    
    eta = Nf / gamma(1 + 1/beta);  % Scale parameter
    
    R = exp(-(t/eta).^beta);
end

%% Function 9: Thermal Transient Response
function [t, Tj] = thermal_step_response(P_step, params, t_end)
    % THERMAL_STEP_RESPONSE Simulates thermal response to power step
    %
    % Inputs:
    %   P_step - Power step [W]
    %   params - Thermal parameters structure
    %   t_end  - End time [s]
    
    if nargin < 3
        t_end = 100;
    end
    
    % Time vector
    t = linspace(0, t_end, 1000);
    
    % Multi-stage Foster network
    R_th = [params.R_th_jc, 0.10, 0.50];  % K/W
    C_th = [params.C_th_j, 0.50, 2.0];    % J/K
    tau = R_th .* C_th;
    
    % Calculate temperature rise
    dT = zeros(size(t));
    for i = 1:length(R_th)
        dT = dT + P_step * R_th(i) * (1 - exp(-t/tau(i)));
    end
    
    % Add ambient temperature
    T_amb = 40;
    Tj = T_amb + dT;
end

%% Function 10: Optimal Switching Frequency
function f_sw_opt = optimize_switching_frequency(V_dc, I_rms, constraints)
    % OPTIMIZE_SWITCHING_FREQUENCY Finds optimal switching frequency
    % Minimizes total losses while meeting constraints
    %
    % Inputs:
    %   V_dc        - DC bus voltage [V]
    %   I_rms       - RMS current [A]
    %   constraints - Structure with Tj_max, THD_max, etc.
    
    % Objective function: minimize losses
    objective = @(f_sw) loss_objective(f_sw, V_dc, I_rms);
    
    % Constraints
    if nargin < 3
        constraints.Tj_max = 150;
        constraints.THD_max = 0.05;
    end
    
    % Bounds
    f_min = 5e3;   % 5 kHz
    f_max = 30e3;  # 30 kHz
    
    % Optimization
    options = optimoptions('fmincon', 'Display', 'off');
    f_sw_opt = fmincon(objective, 10e3, [], [], [], [], f_min, f_max, ...
                       @(f) nonlinear_constraints(f, V_dc, I_rms, constraints), ...
                       options);
    
    fprintf('Optimal switching frequency: %.1f kHz\n', f_sw_opt/1e3);
    
    % Nested functions
    function L = loss_objective(f_sw, V, I)
        Tj = 100;  % Assumed
        [~, P_sw, P_tot] = calculate_losses(V, I, Tj, f_sw);
        L = P_tot;
    end
    
    function [c, ceq] = nonlinear_constraints(f_sw, V, I, cnst)
        Tj = 100;
        [~, ~, P_tot] = calculate_losses(V, I, Tj, f_sw);
        
        % Thermal constraint
        R_th = 1.08;  % K/W
        T_amb = 40;
        Tj_calc = T_amb + P_tot * R_th;
        
        c(1) = Tj_calc - cnst.Tj_max;  % Tj < Tj_max
        
        % THD constraint (simplified)
        THD = 1 / (f_sw / 1000);  % Rough approximation
        c(2) = THD - cnst.THD_max;
        
        ceq = [];  % No equality constraints
    end
end

%% Function 11: Multi-Mission Profile Analysis
function results = analyze_multi_mission(mission_profiles)
    % ANALYZE_MULTI_MISSION Analyzes reliability over multiple missions
    %
    % Input:
    %   mission_profiles - Cell array of mission structures
    %                      Each with fields: name, I_load, duration, weight
    %
    % Output:
    %   results - Structure with reliability metrics
    
    n_missions = length(mission_profiles);
    results.mission_names = cell(n_missions, 1);
    results.cycles_to_failure = zeros(n_missions, 1);
    results.damage_per_mission = zeros(n_missions, 1);
    results.mean_Tj = zeros(n_missions, 1);
    results.max_Tj = zeros(n_missions, 1);
    results.dTj = zeros(n_missions, 1);
    
    for i = 1:n_missions
        mission = mission_profiles{i};
        results.mission_names{i} = mission.name;
        
        % Simulate this mission
        [Tj_profile, P_loss] = simulate_mission(mission.I_load, mission.duration);
        
        % Temperature statistics
        results.mean_Tj(i) = mean(Tj_profile);
        results.max_Tj(i) = max(Tj_profile);
        results.dTj(i) = max(Tj_profile) - min(Tj_profile);
        
        % Lifetime for this mission
        results.cycles_to_failure(i) = calculate_lifetime(results.dTj(i), ...
                                                          results.mean_Tj(i));
        
        % Damage per mission
        results.damage_per_mission(i) = mission.weight / results.cycles_to_failure(i);
    end
    
    % Total damage rate
    results.total_damage_rate = sum(results.damage_per_mission);
    
    % Time to failure
    results.time_to_failure_years = 1 / results.total_damage_rate / 365;
    
    % Print summary
    fprintf('\n=== Multi-Mission Reliability Analysis ===\n');
    for i = 1:n_missions
        fprintf('%s:\n', results.mission_names{i});
        fprintf('  Mean Tj: %.1f°C, dTj: %.1f K\n', ...
                results.mean_Tj(i), results.dTj(i));
        fprintf('  Cycles to failure: %.2e\n', results.cycles_to_failure(i));
        fprintf('  Damage per mission: %.2e\n\n', results.damage_per_mission(i));
    end
    fprintf('Predicted lifetime: %.1f years\n', results.time_to_failure_years);
end

%% Function 12: Simulate Single Mission
function [Tj_profile, P_loss] = simulate_mission(I_load, duration)
    % SIMULATE_MISSION Simulates temperature for a current profile
    %
    % Inputs:
    %   I_load   - Current profile [A]
    %   duration - Duration [s]
    %
    % Outputs:
    %   Tj_profile - Junction temperature vs time [°C]
    %   P_loss     - Power loss vs time [W]
    
    % Operating conditions
    V_dc = 400;   % V
    f_sw = 10e3;  % Hz
    T_amb = 40;   % °C
    
    % Thermal parameters
    R_th = 1.08;  % K/W total
    tau_th = 10;  % s thermal time constant
    
    % Initialize
    n_points = length(I_load);
    Tj_profile = zeros(n_points, 1);
    P_loss = zeros(n_points, 1);
    Tj_profile(1) = T_amb;
    
    dt = duration / n_points;
    
    % Simulation loop
    for i = 2:n_points
        % Calculate losses at previous temperature
        [~, ~, P_loss(i)] = calculate_losses(V_dc, I_load(i), Tj_profile(i-1), f_sw);
        
        % Update temperature (first-order thermal model)
        Tj_steady = T_amb + P_loss(i) * R_th;
        dTj_dt = (Tj_steady - Tj_profile(i-1)) / tau_th;
        Tj_profile(i) = Tj_profile(i-1) + dTj_dt * dt;
    end
end

%% Function 13: Export Results to Report
function export_results_to_report(results, filename)
    % EXPORT_RESULTS_TO_REPORT Generates PDF report with results
    %
    % Inputs:
    %   results  - Results structure from simulation
    %   filename - Output filename (without extension)
    
    % Create figure for report
    fig = figure('Visible', 'off', 'Position', [100 100 1200 1600]);
    
    % Plot 1: Temperature profile
    subplot(4,2,1);
    plot(results.time/60, results.Tj, 'LineWidth', 1.5);
    xlabel('Time [min]'); ylabel('T_j [°C]');
    title('Junction Temperature Profile');
    grid on;
    
    % Plot 2: Loss breakdown
    subplot(4,2,2);
    pie([mean(results.P_cond_IGBT), mean(results.P_sw_IGBT), ...
         mean(results.P_cond_Diode), mean(results.P_sw_Diode)], ...
        {'IGBT Cond', 'IGBT Sw', 'Diode Cond', 'Diode Sw'});
    title('Loss Distribution');
    
    % Plot 3: Temperature histogram
    subplot(4,2,3);
    histogram(results.Tj, 30);
    xlabel('T_j [°C]'); ylabel('Frequency');
    title('Temperature Distribution');
    grid on;
    
    % Plot 4: Reliability curve
    subplot(4,2,4);
    years = linspace(0, 20, 100);
    R = weibull_reliability(years*365*results.cycles_per_day, results.Nf, 2.5);
    plot(years, R*100, 'LineWidth', 2);
    xlabel('Time [years]'); ylabel('Reliability [%]');
    title('Reliability Prediction');
    grid on;
    
    % Plot 5: Power cycling curve
    subplot(4,2,5);
    dTj_range = 10:5:120;
    Nf_range = arrayfun(@(x) calculate_lifetime(x, results.Tj_mean), dTj_range);
    semilogy(dTj_range, Nf_range/1e6, 'LineWidth', 2);
    hold on;
    plot(results.dTj, results.Nf/1e6, 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
    xlabel('\DeltaT_j [K]'); ylabel('Cycles to Failure [×10^6]');
    title('Power Cycling Capability');
    grid on; legend('Capability', 'Operating Point');
    
    % Plot 6: Loss vs time
    subplot(4,2,6);
    plot(results.time/60, results.P_total, 'LineWidth', 1.5);
    xlabel('Time [min]'); ylabel('Power Loss [W]');
    title('Total Power Loss');
    grid on;
    
    % Text summary
    subplot(4,2,[7,8]);
    axis off;
    summary_text = sprintf([...
        '\\fontsize{12}\\bf Reliability Summary\n\n' ...
        '\\fontsize{10}\\rm' ...
        'Mean Junction Temperature: %.1f °C\n' ...
        'Maximum Junction Temperature: %.1f °C\n' ...
        'Temperature Swing (\\DeltaT_j): %.1f K\n\n' ...
        'Average Power Loss: %.1f W\n' ...
        '  - IGBT Conduction: %.1f W (%.1f%%)\n' ...
        '  - IGBT Switching: %.1f W (%.1f%%)\n' ...
        '  - Diode Losses: %.1f W (%.1f%%)\n\n' ...
        'Reliability Metrics:\n' ...
        '  - Cycles to Failure: %.2e cycles\n' ...
        '  - Predicted Lifetime: %.1f years\n' ...
        '  - B10 Life: %.1f years\n' ...
        '  - B50 Life: %.1f years'], ...
        results.Tj_mean, results.Tj_max, results.dTj, ...
        results.P_total_avg, ...
        results.P_cond_IGBT_avg, results.P_cond_IGBT_avg/results.P_total_avg*100, ...
        results.P_sw_IGBT_avg, results.P_sw_IGBT_avg/results.P_total_avg*100, ...
        results.P_diode_avg, results.P_diode_avg/results.P_total_avg*100, ...
        results.Nf, results.lifetime_years, ...
        results.lifetime_years * 0.5, results.lifetime_years);
    
    text(0.1, 0.5, summary_text, 'Interpreter', 'tex', ...
         'VerticalAlignment', 'middle');
    
    % Save figure
    saveas(fig, [filename '.png']);
    fprintf('Report exported to: %s.png\n', filename);
    
    % Also save data to MAT file
    save([filename '.mat'], 'results');
    fprintf('Data saved to: %s.mat\n', filename);
    
    close(fig);
end

%% Function 14: Real-Time Parameter Update
function params_updated = update_params_online(params, measurements)
    % UPDATE_PARAMS_ONLINE Updates parameters based on measurements
    % For health monitoring and prognostics
    %
    % Inputs:
    %   params       - Current parameter structure
    %   measurements - Structure with V_CE, I_C, Tj measurements
    %
    % Output:
    %   params_updated - Updated parameters
    
    persistent filter_state
    
    if isempty(filter_state)
        % Initialize Kalman filter
        filter_state.R_CE = params.r_CE;
        filter_state.P = 0.001;  % Covariance
        filter_state.Q = 1e-8;   % Process noise
        filter_state.R_meas = 0.01;  % Measurement noise
    end
    
    % Predict
    R_CE_pred = filter_state.R_CE;
    P_pred = filter_state.P + filter_state.Q;
    
    % Measurement update
    V_CE0_temp = params.V_CE0 * (1 + 0.001*(measurements.Tj - 25));
    V_CE_expected = V_CE0_temp + R_CE_pred * measurements.I_C;
    
    innovation = measurements.V_CE - V_CE_expected;
    S = measurements.I_C^2 * P_pred + filter_state.R_meas;
    K = P_pred * measurements.I_C / S;
    
    % Update
    filter_state.R_CE = R_CE_pred + K * innovation / measurements.I_C;
    filter_state.P = (1 - K * measurements.I_C) * P_pred;
    
    % Update parameters
    params_updated = params;
    params_updated.r_CE = filter_state.R_CE;
    
    % Calculate health indicator
    R_CE_increase_percent = (filter_state.R_CE - params.r_CE) / params.r_CE * 100;
    
    if R_CE_increase_percent > 20
        warning('Significant degradation detected: R_CE increased by %.1f%%', ...
                R_CE_increase_percent);
    end
end

%% Main test function
function test_utility_functions()
    % TEST_UTILITY_FUNCTIONS Tests all utility functions
    
    fprintf('Testing IGBT Utility Functions...\n\n');
    
    % Test 1: Rainflow counting
    fprintf('Test 1: Rainflow Counting\n');
    signal = 50 + 20*sin(2*pi*0.1*(0:0.1:100)) + 10*randn(1,1001);
    [cycles, ranges, means] = rainflow_counting(signal);
    fprintf('  Found %d cycle types\n\n', length(cycles));
    
    % Test 2: Loss calculation
    fprintf('Test 2: Loss Calculation\n');
    [P_cond, P_sw, P_tot] = calculate_losses(400, 100, 125, 10e3);
    fprintf('  P_cond = %.1f W, P_sw = %.1f W, P_total = %.1f W\n\n', ...
            P_cond, P_sw, P_tot);
    
    % Test 3: Lifetime calculation
    fprintf('Test 3: Lifetime Calculation\n');
    Nf = calculate_lifetime(60, 100);
    fprintf('  Nf = %.2e cycles (%.1f years at 1000 cycles/year)\n\n', ...
            Nf, Nf/1000);
    
    % Test 4: Mission profile
    fprintf('Test 4: Mission Profile Generation\n');
    [time, I_load, P_mech] = generate_mission_profile('WLTP', 1800);
    fprintf('  Generated %d-second WLTP profile\n', length(time));
    fprintf('  Peak current: %.1f A, Average power: %.1f kW\n\n', ...
            max(I_load), mean(P_mech));
    
    % Test 5: Thermal response
    fprintf('Test 5: Thermal Step Response\n');
    params.R_th_jc = 0.48;
    params.C_th_j = 0.05;
    [t, Tj] = thermal_step_response(200, params, 100);
    fprintf('  Steady-state temperature: %.1f °C\n', Tj(end));
    fprintf('  Time to 63%%: %.2f s\n\n', t(find(Tj >= 0.632*Tj(end), 1)));
    
    fprintf('All tests completed successfully!\n');
end