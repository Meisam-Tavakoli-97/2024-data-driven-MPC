from casadi import *
import numpy as np
import scipy as sp
import time
import signal
import matplotlib.pyplot as plt

# Configuration and setup
signal.signal(signal.SIGINT, signal.SIG_DFL)  # Allow ctrl-c interrupt

# Import modules
from four_tanks import *
from Data_Driven_MPC import generate_DDMPC_robust_solver as build_controller
from History_Data import *
from Hankel_matrix import *
from system_id import shift_and_replace

# Global configuration switches
SAVE_PLOTS = False
DEBUG_MODE = False
if SAVE_PLOTS:
    algorithm_name = 'DDMPC'
    import plot_save

class SystemConfig:
    """Configuration class for system parameters"""
    def __init__(self):
        # Data collection parameters
        self.num_identification_samples = 420
        self.measurement_noise_bound = 1.8e-3
        
        # Noise settings
        self.input_disturbance_enabled = True
        self.input_disturbance_level = 1.2e0
        self.output_disturbance_enabled = False
        self.output_disturbance_level = self.measurement_noise_bound
        
        # Control algorithm parameters
        self.initial_window_length = n
        self.forecast_horizon = 28
        self.actuation_steps = n
        self.time_step = 1 / self.actuation_steps
        self.visualization_frequency = 18
        self.mpc_cycles = 180
        
        # Regularization parameters
        self.sigma_regularization = 1.2e3
        self.trajectory_regularization = 0.12 / self.measurement_noise_bound
        
        # Constraints and bounds
        self.actuator_limits = [9.5] * m
        self.sensor_limits = [np.inf] * p
        
        # Cost function weights
        self.output_penalty_matrix = 2.8 * np.eye(p)
        self.input_penalty_matrix = 1.2e-4 * np.eye(m)
        
        # Solver configuration
        self.use_constraint_regularization = False
        self.use_input_rate_penalty = False
        self.input_rate_weight = 0.0
        self.solver_type = 'osqp'
        self.verbose_solver = False
        self.compile_functions = True

# Initialize system configuration
config = SystemConfig()

# System dynamics initialization
print("Setting up four-tank system dynamics...")
dynamics_discrete, measurement_func, states, inputs, outputs = tanks_dyn()

# Equilibrium computation section
steady_state_matrix = vertcat(
    horzcat(DM.eye(n) - A, -B), 
    horzcat(C, D)
)
target_conditions = DM([0, 0, 0, 0, 0.65, 0.77])
steady_state_solution = solve(steady_state_matrix, target_conditions)

# Extract reference values
reference_inputs = steady_state_solution[-2:].full()
reference_outputs = steady_state_solution[0:2].full()
reference_states = steady_state_solution[:-2].full()

# Initial condition setup
init_conditions_rhs = DM(vertcat(DM.zeros(n), DM([0.4, 0.4])))
init_steady_state = solve(steady_state_matrix, init_conditions_rhs)
starting_state = init_steady_state[:-2].full()
equilibrium_input = init_steady_state[-2:].full()

print(f"Reference inputs: {reference_inputs.T}")
print(f"Reference outputs: {reference_outputs.T}")

# Historical data generation setup
data_start_point = np.zeros((n,))
trajectory_controller = None
reference_path = np.zeros((config.num_identification_samples, n))

# Data collection phase
historical_inputs, historical_outputs = collect_system_identification_data(
    m, p, config.num_identification_samples,
    dynamics_discrete, measurement_func, data_start_point,
    config.input_disturbance_enabled, config.input_disturbance_level,
    config.output_disturbance_enabled, config.output_disturbance_level,
    trajectory_controller, reference_path,
    config.actuator_limits, config.time_step
)

# Data visualization (optional)
if DEBUG_MODE:
    plt.figure("Historical Outputs")
    plt.plot(historical_outputs)
    plt.grid(True)
    output_labels = [str(outputs[i]) for i in range(p)]
    plt.legend(output_labels)
    
    plt.figure("Historical Inputs") 
    plt.plot(historical_inputs)
    plt.grid(True)
    input_labels = [str(inputs[i]) for i in range(m)]
    plt.legend(input_labels)
    
    # Statistical analysis
    print(f"Input statistics - Mean: {np.mean(historical_inputs, axis=0)}, Std: {np.std(historical_inputs, axis=0)}")
    
    plt.figure("Signal Autocorrelations")
    for idx in range(m):
        autocorr = sp.signal.correlate(historical_inputs[:, idx], historical_inputs[:, idx])
        plt.plot(autocorr, label=str(inputs[idx]))
    plt.legend()
    plt.show()

# Plot saving functionality
if SAVE_PLOTS:
    output_fig = plot_save.figure()
    plt.title(r"Historical output data $y_d$")
    plt.plot(historical_outputs)
    plt.grid(True)
    output_labels = [r'$y_{' + str(i) + '}$' for i in range(p)]
    plt.legend(output_labels)
    plot_save.savefig("plots/y_d_four_tanks_" + algorithm_name)
    
    input_fig = plot_save.figure()
    plt.title(r"Historical input data $u_d$")
    plt.plot(historical_inputs)
    plt.grid(True)
    input_labels = [r'$u_{' + str(i) + '}$' for i in range(m)]
    plt.legend(input_labels)
    plot_save.savefig("plots/u_d_four_tanks_" + algorithm_name)
    
    autocorr_fig = plot_save.figure()
    plt.title(r"Input signal autocorrelations $u_d$")
    for idx in range(m):
        autocorr = sp.signal.correlate(historical_inputs[:, idx], historical_inputs[:, idx])
        plt.plot(autocorr, label=r'$u_{' + str(idx) + '}$')
    plt.legend()
    plot_save.savefig("plots/autocorr_u_d_four_tanks_" + algorithm_name)

# Hankel matrix construction
print("Constructing Hankel matrices...")
H_combined, U_past, U_future, Y_past, Y_future = construct_hankel_data_matrices(
    n, config.initial_window_length, config.forecast_horizon,
    historical_inputs, historical_outputs
)

# Controller construction
print("Building MPC controller...")
mpc_solver, substitute_parameters, extract_solution, extract_trajectories = build_controller(
    n, m, p, config.num_identification_samples,
    config.initial_window_length, config.forecast_horizon,
    config.use_constraint_regularization, config.use_input_rate_penalty,
    config.input_rate_weight, config.solver_type,
    config.verbose_solver, config.compile_functions
)

# Control execution setup
class ControllerState:
    """Class to manage controller state variables"""
    def __init__(self):
        # Initialize measurement windows
        self.input_window = np.zeros((m * config.initial_window_length, 1))
        self.output_window = np.zeros((p * config.initial_window_length, 1))
        
        # Initialize guess vectors
        self.g_estimate = np.zeros((config.num_identification_samples - 
                                  (config.initial_window_length + config.forecast_horizon) + 1, 1))
        self.sigma_estimate = np.zeros((p * (config.forecast_horizon + config.initial_window_length), 1))
        self.input_trajectory_guess = np.zeros((m * (config.forecast_horizon + config.initial_window_length), 1))
        self.output_trajectory_guess = np.zeros((p * (config.forecast_horizon + config.initial_window_length), 1))
        
        # Setup optimization parameters
        self.input_bounds_lower = -DM(config.actuator_limits)
        self.input_bounds_upper = DM(config.actuator_limits)
        self.output_bounds_lower = -DM(config.sensor_limits)
        self.output_bounds_upper = DM(config.sensor_limits)
        self.reference_input_param = DM(reference_inputs)
        self.reference_output_param = DM(reference_outputs)
        self.output_weights = DM(config.output_penalty_matrix).reshape((-1, 1))
        self.input_weights = DM(config.input_penalty_matrix).reshape((-1, 1))
        self.g_regularization = DM(config.trajectory_regularization)
        self.noise_bound = DM(config.measurement_noise_bound)
        self.sigma_regularization = DM(config.sigma_regularization)
        self.hankel_matrix = DM(H_combined).reshape((-1, 1))
        
        # Result storage
        total_steps = 1 + config.actuation_steps * config.mpc_cycles
        self.input_history = np.zeros((total_steps, m))
        self.output_history = np.zeros((total_steps, p))
        self.computation_time = 0.0

# Initialize controller
controller_state = ControllerState()

# Initial measurement collection
print("Collecting initial measurements...")
if config.output_disturbance_enabled:
    measurement_noise = np.random.uniform(
        -config.output_disturbance_level, 
        config.output_disturbance_level,
        (p * config.initial_window_length, 1)
    )
else:
    measurement_noise = np.zeros((p * config.initial_window_length, 1))

current_state_vector = np.array(starting_state)
for step in range(config.initial_window_length):
    controller_state.input_window[m*step:m*(step+1)] = equilibrium_input
    measured_output = measurement_func(current_state_vector, 
                                     controller_state.input_window[m*step:m*(step+1)]).full()
    controller_state.output_window[p*step:p*(step+1)] = measured_output + measurement_noise[p*step:p*(step+1)]
    current_state_vector = dynamics_discrete(current_state_vector, 
                                           controller_state.input_window[m*step:m*(step+1)]).full()

# Store initial conditions
controller_state.input_history[0] = controller_state.input_window[-m:].squeeze()
controller_state.output_history[0] = controller_state.output_window[-p:].squeeze()

# Main control loop
print("Starting MPC control loop...")
step_counter = 0
for iteration in range(0, config.actuation_steps * config.mpc_cycles, config.actuation_steps):
    iteration_start = time.perf_counter()
    
    # Parameter setup for current iteration
    current_input_window = DM(controller_state.input_window)
    current_output_window = DM(controller_state.output_window)
    
    # Warm start handling
    if iteration == 0:
        g_initial = DM(controller_state.g_estimate)
        sigma_initial = DM(controller_state.sigma_estimate)
        input_traj_initial = DM(controller_state.input_trajectory_guess)
        output_traj_initial = DM(controller_state.output_trajectory_guess)
    else:
        g_initial = optimal_g_solution
        sigma_initial = optimal_sigma_solution
        input_traj_initial = optimal_input_trajectory
        output_traj_initial = optimal_output_trajectory
    
    # Construct parameter vector for optimization
    parameter_vector = vertcat(
        controller_state.input_bounds_lower,
        controller_state.input_bounds_upper,
        controller_state.output_bounds_lower,
        controller_state.output_bounds_upper,
        controller_state.reference_input_param,
        controller_state.reference_output_param,
        controller_state.output_weights,
        controller_state.input_weights,
        controller_state.g_regularization,
        controller_state.noise_bound,
        controller_state.sigma_regularization,
        current_input_window,
        current_output_window,
        controller_state.hankel_matrix,
        g_initial,
        sigma_initial,
        input_traj_initial,
        output_traj_initial
    )
    
    # Prepare optimization problem
    initial_variables, lower_bounds, upper_bounds, constraint_lower, constraint_upper = substitute_parameters(parameter_vector)
    
    if iteration == 0:
        dual_states_initial = DM.zeros(initial_variables.shape)
        dual_constraints_initial = DM.zeros(constraint_lower.shape)
    
    # Solve optimization problem
    solution = mpc_solver(
        x0=initial_variables,
        lbx=lower_bounds,
        ubx=upper_bounds,
        lbg=constraint_lower,
        ubg=constraint_upper,
        p=parameter_vector,
        lam_x0=dual_states_initial,
        lam_g0=dual_constraints_initial
    )
    
    # Extract results
    optimal_variables = solution['x']
    dual_states_initial = solution['lam_x']
    dual_constraints_initial = solution['lam_g']
    
    optimal_g_solution, optimal_sigma_solution, optimal_input_trajectory, optimal_output_trajectory, cost_value = extract_solution(parameter_vector, optimal_variables)
    input_trajectory_plot, output_trajectory_plot = extract_trajectories(parameter_vector, optimal_variables)
    
    # Record computation time
    iteration_end = time.perf_counter()
    controller_state.computation_time += iteration_end - iteration_start
    
    # Apply control actions
    next_inputs = input_trajectory_plot[config.initial_window_length:(config.initial_window_length + config.actuation_steps), :].full()
    controller_state.input_history[iteration+1:iteration+1+config.actuation_steps] = next_inputs.squeeze()
    
    # Generate measurement noise for this iteration
    if config.output_disturbance_enabled:
        step_noise = np.random.uniform(
            -config.output_disturbance_level,
            config.output_disturbance_level,
            (config.actuation_steps, p)
        )
    else:
        step_noise = np.zeros((config.actuation_steps, p))
    
    # Simulate system for actuation horizon
    for act_step in range(config.actuation_steps):
        # Apply input and get measurement
        controller_state.output_history[iteration+1+act_step] = (measurement_func(current_state_vector, 
                                                                                controller_state.input_history[iteration+1+act_step]).full().squeeze() + 
                                                                step_noise[act_step])
        
        # Update state
        current_state_vector = dynamics_discrete(current_state_vector, 
                                               controller_state.input_history[iteration+1+act_step]).full().squeeze()
        
        # Update measurement windows
        controller_state.input_window = shift_and_replace(
            controller_state.input_window,
            controller_state.input_history[iteration+1+act_step]
        )
        controller_state.output_window = shift_and_replace(
            controller_state.output_window,
            controller_state.output_history[iteration+1+act_step]
        )
        
        # Update trajectory guesses for next iteration
        optimal_input_trajectory = shift_and_replace(
            optimal_input_trajectory,
            controller_state.reference_input_param.full()
        )
        optimal_output_trajectory = shift_and_replace(
            optimal_output_trajectory,
            controller_state.reference_output_param.full()
        )
        
        # Visualization during execution
        if ((iteration + config.actuation_steps) % (config.actuation_steps * config.visualization_frequency) == 0) and (act_step == config.actuation_steps - 1):
            time_axis = np.linspace(0, (iteration + 1 + act_step) * config.time_step, iteration + 1 + act_step)
            
            plt.figure('Input Signals')
            plt.clf()
            for j in range(m):
                plt.subplot(m*100 + 10 + (j+1))
                plt.plot(time_axis, controller_state.input_history[:iteration+1+act_step, j], label=str(inputs[j]))
                plt.plot(time_axis, reference_inputs[j] * np.ones_like(time_axis), '--r', label=str(inputs[j])+'_ref')
                plt.grid(True)
                plt.legend()
            plt.draw()
            plt.pause(1e-3)
            
            plt.figure('Output Measurements')
            plt.clf()
            for j in range(p):
                plt.subplot(p*100 + 10 + (j+1))
                plt.plot(time_axis, controller_state.output_history[:iteration+1+act_step, j], label=str(outputs[j]))
                plt.plot(time_axis, reference_outputs[j] * np.ones_like(time_axis), '--r', label=str(outputs[j])+'_ref')
                plt.grid(True)
                plt.legend()
            plt.draw()
            plt.pause(1e-3)
    
    step_counter += 1
    print(f"Step {step_counter:03d}: Solver time {(iteration_end - iteration_start)*1e3:.3f}ms, Status: {mpc_solver.stats()['success']}, Cost: {cost_value}")

# Performance summary
average_computation_time = controller_state.computation_time / config.mpc_cycles * 1e3
print(f"Average solver execution time: {average_computation_time:.3f} ms")

# Final visualization
time_axis = np.linspace(0, config.actuation_steps * config.mpc_cycles * config.time_step, 
                       1 + config.actuation_steps * config.mpc_cycles)

plt.figure('Final Input Trajectory')
plt.clf()
for j in range(m):
    plt.subplot(m*100 + 10 + (j+1))
    plt.plot(time_axis, controller_state.input_history[:, j], label=str(inputs[j]))
    plt.plot(time_axis, reference_inputs[j] * np.ones_like(time_axis), '--r', label=str(inputs[j])+'_ref')
    plt.grid(True)
    plt.legend()
plt.draw()
plt.pause(1e1)

plt.figure('Final Output Trajectory')
plt.clf()
for j in range(p):
    plt.subplot(p*100 + 10 + (j+1))
    plt.plot(time_axis, controller_state.output_history[:, j], label=str(outputs[j]))
    plt.plot(time_axis, reference_outputs[j] * np.ones_like(time_axis), '--r', label=str(outputs[j])+'_ref')
    plt.grid(True)
    plt.legend()
plt.draw()
plt.pause(1e1)

# Save final plots if enabled
if SAVE_PLOTS:
    input_trajectory_fig, input_axes = plot_save.subplots(nrows=m, ncols=1, sharex=True, ratio=0.8)
    input_axes[0].set_title('Control Input Evolution')
    for j in range(m):
        input_axes[j].plot(time_axis, controller_state.input_history[:, j], label=r'$u_{' + str(j) + '}$')
        input_axes[j].plot(time_axis, reference_inputs[j] * np.ones_like(time_axis), '--r', 
                          label=r'$u_{' + str(j) + ',ref}$')
        input_axes[j].grid(True)
        input_axes[j].legend()
    input_axes[-1].set_xlabel("MPC iterations")
    plot_save.savefig("plots/u_traj_four_tanks_" + algorithm_name)
    
    output_trajectory_fig, output_axes = plot_save.subplots(nrows=p, ncols=1, sharex=True, ratio=0.8)
    output_axes[0].set_title('System Output Evolution')
    for j in range(p):
        output_axes[j].plot(time_axis, controller_state.output_history[:, j], label=r'$y_{' + str(j) + '}$')
        output_axes[j].plot(time_axis, reference_outputs[j] * np.ones_like(time_axis), '--r', 
                           label=r'$y_{' + str(j) + ',ref}$')
        output_axes[j].grid(True)
