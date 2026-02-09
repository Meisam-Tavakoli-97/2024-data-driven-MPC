from casadi import *

def build_robust_data_driven_controller(
        system_order,
        num_inputs,
        num_outputs,
        total_data_samples,
        initial_horizon_length,
        prediction_horizon_length,
        enable_regularization_constraints=True,
        enable_input_rate_penalty=False,
        input_rate_penalty_weight=0.0,
        optimization_solver='osqp',
        display_solver_messages=False,
        enable_code_generation=True
):
    """
    Build a robust data-driven MPC controller with measurement uncertainty.
    
    Parameters:
    - system_order: order of the system for terminal constraints
    - num_inputs: number of control inputs
    - num_outputs: number of measured outputs
    - total_data_samples: total number of data samples collected
    - initial_horizon_length: length of initial condition window
    - prediction_horizon_length: MPC prediction horizon length
    - enable_regularization_constraints: flag for regularization constraints
    - enable_input_rate_penalty: flag for input rate penalties
    - input_rate_penalty_weight: weight for input rate penalty
    - optimization_solver: solver choice ('ipopt' or 'osqp')
    - display_solver_messages: flag to show solver output
    - enable_code_generation: flag to enable JIT compilation
    
    Returns:
    - controller_solver: configured MPC solver
    - parameter_substitution_function: function to substitute parameters
    - solution_extraction_function: function to extract optimal solution
    - trajectory_plotting_function: function to extract trajectories for plotting
    """
    
    ### Define symbolic cost function components
    ## Reference values for tracking
    input_reference_sym = SX.sym('u_reference_cost', num_inputs)
    output_reference_sym = SX.sym('y_reference_cost', num_outputs)

    ## Cost weighting matrices
    output_weight_matrix_sym = SX.sym('Q_matrix', num_outputs, num_outputs)
    input_weight_matrix_sym = SX.sym('R_matrix', num_inputs, num_inputs)

    ## Current input-output values
    current_input_sym = SX.sym('current_input', num_inputs)
    current_output_sym = SX.sym('current_output', num_outputs)

    ## Stage cost computation
    stage_cost_sym = ((current_output_sym - output_reference_sym).T @ output_weight_matrix_sym @ 
                     (current_output_sym - output_reference_sym) + 
                     (current_input_sym - input_reference_sym).T @ input_weight_matrix_sym @ 
                     (current_input_sym - input_reference_sym))

    ## Create CasADi cost function
    stage_cost_function = Function(
        'stage_cost_function', 
        [input_reference_sym, output_reference_sym, output_weight_matrix_sym, input_weight_matrix_sym, 
         current_input_sym, current_output_sym], 
        [stage_cost_sym], 
        ['u_ref','y_ref','Q','R','u_current','y_current'],
        ['stage_cost'])

    ### Define optimization problem
    ## Decision variables
    trajectory_weights_var = SX.sym('trajectory_weights', total_data_samples-(initial_horizon_length+prediction_horizon_length)+1)
    measurement_noise_var = SX.sym('measurement_noise', num_outputs*(prediction_horizon_length+initial_horizon_length))
    input_trajectory_var = SX.sym('input_trajectory', num_inputs*(prediction_horizon_length+initial_horizon_length))
    output_trajectory_var = SX.sym('output_trajectory', num_outputs*(prediction_horizon_length+initial_horizon_length))

    ## Problem parameters
    # Input-output constraint bounds
    input_lower_bounds_sym = SX.sym('input_lower_bounds', num_inputs)
    input_upper_bounds_sym = SX.sym('input_upper_bounds', num_inputs)
    output_lower_bounds_sym = SX.sym('output_lower_bounds', num_outputs)
    output_upper_bounds_sym = SX.sym('output_upper_bounds', num_outputs)

    # Reference trajectories
    input_reference_param_sym = SX.sym('input_reference_param', num_inputs)
    output_reference_param_sym = SX.sym('output_reference_param', num_outputs)

    # Cost function parameters
    output_weight_param_sym = SX.sym('output_weight_param', num_outputs, num_outputs)
    input_weight_param_sym = SX.sym('input_weight_param', num_inputs, num_inputs)

    # Regularization parameters
    trajectory_weight_penalty_sym = SX.sym('trajectory_weight_penalty')
    noise_bound_sym = SX.sym('noise_bound')
    noise_penalty_sym = SX.sym('noise_penalty')

    # Initial condition parameters
    initial_input_sequence_sym = SX.sym('initial_input_sequence', num_inputs*initial_horizon_length)
    initial_output_sequence_sym = SX.sym('initial_output_sequence', num_outputs*initial_horizon_length)

    # Data matrix parameter
    data_hankel_matrix_sym = SX.sym('data_hankel_matrix', (num_inputs+num_outputs)*(initial_horizon_length+prediction_horizon_length), 
                                   total_data_samples-(initial_horizon_length+prediction_horizon_length)+1)

    # Initial guesses for warm-starting
    trajectory_weights_initial_sym = SX.sym('trajectory_weights_initial', total_data_samples-(initial_horizon_length+prediction_horizon_length)+1)
    measurement_noise_initial_sym = SX.sym('measurement_noise_initial', num_outputs*(prediction_horizon_length+initial_horizon_length))
    input_trajectory_initial_sym = SX.sym('input_trajectory_initial', num_inputs*(prediction_horizon_length+initial_horizon_length))
    output_trajectory_initial_sym = SX.sym('output_trajectory_initial', num_outputs*(prediction_horizon_length+initial_horizon_length))

    ## Initialize optimization problem structure
    decision_variables = []
    initial_guesses = []
    variable_lower_bounds = []
    variable_upper_bounds = []

    total_cost = DM(0)

    equality_constraints = []
    constraint_lower_bounds = []
    constraint_upper_bounds = []

    parameter_list = []

    # Add trajectory weights to decision variables
    decision_variables.append(trajectory_weights_var)
    variable_lower_bounds.append(repmat(-inf, trajectory_weights_var.shape[0], 1))
    variable_upper_bounds.append(repmat(inf, trajectory_weights_var.shape[0], 1))
    initial_guesses.append(trajectory_weights_initial_sym)

    # Add measurement noise to decision variables
    decision_variables.append(measurement_noise_var)
    variable_lower_bounds.append(repmat(-inf, measurement_noise_var.shape[0], 1))
    variable_upper_bounds.append(repmat(inf, measurement_noise_var.shape[0], 1))
    initial_guesses.append(measurement_noise_initial_sym)

    # Add input trajectory to decision variables
    decision_variables.append(input_trajectory_var)
    variable_lower_bounds.append(repmat(input_lower_bounds_sym, prediction_horizon_length+initial_horizon_length, 1))
    variable_upper_bounds.append(repmat(input_upper_bounds_sym, prediction_horizon_length+initial_horizon_length, 1))
    initial_guesses.append(input_trajectory_initial_sym)

    # Add output trajectory to decision variables
    decision_variables.append(output_trajectory_var)
    variable_lower_bounds.append(repmat(output_lower_bounds_sym, prediction_horizon_length+initial_horizon_length, 1))
    variable_upper_bounds.append(repmat(output_upper_bounds_sym, prediction_horizon_length+initial_horizon_length, 1))
    initial_guesses.append(output_trajectory_initial_sym)

    # Build parameter vector
    parameter_list.extend([
        input_lower_bounds_sym, input_upper_bounds_sym, output_lower_bounds_sym, output_upper_bounds_sym,
        input_reference_param_sym, output_reference_param_sym,
        output_weight_param_sym.reshape((output_weight_param_sym.shape[0]*output_weight_param_sym.shape[1],1)),
        input_weight_param_sym.reshape((input_weight_param_sym.shape[0]*input_weight_param_sym.shape[1],1)),
        trajectory_weight_penalty_sym, noise_bound_sym, noise_penalty_sym,
        initial_input_sequence_sym, initial_output_sequence_sym,
        data_hankel_matrix_sym.reshape((data_hankel_matrix_sym.shape[0]*data_hankel_matrix_sym.shape[1],1)),
        trajectory_weights_initial_sym, measurement_noise_initial_sym, 
        input_trajectory_initial_sym, output_trajectory_initial_sym
    ])

    # Compute total cost function
    for time_step in range(prediction_horizon_length):
        input_slice = input_trajectory_var[num_inputs*(initial_horizon_length+time_step):num_inputs*(initial_horizon_length+time_step+1)]
        output_slice = output_trajectory_var[num_outputs*(initial_horizon_length+time_step):num_outputs*(initial_horizon_length+time_step+1)]
        total_cost += stage_cost_function(input_reference_param_sym, output_reference_param_sym, 
                                         output_weight_param_sym, input_weight_param_sym,
                                         input_slice, output_slice)
        
        # Add input rate penalty if enabled
        if enable_input_rate_penalty and time_step < (prediction_horizon_length-1):
            current_input = input_trajectory_var[num_inputs*time_step:num_inputs*(time_step+1)]
            next_input = input_trajectory_var[num_inputs*(time_step+1):num_inputs*(time_step+2)]
            total_cost += input_rate_penalty_weight*sumsqr(next_input - current_input)
    
    # Add regularization terms to cost
    total_cost += (trajectory_weight_penalty_sym*noise_bound_sym*sumsqr(trajectory_weights_var) + 
                   noise_penalty_sym*sumsqr(measurement_noise_var))

    # Data-driven system identification constraint
    io_trajectory = vertcat(input_trajectory_var, output_trajectory_var + measurement_noise_var)
    equality_constraints.append(io_trajectory - data_hankel_matrix_sym @ trajectory_weights_var)
    constraint_lower_bounds.append(DM.zeros((num_inputs+num_outputs)*(initial_horizon_length+prediction_horizon_length)))
    constraint_upper_bounds.append(DM.zeros((num_inputs+num_outputs)*(initial_horizon_length+prediction_horizon_length)))

    # Initial condition constraint
    initial_trajectory = vertcat(input_trajectory_var[0:num_inputs*initial_horizon_length], 
                                output_trajectory_var[0:num_outputs*initial_horizon_length])
    initial_condition = vertcat(initial_input_sequence_sym, initial_output_sequence_sym)
    equality_constraints.append(initial_trajectory - initial_condition)
    constraint_lower_bounds.append(DM.zeros((num_inputs+num_outputs)*initial_horizon_length))
    constraint_upper_bounds.append(DM.zeros((num_inputs+num_outputs)*initial_horizon_length))

    # Terminal constraint for stability
    terminal_input = input_trajectory_var[-num_inputs*system_order:]
    terminal_output = output_trajectory_var[-num_outputs*system_order:]
    terminal_reference = vertcat(repmat(input_reference_param_sym, system_order, 1), 
                                repmat(output_reference_param_sym, system_order, 1))
    equality_constraints.append(vertcat(terminal_input, terminal_output) - terminal_reference)
    constraint_lower_bounds.append(DM.zeros((num_inputs+num_outputs)*system_order))
    constraint_upper_bounds.append(DM.zeros((num_inputs+num_outputs)*system_order))

    # Robust constraint for measurement noise
    if enable_regularization_constraints:
        for step_idx in range(prediction_horizon_length):
            noise_slice = measurement_noise_var[num_outputs*(initial_horizon_length+step_idx):
                                               num_outputs*(initial_horizon_length+step_idx+1)]
            noise_constraint = norm_inf(noise_slice) - noise_bound_sym*(1+norm_1(trajectory_weights_var))
            equality_constraints.append(noise_constraint)
            constraint_lower_bounds.append(-inf)
            constraint_upper_bounds.append(0)

    # Concatenate all vectors
    decision_variables = vertcat(*decision_variables)
    initial_guesses = vertcat(*initial_guesses)
    variable_lower_bounds = vertcat(*variable_lower_bounds)
    variable_upper_bounds = vertcat(*variable_upper_bounds)

    equality_constraints = vertcat(*equality_constraints)
    constraint_lower_bounds = vertcat(*constraint_lower_bounds)
    constraint_upper_bounds = vertcat(*constraint_upper_bounds)

    parameter_list = vertcat(*parameter_list)

    # Configure and create solver
    print("Building robust data-driven MPC solver...")

    optimization_problem = {'f': total_cost, 'x': decision_variables, 'g': equality_constraints, 'p': parameter_list}

    if optimization_solver == 'ipopt':
        solver_options = {}
        if not display_solver_messages:
            solver_options['print_time'] = False
            solver_options['ipopt.print_level'] = 0
        if enable_code_generation:
            solver_options["compiler"] = "shell"
            solver_options["jit"] = True
            solver_options["jit_options"] = {"compiler": "gcc"}
        
        controller_solver = nlpsol('controller_solver', 'ipopt', optimization_problem, solver_options)
        
    elif optimization_solver == 'osqp':
        solver_options = {}
        solver_options['warm_start_primal'] = True
        solver_options['warm_start_dual'] = True
        if not display_solver_messages:
            solver_options['print_time'] = False
            solver_options['osqp.verbose'] = False
        else:
            solver_options['print_time'] = True
            solver_options['osqp.verbose'] = True
        if enable_code_generation:
            solver_options["compiler"] = "shell"
            solver_options["jit"] = True
            solver_options["jit_options"] = {"compiler": "gcc"}
        
        controller_solver = qpsol('controller_solver', 'osqp', optimization_problem, solver_options)
    else:
        print("Error: Unsupported solver specified. Choose 'ipopt' or 'osqp'.")
        exit()

    print("Robust data-driven MPC solver created successfully.\n")

    # Create utility functions
    parameter_substitution_function = Function('substitute_parameters', [parameter_list], 
                                              [initial_guesses, variable_lower_bounds, variable_upper_bounds, 
                                               constraint_lower_bounds, constraint_upper_bounds])

    solution_extraction_function = Function('extract_solution', [parameter_list, decision_variables], 
                                           [trajectory_weights_var, measurement_noise_var, 
                                            input_trajectory_var, output_trajectory_var, total_cost])

    trajectory_plotting_function = Function('extract_trajectories', [parameter_list, decision_variables], 
                                           [input_trajectory_var.reshape((num_inputs, prediction_horizon_length+initial_horizon_length)).T, 
                                            output_trajectory_var.reshape((num_outputs, prediction_horizon_length+initial_horizon_length)).T])

    return controller_solver, parameter_substitution_function, solution_extraction_function, trajectory_plotting_function

def build_standard_data_driven_controller(
        system_order,
        num_inputs,
        num_outputs,
        total_data_samples,
        initial_horizon_length,
        prediction_horizon_length,
        enable_regularization_constraints=False,
        enable_input_rate_penalty=False,
        input_rate_penalty_weight=0.0,
        optimization_solver='osqp',
        display_solver_messages=False,
        enable_code_generation=True
):
    """
    Build a standard data-driven MPC controller without explicit uncertainty modeling.
    
    Parameters:
    - system_order: order of the system for terminal constraints
    - num_inputs: number of control inputs
    - num_outputs: number of measured outputs
    - total_data_samples: total number of data samples collected
    - initial_horizon_length: length of initial condition window
    - prediction_horizon_length: MPC prediction horizon length
    - enable_regularization_constraints: flag for regularization constraints (unused in standard version)
    - enable_input_rate_penalty: flag for input rate penalties
    - input_rate_penalty_weight: weight for input rate penalty
    - optimization_solver: solver choice ('ipopt' or 'osqp')
    - display_solver_messages: flag to show solver output
    - enable_code_generation: flag to enable JIT compilation
    
    Returns:
    - controller_solver: configured MPC solver
    - parameter_substitution_function: function to substitute parameters
    - solution_extraction_function: function to extract optimal solution
    - trajectory_plotting_function: function to extract trajectories for plotting
    """
    
    ### Define symbolic cost function components
    ## Reference values for tracking
    input_reference_sym = SX.sym('u_reference_cost', num_inputs)
    output_reference_sym = SX.sym('y_reference_cost', num_outputs)

    ## Cost weighting matrices
    output_weight_matrix_sym = SX.sym('Q_matrix', num_outputs, num_outputs)
    input_weight_matrix_sym = SX.sym('R_matrix', num_inputs, num_inputs)

    ## Current input-output values
    current_input_sym = SX.sym('current_input', num_inputs)
    current_output_sym = SX.sym('current_output', num_outputs)

    ## Stage cost computation
    stage_cost_sym = ((current_output_sym - output_reference_sym).T @ output_weight_matrix_sym @ 
                     (current_output_sym - output_reference_sym) + 
                     (current_input_sym - input_reference_sym).T @ input_weight_matrix_sym @ 
                     (current_input_sym - input_reference_sym))

    ## Create CasADi cost function
    stage_cost_function = Function(
        'stage_cost_function', 
        [input_reference_sym, output_reference_sym, output_weight_matrix_sym, input_weight_matrix_sym, 
         current_input_sym, current_output_sym], 
        [stage_cost_sym], 
        ['u_ref','y_ref','Q','R','u_current','y_current'],
        ['stage_cost'])

    ### Define optimization problem
    ## Decision variables
    trajectory_weights_var = SX.sym('trajectory_weights', total_data_samples-(initial_horizon_length+prediction_horizon_length)+1)
    input_trajectory_var = SX.sym('input_trajectory', num_inputs*(prediction_horizon_length+initial_horizon_length))
    output_trajectory_var = SX.sym('output_trajectory', num_outputs*(prediction_horizon_length+initial_horizon_length))

    ## Problem parameters
    # Input-output constraint bounds
    input_lower_bounds_sym = SX.sym('input_lower_bounds', num_inputs)
    input_upper_bounds_sym = SX.sym('input_upper_bounds', num_inputs)
    output_lower_bounds_sym = SX.sym('output_lower_bounds', num_outputs)
    output_upper_bounds_sym = SX.sym('output_upper_bounds', num_outputs)

    # Reference trajectories
    input_reference_param_sym = SX.sym('input_reference_param', num_inputs)
    output_reference_param_sym = SX.sym('output_reference_param', num_outputs)

    # Cost function parameters
    output_weight_param_sym = SX.sym('output_weight_param', num_outputs, num_outputs)
    input_weight_param_sym = SX.sym('input_weight_param', num_inputs, num_inputs)

    # Initial condition parameters
    initial_input_sequence_sym = SX.sym('initial_input_sequence', num_inputs*initial_horizon_length)
    initial_output_sequence_sym = SX.sym('initial_output_sequence', num_outputs*initial_horizon_length)

    # Data matrix parameter
    data_hankel_matrix_sym = SX.sym('data_hankel_matrix', (num_inputs+num_outputs)*(initial_horizon_length+prediction_horizon_length), 
                                   total_data_samples-(initial_horizon_length+prediction_horizon_length)+1)

    # Initial guesses for warm-starting
    trajectory_weights_initial_sym = SX.sym('trajectory_weights_initial', total_data_samples-(initial_horizon_length+prediction_horizon_length)+1)
    input_trajectory_initial_sym = SX.sym('input_trajectory_initial', num_inputs*(prediction_horizon_length+initial_horizon_length))
    output_trajectory_initial_sym = SX.sym('output_trajectory_initial', num_outputs*(prediction_horizon_length+initial_horizon_length))

    ## Initialize optimization problem structure
    decision_variables = []
    initial_guesses = []
    variable_lower_bounds = []
    variable_upper_bounds = []

    total_cost = DM(0)

    equality_constraints = []
    constraint_lower_bounds = []
    constraint_upper_bounds = []

    parameter_list = []

    # Add trajectory weights to decision variables
    decision_variables.append(trajectory_weights_var)
    variable_lower_bounds.append(repmat(-inf, trajectory_weights_var.shape[0], 1))
    variable_upper_bounds.append(repmat(inf, trajectory_weights_var.shape[0], 1))
    initial_guesses.append(trajectory_weights_initial_sym)

    # Add input trajectory to decision variables
    decision_variables.append(input_trajectory_var)
    variable_lower_bounds.append(repmat(input_lower_bounds_sym, prediction_horizon_length+initial_horizon_length, 1))
    variable_upper_bounds.append(repmat(input_upper_bounds_sym, prediction_horizon_length+initial_horizon_length, 1))
    initial_guesses.append(input_trajectory_initial_sym)

    # Add output trajectory to decision variables
    decision_variables.append(output_trajectory_var)
    variable_lower_bounds.append(repmat(output_lower_bounds_sym, prediction_horizon_length+initial_horizon_length, 1))
    variable_upper_bounds.append(repmat(output_upper_bounds_sym, prediction_horizon_length+initial_horizon_length, 1))
    initial_guesses.append(output_trajectory_initial_sym)

    # Build parameter vector (maintain compatibility with robust version)
    parameter_list.extend([
        input_lower_bounds_sym, input_upper_bounds_sym, output_lower_bounds_sym, output_upper_bounds_sym,
        input_reference_param_sym, output_reference_param_sym,
        output_weight_param_sym.reshape((output_weight_param_sym.shape[0]*output_weight_param_sym.shape[1],1)),
        input_weight_param_sym.reshape((input_weight_param_sym.shape[0]*input_weight_param_sym.shape[1],1)),
        SX.sym('trajectory_weight_penalty_placeholder'),  # Placeholder for compatibility
        SX.sym('noise_bound_placeholder'),                # Placeholder for compatibility
        SX.sym('noise_penalty_placeholder'),              # Placeholder for compatibility
        initial_input_sequence_sym, initial_output_sequence_sym,
        data_hankel_matrix_sym.reshape((data_hankel_matrix_sym.shape[0]*data_hankel_matrix_sym.shape[1],1)),
        trajectory_weights_initial_sym,
        SX.sym('measurement_noise_placeholder', num_outputs*(prediction_horizon_length+initial_horizon_length)),  # Placeholder
        input_trajectory_initial_sym, output_trajectory_initial_sym
    ])

    # Compute total cost function
    for time_step in range(prediction_horizon_length):
        input_slice = input_trajectory_var[num_inputs*(initial_horizon_length+time_step):num_inputs*(initial_horizon_length+time_step+1)]
        output_slice = output_trajectory_var[num_outputs*(initial_horizon_length+time_step):num_outputs*(initial_horizon_length+time_step+1)]
        total_cost += stage_cost_function(input_reference_param_sym, output_reference_param_sym, 
                                         output_weight_param_sym, input_weight_param_sym,
                                         input_slice, output_slice)
        
        # Add input rate penalty if enabled
        if enable_input_rate_penalty and time_step < (prediction_horizon_length-1):
            current_input = input_trajectory_var[num_inputs*time_step:num_inputs*(time_step+1)]
            next_input = input_trajectory_var[num_inputs*(time_step+1):num_inputs*(time_step+2)]
            total_cost += input_rate_penalty_weight*sumsqr(next_input - current_input)

    # Data-driven system identification constraint
    io_trajectory = vertcat(input_trajectory_var, output_trajectory_var)
    equality_constraints.append(io_trajectory - data_hankel_matrix_sym @ trajectory_weights_var)
    constraint_lower_bounds.append(DM.zeros((num_inputs+num_outputs)*(initial_horizon_length+prediction_horizon_length)))
    constraint_upper_bounds.append(DM.zeros((num_inputs+num_outputs)*(initial_horizon_length+prediction_horizon_length)))

    # Initial condition constraint
    initial_trajectory = vertcat(input_trajectory_var[0:num_inputs*initial_horizon_length], 
                                output_trajectory_var[0:num_outputs*initial_horizon_length])
    initial_condition = vertcat(initial_input_sequence_sym, initial_output_sequence_sym)
    equality_constraints.append(initial_trajectory - initial_condition)
    constraint_lower_bounds.append(DM.zeros((num_inputs+num_outputs)*initial_horizon_length))
    constraint_upper_bounds.append(DM.zeros((num_inputs+num_outputs)*initial_horizon_length))

    # Terminal constraint for stability
    terminal_input = input_trajectory_var[-num_inputs*system_order:]
    terminal_output = output_trajectory_var[-num_outputs*system_order:]
    terminal_reference = vertcat(repmat(input_reference_param_sym, system_order, 1), 
                                repmat(output_reference_param_sym, system_order, 1))
    equality_constraints.append(vertcat(terminal_input, terminal_output) - terminal_reference)
    constraint_lower_bounds.append(DM.zeros((num_inputs+num_outputs)*system_order))
    constraint_upper_bounds.append(DM.zeros((num_inputs+num_outputs)*system_order))

    # Concatenate all vectors
    decision_variables = vertcat(*decision_variables)
    initial_guesses = vertcat(*initial_guesses)
    variable_lower_bounds = vertcat(*variable_lower_bounds)
    variable_upper_bounds = vertcat(*variable_upper_bounds)

    equality_constraints = vertcat(*equality_constraints)
    constraint_lower_bounds = vertcat(*constraint_lower_bounds)
    constraint_upper_bounds = vertcat(*constraint_upper_bounds)

    parameter_list = vertcat(*parameter_list)

    # Configure and create solver
    print("Building standard data-driven MPC solver...")

    optimization_problem = {'f': total_cost, 'x': decision_variables, 'g': equality_constraints, 'p': parameter_list}

    if optimization_solver == 'ipopt':
        solver_options = {}
        if not display_solver_messages:
            solver_options['print_time'] = False
            solver_options['ipopt.print_level'] = 0
        if enable_code_generation:
            solver_options["compiler"] = "shell"
            solver_options["jit"] = True
            solver_options["jit_options"] = {"compiler": "clang"}
        
        controller_solver = nlpsol('controller_solver', 'ipopt', optimization_problem, solver_options)
        
    elif optimization_solver == 'osqp':
        solver_options = {}
        solver_options['warm_start_primal'] = True
        solver_options['warm_start_dual'] = True
        if not display_solver_messages:
            solver_options['print_time'] = False
            solver_options['osqp.verbose'] = False
        else:
            solver_options['print_time'] = True
            solver_options['osqp.verbose'] = True
        if enable_code_generation:
            solver_options["compiler"] = "shell"
            solver_options["jit"] = True
            solver_options["jit_options"] = {"compiler": "gcc"}
        
        controller_solver = qpsol('controller_solver', 'osqp', optimization_problem, solver_options)
    else:
        print("Error: Unsupported solver specified. Choose 'ipopt' or 'osqp'.")
        exit()

    print("Standard data-driven MPC solver created successfully.\n")

    # Create utility functions (return zero sigma for compatibility with robust version)
    parameter_substitution_function = Function('substitute_parameters', [parameter_list], 
                                              [initial_guesses, variable_lower_bounds, variable_upper_bounds, 
                                               constraint_lower_bounds, constraint_upper_bounds])

    solution_extraction_function = Function('extract_solution', [parameter_list, decision_variables], 
                                           [trajectory_weights_var, DM.zeros(num_outputs*(prediction_horizon_length+initial_horizon_length)), 
                                            input_trajectory_var, output_trajectory_var, total_cost])

    trajectory_plotting_function = Function('extract_trajectories', [parameter_list, decision_variables], 
                                           [input_trajectory_var.reshape((num_inputs, prediction_horizon_length+initial_horizon_length)).T, 
                                            output_trajectory_var.reshape((num_outputs, prediction_horizon_length+initial_horizon_length)).T])

    return controller_solver, parameter_substitution_function, solution_extraction_function, trajectory_plotting_function

# Maintain backward compatibility with original function names if needed
generate_DDMPC_robust_solver = build_robust_data_driven_controller
generate_DDMPC_solver = build_standard_data_driven_controller
