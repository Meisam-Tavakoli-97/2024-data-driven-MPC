import numpy as np
from system_id import generate_PRBS15

# Initialize random number generator with seed for reproducibility
random_generator = np.random.default_rng(1234)

def collect_system_identification_data(
        num_inputs,
        num_outputs, 
        total_samples,
        system_dynamics,
        measurement_function,
        initial_state,
        enable_input_disturbance,
        input_disturbance_magnitude,
        enable_output_disturbance,
        output_disturbance_magnitude,
        feedback_controller=None,
        reference_signal=None,
        actuator_constraints=[np.inf, np.inf],
        sampling_period=0.0
):
    """
    Collect input-output data for system identification purposes.
    
    Parameters:
    - num_inputs: number of system inputs
    - num_outputs: number of system outputs  
    - total_samples: total number of data samples to collect
    - system_dynamics: discrete-time system dynamics function
    - measurement_function: output measurement function
    - initial_state: starting state of the system
    - enable_input_disturbance: flag to add noise to inputs
    - input_disturbance_magnitude: magnitude of input disturbance
    - enable_output_disturbance: flag to add noise to outputs
    - output_disturbance_magnitude: magnitude of output disturbance
    - feedback_controller: optional controller function
    - reference_signal: reference trajectory for controller
    - actuator_constraints: input saturation limits
    - sampling_period: time between samples
    
    Returns:
    - input_sequence: collected input data
    - output_sequence: collected output data
    """
    
    # Initialize data storage arrays
    input_sequence = np.zeros((total_samples, num_inputs))
    output_sequence = np.zeros((total_samples, num_outputs))
    
    # Generate disturbance signals
    input_disturbance_signal = np.zeros((total_samples, num_inputs))
    if enable_input_disturbance:
        input_disturbance_signal = random_generator.uniform(
            low=-input_disturbance_magnitude, 
            high=input_disturbance_magnitude, 
            size=(total_samples, num_inputs)
        )
        # Alternative PRBS generation (commented out):
        # for channel in range(num_inputs):
        #     input_disturbance_signal[:,channel] = np.array(generate_PRBS15(
        #         start=channel+1, 
        #         max_amplitude=input_disturbance_magnitude, 
        #         length=total_samples))
        
    # Generate measurement noise
    if enable_output_disturbance:
        measurement_noise = random_generator.uniform(
            low=-output_disturbance_magnitude, 
            high=output_disturbance_magnitude, 
            size=(total_samples, num_outputs)
        )
    else:
        measurement_noise = np.zeros((total_samples, num_outputs))

    # Initialize system state and controller variables
    current_state = np.copy(initial_state)
    integral_error = 0.0
    
    # Main data collection loop
    for sample_index in range(total_samples):
        # Determine control action
        if feedback_controller is None:
            # Open-loop excitation with disturbance only
            input_sequence[sample_index] = input_disturbance_signal[sample_index]
        else:
            # Closed-loop control with reference tracking
            control_signal, integral_error = feedback_controller(
                current_state, 
                reference_signal[sample_index], 
                actuator_constraints, 
                integral_error, 
                sampling_period
            )
            input_sequence[sample_index] = control_signal.squeeze() + input_disturbance_signal[sample_index]

        # Simulate system response
        output_sequence[sample_index] = (measurement_function(current_state, input_sequence[sample_index]).full().squeeze() + 
                                       measurement_noise[sample_index])
        current_state = system_dynamics(current_state, input_sequence[sample_index]).full().squeeze()

    return input_sequence, output_sequence