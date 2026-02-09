import numpy as np

def construct_hankel_data_matrices(
        system_order,
        initial_horizon_length,
        prediction_horizon_length,
        historical_input_data,
        historical_output_data
):
    """
    Construct Hankel matrices from historical input-output data for data-driven control.
    
    Parameters:
    - system_order: estimated order of the system
    - initial_horizon_length: length of initial condition window
    - prediction_horizon_length: length of prediction window  
    - historical_input_data: past input data matrix
    - historical_output_data: past output data matrix
    
    Returns:
    - combined_hankel_matrix: concatenated input-output Hankel matrix
    - input_past_block: past input Hankel block
    - input_future_block: future input Hankel block  
    - output_past_block: past output Hankel block
    - output_future_block: future output Hankel block
    """
    
    # Extract data dimensions
    total_samples, num_inputs = historical_input_data.shape
    num_outputs = historical_output_data.shape[1]
    
    # Verify persistent excitation condition
    total_horizon = initial_horizon_length + prediction_horizon_length + system_order
    pe_matrix = np.zeros((num_inputs * total_horizon, total_samples - total_horizon + 1))

    for row_idx in range(total_horizon):
        for col_idx in range(total_samples - total_horizon + 1):
            start_row = num_inputs * row_idx
            end_row = num_inputs * (row_idx + 1)
            pe_matrix[start_row:end_row, col_idx] = historical_input_data[row_idx + col_idx].squeeze()

    matrix_rank = np.linalg.matrix_rank(pe_matrix)
    required_rank = num_inputs * total_horizon
    print(f"PE matrix shape: {pe_matrix.shape}, rank: {matrix_rank}, PE satisfied: {matrix_rank == required_rank}")

    # Build input-output Hankel matrices
    io_horizon = initial_horizon_length + prediction_horizon_length
    num_columns = total_samples - io_horizon + 1
    
    input_hankel_matrix = np.zeros((num_inputs * io_horizon, num_columns))
    output_hankel_matrix = np.zeros((num_outputs * io_horizon, num_columns))

    for time_idx in range(io_horizon):
        for sample_idx in range(num_columns):
            # Fill input Hankel matrix
            input_start = num_inputs * time_idx
            input_end = num_inputs * (time_idx + 1)
            input_hankel_matrix[input_start:input_end, sample_idx] = historical_input_data[time_idx + sample_idx]
            
            # Fill output Hankel matrix  
            output_start = num_outputs * time_idx
            output_end = num_outputs * (time_idx + 1)
            output_hankel_matrix[output_start:output_end, sample_idx] = historical_output_data[time_idx + sample_idx]

    # Combine input and output Hankel matrices
    combined_hankel_matrix = np.vstack((input_hankel_matrix, output_hankel_matrix))

    # Decompose Hankel matrices into past and future blocks
    input_past_rows = num_inputs * initial_horizon_length
    output_past_rows = num_outputs * initial_horizon_length
    
    input_past_block = input_hankel_matrix[0:input_past_rows, :]
    input_future_block = input_hankel_matrix[input_past_rows:, :]
    output_past_block = output_hankel_matrix[0:output_past_rows, :]
    output_future_block = output_hankel_matrix[output_past_rows:, :]

    # Debug information (uncommented for verification)
    # print(f"Input Hankel shape: {input_hankel_matrix.shape}, rank: {np.linalg.matrix_rank(input_hankel_matrix)}")
    # print(f"Combined Hankel shape: {combined_hankel_matrix.shape}, rank: {np.linalg.matrix_rank(combined_hankel_matrix)}")
    # print(f"Block decomposition - Up: {input_past_block.shape}, Uf: {input_future_block.shape}, Yp: {output_past_block.shape}, Yf: {output_future_block.shape}")
    
    return combined_hankel_matrix, input_past_block, input_future_block, output_past_block, output_future_block