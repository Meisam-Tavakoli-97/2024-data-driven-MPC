import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

def roll_and_update_buffer(data_vector, new_value):
    """
    Shift vector elements and replace last elements with new value.
    
    Parameters:
    - data_vector: existing data buffer to update
    - new_value: new data to append at the end
    
    Returns:
    - updated_vector: buffer with shifted and updated values
    """
    value_dimension = new_value.shape[0]
    shifted_vector = np.roll(data_vector, -value_dimension)
    shifted_vector[-value_dimension:, :] = new_value.reshape(shifted_vector[-value_dimension:, :].shape)
    return shifted_vector

def create_pseudo_random_binary_signal(seed_value, signal_amplitude, sequence_length):
    """
    Generate a pseudo-random binary sequence (PRBS) for system excitation.
    
    Parameters:
    - seed_value: initial seed for the linear feedback shift register
    - signal_amplitude: maximum amplitude of the generated signal
    - sequence_length: desired length of the sequence
    
    Returns:
    - binary_sequence: generated PRBS signal as a list
    """
    # Initialize binary sequence storage
    binary_sequence = []
    shift_register = seed_value
    sample_count = 1
    
    # Generate PRBS using 15-bit linear feedback shift register
    while sample_count <= sequence_length:
        # Calculate feedback bit from taps at positions 14 and 13
        feedback_bit = ((shift_register >> 14) ^ (shift_register >> 13) & 1)
        
        # Update shift register with feedback
        shift_register = ((shift_register << 1) + feedback_bit) & (2**15 - 1)
        binary_sequence.append(feedback_bit)
        
        # Debug output (commented for production)
        # print(sample_count, shift_register, feedback_bit, bin(shift_register))
        
        # Check for cycle completion
        if shift_register == seed_value:
            # print('Pattern repeat detected at length:', sample_count)
            break
        sample_count += 1

    # Convert binary sequence to floating point
    float_sequence = [float(bit_value) for bit_value in binary_sequence]
    
    # Scale and center the signal around zero
    for index in range(len(float_sequence)):
        float_sequence[index] = signal_amplitude * 2 * (float_sequence[index] - 0.5)

    # Optional visualization and analysis (commented out)
    # print("Pseudo-random binary sequence generated successfully")
    
    # Signal visualization
    # plt.plot(float_sequence)
    # plt.title('Generated PRBS Signal')
    # plt.show()
    
    # Autocorrelation analysis  
    # autocorrelation = sp.signal.correlate(float_sequence, float_sequence)
    # plt.plot(autocorrelation)
    # plt.title('PRBS Autocorrelation')
    # plt.show()

    return float_sequence

# Alias for backward compatibility (if needed)
shift_and_replace = roll_and_update_buffer
generate_PRBS15 = create_pseudo_random_binary_signal
