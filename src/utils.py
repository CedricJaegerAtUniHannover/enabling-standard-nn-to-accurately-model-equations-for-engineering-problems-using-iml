# src/utils.py

def get_short_model_name(dataset_name, model_details, seed):
    """Creates a shortened, standardized name for the model."""
    
    # Shorten dataset name by removing or replacing substrings
    s_name = dataset_name.replace('SYNTH_', 'S_')
    s_name = s_name.replace('_samples-10000', '')
    s_name = s_name.replace('_inputnoise-true', '_in-noise')
    s_name = s_name.replace('_noise-', '_out-noise-')
    s_name = s_name.replace('_vars-', '-')
    s_name = s_name.replace('complex_interaction', 'complex')
    s_name = s_name.replace('trigonometric', 'trig')
    s_name = s_name.replace('polynomial', 'poly')
    s_name = s_name.replace('linear', 'lin')
    s_name = s_name.replace('rational', 'rat')

    # Shorten model architecture details
    s_details = model_details.replace('layers-', 'l-').replace('neurons-', 'n-')

    # Assemble the final short name
    return f"{s_name}_ann_{s_details}_s-{seed}"
