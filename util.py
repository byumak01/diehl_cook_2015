import os, csv

def write_to_csv(args, accuracy, sim_time, filename='default_runs.csv'):
    # Get a dictionary of all arguments
    args_dict = vars(args)

    # Add the accuracy to the dictionary
    args_dict['accuracy'] = accuracy
    args_dict['sim_time'] = sim_time

    # Check if the file exists to determine if we need to write the header
    file_exists = os.path.isfile(filename)

    # Writing to a CSV file in append mode
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)

        # Write header (parameter names) if file does not exist
        if not file_exists:
            writer.writerow(args_dict.keys())

        # Write parameter values (append to the file)
        writer.writerow(args_dict.values())

    print(f"Parameters and accuracy appended to {filename}")