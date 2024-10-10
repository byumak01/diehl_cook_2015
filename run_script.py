import libtmux
import itertools
import time

server = libtmux.Server()

params = {}
params["--normalization_const"] = [2, 3, 4, 5, 6, 7, 8, 9, 10]
params["--g_e_multiplier"] = [2.5, 3, 3.5, 4]
params["--image_count"] = [10000]
params["--acc_update_interval"] = [2500]
params["--draw_update_interval"] = [500]

param_count = len(params)
param_names = list(params.keys())

ranges = [item for item in params.values()]

arguments = ""
for run_id, param_values in enumerate(itertools.product(*ranges)):
    # Construct the argument string
    for i in range(param_count):
        arguments += f" {param_names[i]} {param_values[i]}"
    
    # Create a new session and capture the session ID
    session_id = server.cmd('new-session', '-d', '-P', '-F#{session_id}').stdout[0].strip()
    print(f"session id {session_id}") 
    # Get the newly created session using the session ID, not run_id
    session = server.find_where({"session_id": session_id})
    if session is None:
        print(f"Error: Session {session_id} not found.")
        continue
    
    print(f"Created session: {session_id}, server.sessions: {server.sessions}")
    
    # Access the window and pane in the new session
    window = session.attached_window
    pane = window.panes[0]
    
    # Send commands to the new pane
    pane.send_keys("brian2")
    # pane.send_keys("source /home/bymk/Documents/tubitak/diehl_cook_2015/.venv_linux/bin/activate")
    pane.send_keys(f"python test.py --seed_data {arguments}")

    print(f"Run {run_id} arguments: {arguments}")
    
    # Clear arguments for the next run
    arguments = ""
    
    # Sleep to avoid overloading the system with too many sessions at once
    time.sleep(2)

