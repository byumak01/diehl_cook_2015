import libtmux
import itertools
import time

server = libtmux.Server()

params = {}

params["--g_e_multiplier"] = []
params["--image_count"] = [10000]
params["--acc_update_interval"] = [2500]
params["--draw_update_interval"] = [500]

param_count = len(params)
param_names = list(params.keys())
print(param_names)

ranges = [item for item in params.values()]
print(f"ranges: {ranges}")

arguments = ""
for run_id, param_values in enumerate(itertools.product(*ranges)):
    for i in range(param_count):
        arguments += f" {param_names[i]} {param_values[i]}"

    server.cmd('new-session', '-d', '-P', '-F#{session_id}').stdout[0]
    print(f"server.sessions: {server.sessions}")

    session = server.sessions[run_id]
    print(f"session.window: {session.windows}")

    window = session.windows[0].panes[0]
    
    window.send_keys("brian2")
    #window.send_keys("source /home/bymk/Documents/tubitak/diehl_cook_2015/.venv_linux/bin/activate")
    window.send_keys(f"python test.py --seed_data  {arguments}")

    print(arguments)
    arguments = ""

    time.sleep(2)

