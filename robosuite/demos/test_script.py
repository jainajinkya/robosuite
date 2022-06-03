import pickle
from pathlib import Path
from robosuite.controllers import load_controller_config
from robosuite.utils.input_utils import *

import plotly.graph_objects as go
from plotly.subplots import make_subplots


if __name__ == "__main__":

    # Load test script
    test_file_path = Path(__file__).parents[1].resolve().joinpath("data/sampled_trajectories/sample_jnt_actions_1.pkl")
    test_data = pickle.load(open(test_file_path, "rb"))
    test_jnt_data = test_data["qpos"]

    # Create dict to hold options that will be passed to env creation call
    options = {}

    # Choose environment and add it to options
    options["env_name"] = "Door"  # choose_environment()
    options["robots"] = "Panda"

    # Choose controller
    # controller_name = choose_controller()
    controller_name = "COMPUTED_TORQUE"

    # Load the desired controller
    options["controller_configs"] = load_controller_config(default_controller=controller_name)

    # Help message to user
    print()
    print('Press "H" to show the viewer control panel.')

    # initialize the task
    env = suite.make(
        **options,
        has_renderer=True,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
    )
    env.reset()
    env.viewer.set_camera(camera_id=0)

    # Generate path

    # do visualization with renderer
    actual_jnt_trajectories = []

    for act in test_jnt_data:
        obs, reward, done, _ = env.step(np.array(act))
        actual_jnt_trajectories.append(env.robots[0].recent_qpos.current)
        env.render()

    # Visualize Controller performance
    desired_jnt_trajectories = np.array(test_jnt_data)
    x_vec = np.arange(len(desired_jnt_trajectories))
    actual_jnt_trajectories = np.array(actual_jnt_trajectories)

    fig = make_subplots(
        rows=7,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=[f"Jnt_{i}" for i in range(7)],
    )

    SHOW_LEGEND = [True] + [False] * 6

    for i in range(7):
        fig.add_trace(
            go.Scatter(
                x=x_vec,
                y=desired_jnt_trajectories[:, i],
                mode="lines+markers",
                marker=dict(color="blue", size=5, symbol="square"),
                name="desired",
                legendgroup="desired",
                showlegend=SHOW_LEGEND[i],
            ),
            row=i + 1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x_vec,
                y=actual_jnt_trajectories[:, i],
                mode="lines+markers",
                marker=dict(color="red", size=5, symbol="circle"),
                name="actual",
                legendgroup="actual",
                showlegend=SHOW_LEGEND[i],
            ),
            row=i + 1,
            col=1,
        )

    fig.update_layout(title_text="Cubic Interpolation")
    fig.show()
