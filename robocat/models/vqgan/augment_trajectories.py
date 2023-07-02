import numpy as np

def augment_trajectories_with_goal_images(tokenized_trajectories, success_detector):
    augmented_tokenized_trajectories = []

    for i, trajectory in enumerate(tokenized_trajectories):
        is_successful = success_detector(trajectory)

        if is_successful:
            # Select a goal image from the last image of another successful episode
            successful_episodes = [traj for j, traj in enumerate(tokenized_trajectories) if j != i and success_detector(traj)]
            goal_image = np.random.choice([traj[-1] for traj in successful_episodes])
        else:
            # Use the last image of the same episode as the goal image
            goal_image = trajectory[-1]

        # Add the selected goal image to each time step of the tokenized trajectory
        augmented_trajectory = [(x, I, goal_image, a) for x, I, a in trajectory]
        augmented_tokenized_trajectories.append(augmented_trajectory)

    return augmented_tokenized_trajectories

# Assuming tokenized_data is a list of tokenized trajectories
# and success_detector is a function that determines if a trajectory is successful
augmented_tokenized_trajectories = augment_trajectories_with_goal_images(tokenized_data, success_detector)
