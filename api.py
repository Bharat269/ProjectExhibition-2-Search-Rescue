from flask import Flask, request, jsonify
from finalenva import CustomSearchAndRescueEnv  # <- Replace with actual module if separate
import random
import json

app = Flask(__name__)

# Config for your environment
config = {
    "grid_size": 10,
    "num_agents": 2,
    "num_targets": 2,
    "observation_size": 3,
    "max_steps": 50,
    "output_file": "output.json"
}

# Load environment
env = CustomSearchAndRescueEnv(config)
observations = env.reset()
done = {"__all__": False}


@app.route('/', methods=['GET'])
def home():
    return "Search & Rescue Environment is Running. Use POST /step to interact."


@app.route('/step', methods=['POST'])
def step():
    global observations, done

    if done["__all__"]:
        return jsonify({"message": "Episode finished. Resetting environment."}), 200

    # You can accept action input from the user if needed.
    actions = {agent: env.action_space.sample() for agent in env.agents}  # random actions for demo
    observations, rewards, done, info = env.step(actions)

    # Get latest agent state (logged last)
    latest_data = env.data_to_log[-len(env.agents):]

    if done["__all__"]:
        env.close()

    return jsonify(latest_data), 200


@app.route('/reset', methods=['POST'])
def reset():
    global observations, done
    observations = env.reset()
    done = {"__all__": False}
    return jsonify({"message": "Environment reset!"}), 200


if __name__ == '__main__':
    app.run(debug=True)
