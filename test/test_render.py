"""
GUI render test — opens the KAZ environment window with random agents.
Run with: .venv/bin/python test/test_render.py
Close the window or press Ctrl+C to stop.
"""

from pettingzoo.butterfly.knights_archers_zombies import knights_archers_zombies as kaz

env = kaz.env(render_mode="human")
env.reset(seed=42)

print("KAZ window should be open. Close it or press Ctrl+C to exit.")

try:
    for agent in env.agent_iter():
        obs, reward, terminated, truncated, info = env.last()
        if terminated or truncated:
            action = None
        else:
            action = env.action_space(agent).sample()
        env.step(action)
except KeyboardInterrupt:
    print("\nStopped by user.")
finally:
    env.close()
    print("Environment closed.")
