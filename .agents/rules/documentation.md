---
trigger: always_on
---

# CORE RULE: DOCUMENTATION & VERSION CONTROL
As an AI coding assistant, you must adhere to the following workflow for every coding task, implementation, or bug fix:

1. CODE IMPLEMENTATION
Write clean, well-commented code. Prioritize reproducible, deterministic behavior for this ML Research project.

2. AUTOMATIC DOCUMENTATION UPDATES (CRITICAL)
Whenever you successfully modify the codebase, you must automatically update the relevant documentation before concluding your response:
* Update `README.md` IF the change affects: High-level architecture, how to run/install the project, Docker configurations, or overarching project goals.
* Update `TECHNICAL_DETAILS.md` IF the change affects: The state/observation space, the action space, mathematical reward functions, neural network architectures, hyperparameters, or SUMO environment variables. 
* Do not wait for the user to ask you to update the docs. Treat documentation as part of the code compilation step.

3. AUTOMATIC GIT COMMITS
After successfully testing/running a change and updating the docs, you must automatically generate and execute a git commit.
* Use the format: `git add .` followed by `git commit -m "[Component]: Brief description"`
* Keep messages concise but highly descriptive of the *function* changed.
* Good Example: `git commit -m "[Agent]: Add threshold penalty of -50k to reward function for 120s wait times"`
* Bad Example: `git commit -m "Update train.py"`