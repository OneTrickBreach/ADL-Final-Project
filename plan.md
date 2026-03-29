### **Final Project Plan (Execute Solo in WSL: Ubuntu)**

#### **Phase 1: High-Performance Stack (RTX 5070 Ti Optimization)**
*   [ ] Create a native Linux virtual environment: `python3 -m venv.venv`.
*   [ ] Activate and install **Torch with CUDA 12.4** (essential for the 5070 Ti):
    `./venv/bin/pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124`.
*   [ ] Install the MARL stack: `./venv/bin/pip install 'pettingzoo[butterfly]' supersuit 'ray[rllib]' tensorboard opencv-python`.
*   [ ] **GPU Handshake:** Run a script to verify `torch.cuda.is_available()` returns `True` and identifies the 5070 Ti.

#### **Phase 2: Game 1 - The Villain (Standard Role-Failure)**
*   [ ] **Environment Setup:** Implement `src/env_villain.py` with standard `knights_archers_zombies_v10`. Use `vector_state=False` for full visual parity .
*   [ ] **Training:** Use **Independent PPO (IPPO)** but with a **Shared Policy**. This forces role-confusion, showcasing how the Knight "forgets" to defend the Archer .
*   [ ] **Data Capture:** Save TensorBoard logs showing high "Average Reward" but frequent "Border Breaches" and "Teammate Death" events.
*   [ ] **Visual Proof:** Generate a GIF of a Knight charging into a swarm while the Archer is killed from behind.[3]

#### **Phase 3: Game 2 - The Hero (Solving the "Impossible" Constraint)**
*   [ ] **The "Impossible" Wrapper:** Create `src/env_hero.py` with a custom logic layer:
    *   **Resource Scarcity:** Archer starts with only 20 arrows. Knight swings cost -0.2 reward (Stamina penalty) .
    *   **Perceptual Smoke:** Add Gaussian Noise to observations. Agents must learn to "wait for clarity" to avoid wasting arrows .
*   [ ] **The Machinery (Divergent Rewards):**
    *   **Knight:** +1.0 kill, -10.0 if Archer dies (High Preservation Bias) .
    *   **Archer:** +1.0 kill, -20.0 if Border breached (Mission Integrity).
*   [ ] **Training:** Implement a **Pareto Sweep** over the "Stamina Penalty" weight to find the point where agents become "Cautious Experts" .

#### **Phase 4: Interpretability & "The Story" (The 96+ Grade Layer)**
*   [ ] **Saliency Maps:** Implement `src/interpret.py` using **LRP (Layer-wise Relevance Propagation)** or **Saliency Maps** to show which zombies the Archer is "watching" versus the Knight .
*   [ ] **The Margin Narrative:** Calculate a "Profit Metric." Arrows/Stamina = Business Cost. Surviving = Business Revenue. 
*   [ ] **Success Metric:** Prove the Hero model achieves a **lower "Cost-per-Kill"** and higher "Team Profit" than the Villain, satisfying Steve’s demand for business grounding.[3]

#### **Phase 5: Submit & Deploy**
*   [ ] Perform a **Stress Test**: Increase zombie spawn rate by 50% and document how the Hero's "Fire Discipline" allows them to survive longer than the Villain .
*   [ ] Finalize `README.md` and push all code/checkpoints to GitHub.