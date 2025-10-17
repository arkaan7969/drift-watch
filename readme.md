# Drift Watch


**Drift Watch** is a lightweight system that monitors, compares, and calibrates distributed AI "canaries" to detect model drift or unexpected behavior over time.  
It’s designed for experiments in **self-healing AI** and **model integrity validation**.

---

## 🚀 Features

### Core Capabilities

- **Canary Network** — Deploy multiple fine-tuned micro-models to watch for behavioral drift.  
- **Drift Injection** — Simulate adversarial changes to test model stability.  
- **Comparison Engine** — Evaluate and visualize divergence between canary outputs.  
- **Calibration Loop** — Continuously retrain or rebalance canaries when deviation is detected.  

---

## 🧠 Goals

- Research model drift detection in distributed LLMs  
- Prototype self-healing inference systems  
- Build foundations for AI integrity verification frameworks  

---

## ⚙️ Setup

### Clone the repository
```bash
git clone git@github.com:arkaan7969/drift-watch.git
cd drift-watch
   ```
Install dependencies
```bash
pip install -r requirements.txt
  ```
Run the main system
```bash
python run_canaries.py
  ```

🧩 Project Structure
```bash
configs/               # Model and test configuration files
compare_canaries.py    # Drift comparison logic
calibrate_canaries.py  # Recalibration and healing module
inject_drift.py        # Drift injection utilities
heal_canary.py         # Recovery mechanisms
run_canaries.py        # Main orchestration script
live_loop.py           # Continuous monitoring loop
test_tinyllama.py      # Example model test harness
  ```

🧰 Built With

Python 3.11+ — Core programming language

PyTorch — Model fine-tuning and inference

NumPy / Pandas — Data analysis and drift metrics

Matplotlib — Visualization of drift patterns

FastAPI — Optional API endpoints for live monitoring

Docker — Containerized deployment

🧪 Experimental Extensions

Adversarial drift scenarios for robustness benchmarking

Self-healing loop for automated rebalancing

Audit logs for model behavior tracking over time


🖥️ Run via Docker

Using Docker Compose (Recommended)
```bash
docker compose up
  ```
Using Docker Directly
```bash
docker build -t drift-watch.app .
docker run -p 8080:8080 drift-watch.app
```


🧭 Research Focus

Drift Watch explores AI resilience, autonomy, and integrity verification through continuous evaluation of distributed canary models.
It’s an experimental framework for future trustworthy AI architectures.

📬 Author

Arkaan Sheikh
Founder of Audena | AI & Cybersecurity Researcher
GitHub

🪪 License
This project is licensed under the MIT License — see the LICENSE file for details.
