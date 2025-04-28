
# Cochlear Hair Cell Regeneration Prototype

This project simulates a basic prototype system to explore using AI for cochlear hair cell regeneration via scaffold evolution, mechanical resonance simulation, and cell response prediction.

## Project Structure

```
cochlear_prototype/
|— evolve_scaffold.py            # AI script to evolve scaffold shapes
|— simulate_resonance.py         # Physics script to simulate vibration frequency
|— predict_cell_response.py      # ML script to predict cell response based on resonance
|— utils/
    |— scaffold_utils.py         # Helper functions for scaffold geometry
    |— physics_utils.py          # Helper functions for resonance simulation
    |— model_utils.py            # Helper functions for ML model training
|— data/
    |— generated_scaffolds/      # Folder where evolved scaffold parameters are saved
    |— cell_response_data/       # Folder for training data for ML model
|— README.md
```

## Setup Instructions

**Environment:**  
Python 3.8+

**Install dependencies:**
```bash
pip install numpy scipy scikit-learn matplotlib tqdm
```

## Usage

### a. Evolve Scaffold Shapes:
```bash
python evolve_scaffold.py
```
Evolves scaffold shapes targeting a vibration frequency (default 100 Hz).

### b. Simulate Resonance:
```bash
python simulate_resonance.py
```
Simulates the mechanical resonance of generated scaffolds.

### c. Predict Cell Response:
```bash
python predict_cell_response.py
```
Trains and evaluates a simple machine learning model predicting cell differentiation likelihood.

### Change Target Frequency
Edit `TARGET_FREQUENCY` inside `evolve_scaffold.py` to customize.

## Data

- `generated_scaffolds/` stores scaffold parameters.
- `cell_response_data/` stores synthetic training data (can be replaced with real experimental data later).

## Project Goals

- Prototype an AI-driven system for designing cochlear scaffolds.
- Predict how mechanical cues affect cell behavior.
- Lay groundwork for more complex models (e.g., 3D simulations, reinforcement learning scaffold evolution).

## Future Expansion

- Add deep learning (e.g., CNNs) for predicting cell response.
- Integrate finite element simulation for realistic mechanical vibration.
- Connect to biological databases for real gene/protein mapping.
