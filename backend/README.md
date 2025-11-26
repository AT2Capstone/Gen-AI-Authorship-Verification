# Authorship Verification - Person A (Data & Infrastructure)

Project skeleton and core data utilities produced for the 50% milestone.

Project Root: Gen-AI-Authorship-Verification/authorship_verification/authorship_verification

Structure:
- src/data_processing: data loader, preprocessing, validation
- src/utils: helper functions (save/load/report)
- data/: place your raw train/validation files here
- notebooks/: demo and exploratory notebooks
- requirements.txt
- setup.py

Quick start:
1. Create a virtualenv and install dependencies:
   py -3 -m venv venv
   source venv/bin/activate   # or venv\Scripts\activate on Windows
   pip install -r requirements.txt
   (Skip steps 2 and 3)
2. Paste any text inside sample-input.txt
3. Run the ensemble model:
   python ensemble.py

To run models individually:
1. To run Stylometry model, run "python stylometry_test.py" in the project root. (or python3 stylometry_test.py)
2. To run Bert Model, run "python tiny_berta_test.py" in project root.
Note: You don't need to train anything, it's already done.