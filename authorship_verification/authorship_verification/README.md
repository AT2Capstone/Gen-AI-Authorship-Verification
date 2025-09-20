# Authorship Verification - Person A (Data & Infrastructure)

Project skeleton and core data utilities produced for the 50% milestone.

Structure:
- src/data_processing: data loader, preprocessing, validation
- src/utils: helper functions (save/load/report)
- data/: place your raw train/validation files here
- notebooks/: demo and exploratory notebooks
- requirements.txt
- setup.py

Quick start:
1. Create a virtualenv and install dependencies:
   python -m venv venv
   source venv/bin/activate   # or venv\Scripts\activate on Windows
   pip install -r requirements.txt
2. Place your dataset under ./data/ (see data_loader docstring for supported formats)
3. From project root run small python scripts or import the modules:
   from src.data_processing.data_loader import AuthorshipDataLoader
   loader = AuthorshipDataLoader('data/train')
   data = loader.load_data()

To run models:
1. To run Stylometry model, run "python stylometry_test.py" in the project root. (or python3)

