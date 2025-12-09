#!/bin/bash
# Helper script to run the project with virtual environment

# Activate virtual environment
source venv/Scripts/activate

# Run main.py
python app.py

# Deactivate when done
deactivate
