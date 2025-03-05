import sys
sys.path.append('.')  # Adds the current directory to the Python path

# Make tests a proper package
# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)