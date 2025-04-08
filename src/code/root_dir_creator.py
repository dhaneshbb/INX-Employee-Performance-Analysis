import os
from pathlib import Path

def create_dir(path):
    """ Helper function to create directories if they don't exist. """
    path.mkdir(parents=True, exist_ok=True)

def create_file(path, content=''):
    """ Helper function to create and write to files. """
    with open(path, 'w') as file:
        file.write(content)

# Base directory
base_dir = Path("D:/00.Workspace/00 Projects/~ 02 pendding/iabac/INX_Future_Employee_Performance_Project")

# Directory structure as a dictionary where keys are paths and values are file content
structure = {
    base_dir / "Project_Summary/Requirements/CDS_Project_2_INX_Future_Emp_Data_V1.6.pdf": "Project Brief",
    base_dir / "Project_Summary/Analysis/Algorithm_Selection.md": "# Algorithm Selection\nDetails about chosen ML models.",
    base_dir / "Project_Summary/Analysis/Feature_Engineering_Report.pdf": "Feature Engineering Report",
    base_dir / "Project_Summary/Summary/Final_Report.pdf": "Final Project Report with Insights and Recommendations",
    base_dir / "data/external/": None,
    base_dir / "data/processed/cleaned_employee_data.csv": "employee_id,name,performance_score",
    base_dir / "data/raw/INX_Future_Inc_Employee_Performance_CDS_Project2_Data_V1.8.xls": "Excel data content",
    base_dir / "src/Data_Processing/data_processing.ipynb": "# Data Processing Notebook\nThis notebook contains data cleaning and encoding.",
    base_dir / "src/Data_Processing/data_exploratory_analysis.ipynb": "# Exploratory Data Analysis\nThis notebook explores the data statistically.",
    base_dir / "src/models/train_model.ipynb": "# Model Training\nThis notebook handles the training of the model.",
    base_dir / "src/models/model_performance.txt": "Model accuracy, F1-score, etc.",
    base_dir / "src/visualization/visualize.ipynb": "# Visualization\nThis notebook visualizes various aspects of the data.",
    base_dir / "src/visualization/plots/department_performance.png": "Plot image content",
    base_dir / "src/visualization/plots/feature_importance_shap.png": "SHAP values plot image content",
    base_dir / "references/IABAC_CDS_Project_Submission_Guidelines_V1.2.pdf": "Submission Guidelines",
    base_dir / "README.md": "# Project Overview\nThis project aims to...",
    base_dir / "requirements.txt": "numpy\npandas\nmatplotlib\nscikit-learn"
}

# Create directories and files
for path, content in structure.items():
    if path.suffix:  # It's a file
        create_dir(path.parent)
        create_file(path, content if content else '')
    else:  # It's a directory
        create_dir(path)

print("Directory structure and files have been created successfully.")
