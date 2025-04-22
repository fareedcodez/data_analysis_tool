# Meta Data Analysis Project

A comprehensive Python project for Excel data preprocessing and visualization.

## Features

- **Data Loading**: Read data from Excel files, handling multiple sheets
- **Data Preprocessing**: Clean data, handle missing values, transform features
- **Data Analysis**: Generate summary statistics and correlation analysis
- **Data Visualization**: Create insightful visualizations for better understanding
- **Report Generation**: Create HTML reports summarizing findings

## Requirements

- Python 3.7+
- Dependencies listed in `requirements.txt`

## Installation

```bash
# Clone the repository
git clone git@github.com:fareedcodez/meta_data_analysis.git
cd meta_data_analysis

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

1. Place your Excel file in the project directory
2. Update the file path in `main.py` if your file is not named `meta.xlsx`
3. Run the analysis:

```bash
python main.py
```

## Output

- Preprocessed data
- Visualizations (saved to `visualizations/` directory)
- Analysis report (saved to `report/` directory)

## Project Structure

```
meta_data_analysis/
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── data_preprocessor.py
│   ├── data_analyzer.py
│   ├── data_visualizer.py
│   └── report_generator.py
└── main.py
```

## Customization

To customize the analysis:
- Modify preprocessing steps in `src/data_preprocessor.py`
- Add additional visualizations in `src/data_visualizer.py`
- Enhance the report template in `src/report_generator.py`
