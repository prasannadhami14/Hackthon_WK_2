# Eye Tracking System

## Overview
The Eye Tracking System is a hackathon project developed during Week 2 (WK2) to track and analyze eye movements, potentially for applications like eye disease detection, user interface control, or accessibility features. This project integrates an API, a frontend interface, and a machine learning model to process and interpret eye-tracking data.

## Features
- Real-time eye tracking using a trained model.
- API endpoint for data processing and integration.
- Interactive frontend for visualizing tracking results.
- Support for custom eye-tracking datasets and models.

## Project Structure
Hackathon_WK2/
├── api/                # API-related files (e.g., endpoints for eye tracking data)
├── eye_tracking/       # Core eye-tracking logic and utilities
│   ├── tracker.py     # Main Python script for eye tracking
├── frontend/           # Frontend interface (e.g., HTML, CSS, JS)
├── model/              # Machine learning model files (e.g., weights, configuration)
├── .gitignore          # Git ignore file to exclude unnecessary files
├── README.md           # Project documentation
├── requirements.txt    # Python dependencies
└── myenv/              # Virtual environment (auto-generated, not committed)

## Prerequisites
- Python 3.8 or higher
- Virtualenv (recommended for dependency management)

## Installation
Follow these steps to set up the project locally:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/prasannadhami14/Hackthon_WK_2.git
   cd Hackathon_WK2
   Create a Virtual Environment:
2. **Create a Virtual Environment**: 
    ```bash 
    python -m venv myenv
3. **Activate the Virtual Environment:**: 
    ```bash 
    myenv\Scripts\activate
4. **Install Dependencies**: 
    ```bash 
   pip install -r requirements.txt
5. **Update Requirements (if needed)**: 
    ```bash 
   pip freeze > requirements.txt
