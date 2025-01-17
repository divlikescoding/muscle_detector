# Louie Lab Muscle v. Non-muscle Detector

Welcome to the Louie Lab Muscle v. Non-muscle Detector web application! This tool allows you to analyze heart cross-sections by detecting muscle and non-muscle regions using a Machine Learning Segmentation Model. Below are the steps to get started and use the application effectively.

## Getting Started

1. **Download the Repository**
   - Navigate to the repository page and download the zip file.
   - Extract the contents of the zip file to a folder on your local machine.

2. **Run the Project**
   - Open a terminal or command prompt.
   - Download python by going to this link(choose mac or windows or linux based on your computer): https://www.python.org/downloads/
   - Navigate to the folder where the repository files are extracted.
   - Follow the instructions below to start the application:

     ```bash

      # Create and activate Virtual Environment to run website
     python3 -m venv venv
     source venv/bin/activate
     
     # Install dependencies
     pip install -r requirements.txt

     # Start the application
     python manage.py runserver
     ```
   - Open your web browser and copy the specified URL in the search bar to access the application.

## Using the Web Application

1. **Upload an Image**
   - Prepare a `.tif` image of a heart cross-section.
   - Click the "Upload" button on the application homepage.
   - Select and upload your `.tif` image.

2. **Download Results**
   - Once the analysis is complete, the ROIs (regions of interest) will be ready for download.
   - Click the "Download Zip" button to download a zip file containing the ROIs.
