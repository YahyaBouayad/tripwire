# Cable Detection Application

## Overview
This project is a machine learning application designed to detect the presence of cables in an image. The application uses a ResNet18 neural network model and is built with the Python-based Streamlit library to provide an intuitive and user-friendly interface. The model classifies images into two categories: "Cable Present" or "Cable Absent."

## Project Structure
- `dataset_raw/`: Contains the raw dataset organized into:
  - `cable_present/`: Images with cables present.
  - `cable_absent/`: Images with no cables.
- `dataset/`: Contains the prepared dataset split into:
  - `train/`: Training images.
  - `validation/`: Validation images.
  - `test/`: Testing images.
- `app.py`: The Streamlit application file that runs the cable detection interface.
- `Modelcreation.ipynb`: Jupyter Notebook file used to train and save the ResNet18 model.
- `cable_detector.pth`: The trained model file used for inference.

## Prerequisites
Before running the application, ensure you have the following installed:
- Python 3.7 or higher
- Required Python libraries (see below for installation instructions)

## Installation
1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/your-repo/cable-detection.git
   cd cable-detection
   ```

2. Install the required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```
   If `requirements.txt` is not available, install the following libraries manually:
   ```bash
   pip install streamlit torchvision matplotlib pillow torch
   ```

3. Ensure the `cable_detector.pth` file is present in the project directory. If not, train the model using the `Modelcreation.ipynb` notebook.

## Running the Application
1. Launch the Streamlit app by running the following command:
   ```bash
   streamlit run app.py
   ```
2. Open the URL provided by Streamlit in your browser (usually `http://localhost:8501`).

3. Upload an image file (supported formats: JPG, PNG, JPEG).

4. View the prediction result:
   - If a cable is detected, the background changes to red, and a warning message is displayed.
   - If no cable is detected, the background changes to green, and a success message is displayed.

## How It Works
1. **Model Loading:**
   - A ResNet18 model is loaded with custom weights from the `cable_detector.pth` file. The model's final fully connected layer is modified to output predictions for two classes: "Cable Present" and "Cable Absent."

2. **Image Preprocessing:**
   - Uploaded images are resized to 224x224 pixels.
   - Images are converted to tensors and normalized.

3. **Prediction:**
   - The processed image is passed through the model to generate predictions.
   - The predicted class is determined based on the model's output probabilities.

4. **User Feedback:**
   - The app visually displays the uploaded image along with the prediction result.
   - Background color dynamically changes based on the result for a more engaging user experience.

## Training the Model
If you need to retrain the model:
1. Open `Modelcreation.ipynb` in Jupyter Notebook.
2. Follow the steps to preprocess the data, define the model, and train it.
3. Save the trained model as `cable_detector.pth`.

## Future Improvements
- Add confidence scores for predictions.
- Include Grad-CAM visualization to show which regions of the image influenced the prediction.
- Implement a feature to upload multiple images at once for batch processing.
- Allow users to retrain the model from the app interface.

## Troubleshooting
- If the model fails to load, ensure the `cable_detector.pth` file is in the project directory.
- Verify that the required Python libraries are installed and compatible with your Python version.
- Check the logs in the terminal for detailed error messages.

## License
This project is open-source and available under the MIT License.

---
Feel free to reach out for any questions or issues regarding this project!
yahya.bouayad@edu.devinci.fr
