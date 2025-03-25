# Real-Time-Drug-Review-Sentiment-Analysis-Image-Based-

📌 Project Description

This project utilizes a CNN-LSTM model to classify sentiment from pre-saved images representing three categories: Positive, Neutral, and Negative. The model is integrated into a Streamlit web application that allows users to upload images in real-time and visualize sentiment predictions using interactive analytics.

🚀 Features

CNN-LSTM Model for image-based sentiment classification.

Real-time image uploads for sentiment prediction.

Interactive visualization using bar charts and pie charts.

User-friendly Streamlit interface for ease of access.

🛠 Tools & Technologies

Deep Learning: TensorFlow, Keras

Data Visualization: Seaborn, Matplotlib

Web Framework: Streamlit

Image Processing: OpenCV, PIL

📂 Repository Structure

├── dataset/                 # Pre-saved sentiment images
├── models/                  # Trained CNN-LSTM models
├── src/                     # Python scripts for training & inference
│   ├── train_model.py        # Model training script
│   ├── predict.py            # Prediction script
│   ├── app.py                # Streamlit app
├── requirements.txt          # Required dependencies
├── README.md                 # Project documentation
├── LICENSE                   # License file
├── .gitignore                # Ignore unnecessary files

⚡ Installation & Usage

Clone the Repository

git clone https://github.com/yourusername/Drug-Review-Sentiment-Analysis-Image.git
cd Drug-Review-Sentiment-Analysis-Image

Install Dependencies

pip install -r requirements.txt

Run the Streamlit App

streamlit run app.py

📜 License

This project is licensed under the MIT License.

🤝 Contributing

Feel free to submit issues or pull requests to improve the project!
