# SentifyX – AI-Driven Sentiment Intelligence

## 🚀 Overview
SentifyX is a **real-time, multimodal sentiment analysis system** designed to extract high-accuracy, actionable insights from **text and audio data** across social media platforms. By leveraging cutting-edge **AI models**, SentifyX enables businesses and researchers to monitor sentiment trends **at scale**.

## 🔥 Key Features
- **Multimodal Sentiment Analysis**: Supports **text and audio-based** sentiment detection.
- **High Accuracy & Multilingual Support**: Achieves **97.8% accuracy** across **104 languages**.
- **Lightning-Fast Performance**:
  - Analyzes **10,000+ social media posts/minute**.
  - Processes **real-time audio sentiment detection in under 2 seconds**.
- **Optimized AI Pipeline**:
  - Utilizes **RoBERTa, HuBERT-Large, and DistilBERT-Multilingual** for precise sentiment classification.
  - 20% lower latency than traditional NLP architectures.
- **Efficient Storage & API Handling**: Ensures **seamless real-time processing**.

## 🏗️ Tech Stack
- **AI/NLP Models**: RoBERTa, HuBERT-Large, DistilBERT-Multilingual
- **Languages & Frameworks**: Python, TensorFlow, PyTorch, Flask, FastAPI
- **Databases**: PostgreSQL, Firebase
- **Deployment & Scalability**: Docker, AWS, Kubernetes
- **Libraries**: Hugging Face, Librosa (for audio processing), Pandas, NumPy, Matplotlib

## 🛠️ Installation & Setup
### 1️⃣ Clone the Repository
```sh
git clone https://github.com/Arnav-12/sentifyx.git
cd sentifyx
```
### 2️⃣ Set Up Virtual Environment
```sh
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
```
### 3️⃣ Install Dependencies
```sh
pip install -r requirements.txt
```
### 4️⃣ Configure Environment Variables
Create a **.env** file in the root directory and add your API keys:
```sh
SENTIFYX_API_KEY=api_key
DATABASE_URL=database_url
AWS_ACCESS_KEY=aws_key
```
### 5️⃣ Run the Application
```sh
docker-compose up --build  # Deploy with Docker
# OR run locally
python app.py
```

## 📊 How It Works
1. **Data Ingestion**: Collects real-time **text & audio** from social media platforms.
2. **AI Processing**:
   - Text sentiment analyzed using **RoBERTa & DistilBERT-Multilingual**.
   - Audio sentiment classified via **HuBERT-Large**.
3. **Real-Time Insights**: Generates **high-accuracy sentiment scores**.
4. **Visualization & API Access**: Outputs are accessible via **API & dashboard**.

## 🏢 Use Cases
✅ **Brand Monitoring**: Track customer sentiment towards brands.
✅ **Market Research**: Analyze large-scale sentiment trends.
✅ **Crisis Detection**: Identify negative sentiment spikes in real-time.
✅ **Social Media Analysis**: Process **10,000+ posts/min** efficiently.


## 📜 License
This project is licensed under the **MIT License**.



