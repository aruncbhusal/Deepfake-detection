# DeepDetect — Deepfake Video Detection System

DeepDetect is a full-stack machine learning system that detects whether a video is real or synthetically manipulated (deepfake). It combines a PyTorch-based inference pipeline with a Next.js frontend for video upload and result visualization.

---

## System Overview

DeepDetect follows an end-to-end ML system pipeline:

**User Upload -> Frame Extraction -> CNN Feature Encoding -> BiLSTM Temporal Modeling -> Classification**

The system processes uploaded videos and returns a prediction along with a confidence score.

---

## Model Architecture

- Backbone: ResNet18 (pretrained on ImageNet)
- Temporal Model: Bidirectional LSTM (BiLSTM)
- Task: Binary classification (Real vs Fake)

### Pipeline

```
1. Extract frames from video
2. Resize and normalize frames
3. Extract spatial features using CNN
4. Model temporal dependencies using BiLSTM
5. Aggregate sequence features (mean pooling)
6. Predict final class
```

---

## Training Pipeline

The model was trained on a subset of the **Celeb-DF dataset** using a structured training pipeline.

### Key steps:

```
- Frame extraction from videos
- Data augmentation (rotation, flip, color jitter, blur)
- Preprocessing into tensor datasets (.pt files)
- Train/validation/test split (75/15/10)
- Training using CrossEntropyLoss and Adam optimizer
- Model checkpointing based on validation loss
```

> **Note:**  
> Due to dataset size constraints, the dataset and preprocessed tensors are not included in this repository.  
> The training pipeline is provided for reference and reproducibility.

---

## Tech Stack

### Frontend

- Next.js
- React
- TypeScript
- Tailwind CSS

### Backend

- Flask
- PyTorch
- OpenCV
- Hugging Face (model hosting)

---

## Project Structure

```
backend/
├── app.py
├── inference.py
├── preprocessing.py
├── model.py
├── config.py
├── download_model.py
└── models/

frontend/
├── src/app/layout.tsx
├── src/app/page.tsx
└── src/lib/api.ts

training/
├── train.py
├── dataset.py
├── evaluate.py
└── config.py

main.py
```

---

## Setup Instructions

### Clone Repository

```
git clone https://github.com/aruncbhusal/deepdetect.git
cd deepdetect
```

---

## Backend Setup

Install dependencies:

```
pip install -r requirements.txt
```

Run server:

```
python main.py
```

Backend runs at:

```
http://localhost:5000
```

---

## API Endpoints

### Health Check

```
GET /health
```

Response:

```
{
  "status": "ok"
}
```

---

### Prediction

```
POST /predict
```

#### Input

- multipart/form-data
- Field: file (video)

#### Response

```
{
  "success": true,
  "prediction": "Real",
  "label": 0,
  "confidence": 0.93
}
```

---

## Frontend Setup

```
cd frontend
npm install
```

Create `.env.local`:

```
NEXT_PUBLIC_API_URL=http://localhost:5000
```

Run frontend:

```
npm run dev
```

Frontend runs at:

```
http://localhost:3000
```

---

## End-to-End Flow

```
1. User uploads video via frontend
2. Frontend sends request to backend API
3. Backend extracts frames using OpenCV
4. Frames processed through CNN + BiLSTM model
5. Prediction and confidence returned
6. UI displays result
```

---

## Key Features

- Video-based deepfake detection
- Spatio-temporal modeling (CNN + BiLSTM)
- Confidence scoring via softmax probabilities
- Automatic model download from Hugging Face
- Modular ML pipeline (training + inference separation)
- API-based deployment with frontend integration
- File validation and safe temporary file handling

---

## Design Decisions

- Flask used for lightweight API development
- ResNet18 chosen for efficient feature extraction
- BiLSTM used to capture temporal inconsistencies in video
- Mean pooling used for stable sequence representation
- Lazy model loading to reduce startup overhead
- Preprocessed tensor datasets used to speed up training iterations

---

## Limitations

- CPU inference can be slow for large videos
- No batching for concurrent requests
- No real-time streaming inference
- Model performance depends on dataset quality and diversity

---

## Future Improvements

- Migration to FastAPI for async performance
- GPU acceleration and batching
- Model explainability (Grad-CAM)
- Dockerized deployment
- Real-time inference pipeline
- Multi-modal detection (audio + visual)

---

## Contribution

This project was developed collaboratively.

My part included:

- Designing and implementing the data preprocessing pipeline
- Building the model training pipeline
- Refactoring and modularizing the backend inference system
- Integrating ML pipeline with API for deployment

---

## License

MIT License
