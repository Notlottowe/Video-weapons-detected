# Weapon Detection System

** University Hackathon 2nd Place Project Remake**

A real-time weapon detection system built with Streamlit, YOLO, and Google Gemini AI. This application analyzes video files to detect weapons and provides AI-powered threat analysis.

## Features

- **Video Analysis**: Upload and analyze video files (MP4, AVI, MOV, MKV)
- **Real-time Detection**: YOLO-based weapon detection with confidence scoring
- **AI Analysis**: Google Gemini AI provides detailed threat analysis
- **Best Shot Detection**: Automatically identifies the frame with highest detection confidence
- **Modern UI**: Clean, responsive interface with real-time status indicators

## Prerequisites

- Python 3.8 or higher
- Google API Key for Gemini AI (optional, for AI analysis feature)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/Weapons-detected-systeam.git
cd Weapons-detected-systeam
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up your Google API Key:
   - Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a `.env` file in the project root:
   ```
   GOOGLE_API_KEY=your_api_key_here
   ```
   - Or export it as an environment variable:
   ```bash
   # On Windows (PowerShell):
   $env:GOOGLE_API_KEY="your_api_key_here"
   
   # On macOS/Linux:
   export GOOGLE_API_KEY="your_api_key_here"
   ```

5. Add your YOLO model:
   - Place your trained YOLO model file (`best.pt`) in the `model/` directory
   - The model should be trained for weapon detection

## Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Open your browser and navigate to the URL shown in the terminal (usually `http://localhost:8501`)

3. Upload a video file using the file uploader

4. The system will:
   - Process the video frame by frame
   - Display real-time detection results
   - Show the frame with the highest confidence detection
   - Generate an AI-powered analysis (if API key is configured)

5. Click "End Session" to stop processing and view the final results

## Project Structure

```
Weapons-detected-systeam/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── model/                 # YOLO model directory
│   └── best.pt           # Trained YOLO model (not included in repo)
├── .gitignore            # Git ignore rules
├── .env.example          # Environment variables template
└── README.md             # This file
```

## Configuration

- **Detection Confidence**: Currently set to 0.4 (can be modified in `app.py` line 203)
- **Processing Width**: Set to 640px for performance (can be modified in `app.py` line 172)
- **Frame Skip**: Automatically calculated to process ~3 FPS for efficiency

## Notes

- The model file (`best.pt`) is not included in this repository due to size limitations
- You'll need to train your own YOLO model or obtain one separately
- The AI analysis feature requires a valid Google API key
- Large video files may take time to process

## Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Uses [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- AI analysis powered by [Google Gemini](https://deepmind.google/technologies/gemini/)



