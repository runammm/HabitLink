# HabitLink: AI-Powered Korean Speech Habit Correction System

> A real-time analysis and feedback system designed to help Korean speakers recognize and correct their speech habits with immediate visual feedback and detailed post-session reports.

---

## 🎯 Project Goals

HabitLink provides comprehensive speech analysis to help Korean speakers improve their communication skills by identifying and correcting speech habits that are difficult to self-monitor.

Our main goals are:
- **Real-time Analysis**: Analyze speech habits as you speak with immediate visual feedback
- **Comprehensive Feedback**: Track speech rate, keywords, profanity, grammar, context, and stuttering
- **Visual UI**: Interactive Pygame-based visualization with audio waveforms and alerts
- **Detailed Reports**: Generate PDF reports after each session for long-term progress tracking

---

## ✨ Current Features

- **🎤 Real-time Speech-to-Text**: Google Cloud STT with websocket streaming and automatic reconnection
- **🎨 Interactive UI**: Pygame-based 3D sphere that responds to voice input with real-time waveforms
- **🚀 Speech Rate Analysis**: Measures Words Per Minute with customizable target speeds
- **🔎 Custom Keyword Detection**: Track specific filler words or repetitive phrases
- **⚠️ Profanity Detection**: Real-time detection of inappropriate language
- **🤖 Grammar Analysis**: LLM-powered detection of spoken grammar errors
- **🧠 Context Analysis**: Evaluates contextual appropriateness of utterances
- **🗣️ Stutter Analysis**: Dual-mode detection using both real-time audio and post-processing
- **🌏 Dialect Detection**: AI-powered binary classification (Standard vs Non-Standard Korean)
- **📄 PDF Reports**: Comprehensive post-session reports with detailed analytics

---

## 🛠️ Technology Stack

| Category      | Technology                                                                          |
|---------------|-------------------------------------------------------------------------------------|
| **STT**       | Google Cloud Speech-to-Text (Streaming API)                                        |
| **LLM**       | Groq API (for grammar and context analysis)                                         |
| **AI Model**  | Wav2Vec2 (for dialect classification)                                               |
| **Audio**     | PyAudio, SoundDevice, Librosa                                                      |
| **UI**        | Pygame                                                                              |
| **Reports**   | ReportLab                                                                           |
| **Language**  | Python 3.11+                                                                        |

---

## 🏛️ System Architecture

HabitLink uses a multi-threaded streaming architecture:

1. **Streaming STT Module**: Captures microphone input and streams to Google Cloud STT with automatic reconnection
2. **Real-time Analysis**: Fast analyses (keywords, profanity, speech rate) run immediately on each transcript
3. **LLM Analysis**: Grammar and context analysis runs periodically (every 5 seconds)
4. **UI Visualizer**: Pygame-based main thread displays real-time audio visualization
5. **Audio-based Stutter Detection**: Analyzes raw audio for stuttering patterns independently of STT
6. **Report Generator**: Creates comprehensive PDF reports after each session

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- Google Cloud Account with Speech-to-Text API enabled
- Groq API Key
- macOS/Linux/Windows (UI works best on macOS)

### Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/HabitLink.git
   cd HabitLink
   ```

2. **Create a virtual environment and activate it:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up Google Cloud credentials:**
   - Download your service account JSON key from Google Cloud Console
   - Save it as `gcp_credentials.json` in the project root, or
   - Set `GOOGLE_APPLICATION_CREDENTIALS` environment variable

5. **Set up environment variables:**
   Create a `.env` file in the root directory:
   ```
   GROQ_API_KEY="your_groq_api_key"
   GOOGLE_APPLICATION_CREDENTIALS="path/to/gcp_credentials.json"
   ```

6. **Prepare profanity list:**
   - Create `resources/profanity_list_ko.txt` with Korean profanity words (one per line)

7. **(Optional) Set up Dialect Detection:**
   - Follow the guide in `DIALECT_BINARY_GUIDE.md` for binary classification
   - **Recommended**: Train model on Google Colab (includes free GPU!)
   - Model training notebook: `notebooks/dialect_model_training.ipynb`
   - Requires additional dependencies (auto-installed in Colab):
   ```bash
   pip install transformers torch datasets accelerate evaluate scikit-learn
   ```

### Running the Application

```bash
python main.py
```

The application will:
1. Ask if you want to enable UI visualization (recommended)
2. Let you select which analysis modules to activate
3. If speech rate analysis is selected, record a calibration sample
4. Start real-time analysis with visual feedback
5. Generate a PDF report when you close the session

---

## 📊 Analysis Modules

### 1. Keyword Detection
Track specific words or phrases you want to monitor (e.g., filler words like "이제", "근데").

### 2. Profanity Detection
Automatically detect inappropriate language from a customizable list.

### 3. Speech Rate Analysis
Measures speaking speed in Words Per Minute (WPM) with personalized target rates.

### 4. Grammar Analysis
Uses LLM to detect spoken Korean grammar errors (particles, conjugations, honorifics, word order).

### 5. Context Analysis
Evaluates if utterances fit the conversational context and flow.

### 6. Stutter Analysis
Dual-mode detection:
- **Real-time**: Audio-based detection using MFCC, ZCR, and RMS energy
- **Post-processing**: Text-based pattern matching for repetitions, prolongations, and blocks

### 7. Dialect Detection (AI Model)
AI-powered binary classification for Korean speech:
- **Classification**: Standard Korean (표준어) vs Non-Standard Korean (비표준어)
- **Technology**: Fine-tuned Wav2Vec2 model
- **Output**: Binary classification with confidence scores
- **Training**: Use Google Colab for easy model training
- **Setup Required**: Train the model using your own dataset (see `DIALECT_BINARY_GUIDE.md`)

**Note**: This feature requires model training. See [Binary Classification Guide](DIALECT_BINARY_GUIDE.md) for detailed instructions.

---

## 🎨 UI Features

The Pygame UI displays:
- **3D Blue Sphere**: Size changes with voice volume
- **Real-time Waveform**: Shows audio input on the sphere's surface
- **Color Alerts**: Sphere turns light green when issues are detected
- **Notification Messages**: Top-right corner displays detection alerts

---

## 📄 PDF Reports

After each session, HabitLink generates a comprehensive PDF report containing:
- Full transcript with timestamps
- Keyword detection summary and occurrences
- Profanity detection results
- Speech rate analysis with segment-by-segment breakdown
- Grammar errors with corrections and explanations
- Context errors with reasoning
- Stutter analysis (both real-time and text-based)
- Dialect analysis with binary classification result (Standard vs Non-Standard)

Reports are saved to `.data/report/habitlink_report_YYYYMMDD_HHMMSS.pdf`

---

## 🗺️ Project Status

**Current Version**: v1.0 (MVP Complete)

### Completed Features
- ✅ Real-time Google Cloud STT streaming with reconnection
- ✅ Interactive Pygame UI with audio visualization
- ✅ All 7 analysis modules functional (including AI-powered dialect detection)
- ✅ PDF report generation with Korean font support
- ✅ Multi-threaded architecture for responsiveness
- ✅ Comprehensive error handling
- ✅ Dialect classification with fine-tunable Wav2Vec2 model

### Future Goals
- [ ] Web dashboard for progress tracking
- [ ] User authentication and session history
- [ ] Advanced analytics and trends
- [ ] Mobile application support
- [ ] Multi-language support

---

## 🤝 How to Contribute

We welcome contributions! Please follow these steps:

1. **Fork** the repository
2. Create a new **branch** (`git checkout -b feature/YourFeature`)
3. **Commit** your changes (`git commit -m 'Add some feature'`)
4. **Push** to the branch (`git push origin feature/YourFeature`)
5. Open a **Pull Request**

---

## 📄 License

This project is licensed under the KAIST License. See the `LICENSE` file for details.

---

## 🙏 Acknowledgments

- Google Cloud Speech-to-Text for reliable STT
- Groq for fast LLM inference
- PyGame community for visualization tools
- ReportLab for PDF generation
