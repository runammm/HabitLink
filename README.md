# HabitLink: AI-Powered Korean Speech Habit Correction System

![HabitLink Banner](https://user-images.githubusercontent.com/12345678/123456789-placeholder.png) <!-- You can replace this with a real banner later -->

> A real-time analysis and feedback system designed to help Korean speakers recognize and correct their speech habits. This project provides immediate feedback and detailed post-analysis reports to improve communication skills.

---

## 🎯 Project Goals

It can be challenging for people to perceive and fix their own speech habits, such as using filler words, speaking too quickly, or making grammatical errors. HabitLink aims to solve this by providing a comprehensive analysis and feedback system specifically for the **Korean language**.

Our main goals are:
- **Comprehensive Analysis**: To analyze key indicators in real-time, including speech rate, filler words, and grammatical errors.
- **Dual Feedback System**: To offer both real-time corrective feedback and long-term learning motivation through post-session reports.
- **Modular Architecture**: To build the system with a microservices-based approach for scalability and maintainability.
- **Versatile Dashboard**: To provide a user-friendly dashboard for users to track their progress.

---

## ✨ Key Features (MVP)

The current development is focused on delivering a Minimum Viable Product (MVP) with the following core features:

- **🎤 Real-time Audio Transcription**: Captures microphone input and transcribes Korean speech into text in real-time.
- **🚀 Speech Rate (WPM) Analysis**: Measures Words Per Minute to provide feedback on speaking pace.
- **🔎 Custom Keyword Detection**: Counts the frequency of user-defined keywords, such as filler words or profanity.
- **🤖 LLM-based Grammar Check**: Leverages a Large Language Model (LLM) to detect grammatical errors and suggest corrections.
- **🖥️ Web Dashboard**: A web-based interface built with Streamlit for easy prototyping, allowing users to control sessions and view real-time analysis and post-session summaries.

---

## 🛠️ Technology Stack

| Category      | Technology                                                                          |
|---------------|-------------------------------------------------------------------------------------|
| **Backend**   | Python, FastAPI                                                                     |
| **Frontend**  | Streamlit (for MVP)                                                                 |
| **AI / ML**   | Google Cloud Speech-to-Text, OpenAI API (GPT)                                       |
| **Database**  | SQLite                                                                              |

---

## 🏛️ System Architecture (Conceptual)

HabitLink is designed with a modular, microservices-oriented architecture.

1.  **Voice Pre-processing Module**: Captures the user's voice, isolates it from background noise, and converts speech to text using an STT/ASR engine. The original audio is immediately deleted to protect privacy.
2.  **Analysis Loop Module**: The core engine that analyzes the transcribed text for the target indicators (Speech Rate, Filler Words, Grammar).
3.  **Feedback Module**: Delivers feedback based on the analysis, primarily through the web dashboard in the MVP.
4.  **User Interface (UI)**: A web dashboard that allows users to start/stop analysis sessions, configure settings, and view results.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- An external API key for STT (e.g., Google Cloud) and LLM (e.g., OpenAI).

### Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/HabitLink.git
    cd HabitLink
    ```

2.  **Create a virtual environment and activate it:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up environment variables:**
    Create a `.env` file in the root directory and add your API keys:
    ```
    OPENAI_API_KEY="your_openai_api_key"
    GOOGLE_APPLICATION_CREDENTIALS="path/to/your/google_credentials.json"
    ```

### Running the Application

1.  **Start the backend server:**
    ```bash
    uvicorn main:app --reload 
    ```

2.  **Run the frontend application:**
    (Assuming the Streamlit app is in a file named `app.py`)
    ```bash
    streamlit run app.py
    ```
---

## 🗺️ Project Roadmap

This project is currently under active development with a defined 2-month MVP plan. For a detailed week-by-week breakdown, please see the [ActionPlan.md](./ActionPlan.md) file.

- **Phase 1 (September 2024):** Core backend development, including STT/LLM integration and the implementation of core analysis modules.
- **Phase 2 (October 2024):** Development of the Streamlit-based web UI, API integration, and end-to-end testing.

### Future Goals
- [ ] Mobile & Wearable application with real-time haptic feedback.
- [ ] Advanced analysis: Dialect detection (Spectrogram-CNN) and contextual appropriateness.
- [ ] Speaker Diarization to distinguish speakers in a multi-person conversation.
- [ ] Comprehensive clinical reporting features for professionals (VLPs).

---

## 🤝 How to Contribute

We welcome contributions to the HabitLink project! Please follow these steps:

1.  **Fork** the repository.
2.  Create a new **branch** (`git checkout -b feature/YourFeature`).
3.  **Commit** your changes (`git commit -m 'Add some feature'`).
4.  **Push** to the branch (`git push origin feature/YourFeature`).
5.  Open a **Pull Request**.

---

## 📄 License

This project is licensed under the KAIST License. See the `LICENSE` file for details.