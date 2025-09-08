# HabitLink Development Action Plan (MVP)

## 1. Project Overview

- **Development Period:** September 1, 2025 - October 31, 2025 (8 weeks total)
- **Core Objective:** To develop an MVP (Minimum Viable Product) web application prototype within 2 months, including core features for correcting Korean speech habits (speech rate, keyword counting, grammar error analysis).
- **Proposed Tech Stack:**
    - **Backend:** Python (FastAPI or Flask)
    - **Frontend:** Streamlit (Chosen for rapid prototyping)
    - **STT (Speech-to-Text):** WhisperLiveKit (Local-based real-time STT and speaker recognition)
    - **LLM:** OpenAI GPT API (For grammar correction and analysis)
    - **Database:** SQLite (For initial data storage)

---

## 2. Development Roadmap & Weekly Plan

### Phase 1: Core Backend & Analysis Module Development (September, 4 weeks)

**Weeks 1-2 (9/1 ~ 9/15): Environment Setup & Core Module Implementation**

- **Task 1: Development Environment Setup**
    - Set up Git repository and establish a branching strategy.
    - Configure a Python virtual environment and manage libraries in `requirements.txt`.
    - Design the basic backend project structure using FastAPI or Flask.
- **Task 2: Real-time Voice Capture & STT Module Development (based on WhisperLiveKit)**
    - Install dependency libraries for `WhisperLiveKit`, such as `ffmpeg`.
    - Write a script to run a local STT server using the `WhisperLiveKit` library.
    - Implement a module to receive audio streams from the user's microphone, send them to the local STT server, and receive the transcribed text in real-time via WebSocket.
    - Include logic to immediately discard voice data after STT conversion to protect privacy.
- **Task 3: Basic Analysis Module Implementation (2 Metrics)**
    - **Speech Rate (WPM):** Implement logic to calculate words per minute based on the transcribed text.
    - **Filler Word/Profanity Counting:** Implement a feature to count the frequency of user-defined keywords.

**Weeks 3-4 (9/16 ~ 9/30): LLM Integration & Feedback Prototype**

- **Task 4: LLM-based Grammar Error Analysis Module**
    - Develop a module to integrate with the OpenAI GPT API.
    - Implement a feature to send transcribed text to the LLM to detect grammar errors and suggest corrections.
    - Perform prompt engineering to optimize API costs and response speed.
- **Task 5: Design and Store Analysis Data Structure**
    - Design a data model (e.g., SQLite table) to store analysis results (WPM, keyword frequency, grammar errors, etc.).
    - Implement logic to save session-based analysis data to the database.
- **Task 6: Feedback Simulation & API Design**
    - Simulate a "vibration feedback event" via console logs or events when analysis results meet specific conditions (e.g., exceeding target WPM, keyword detection).
    - Design basic REST API endpoints for frontend integration (`/start`, `/stop`, `/get_analysis`).

---

### Phase 2: MVP Frontend & Integration (October, 4 weeks)

**Weeks 5-6 (10/1 ~ 10/15): UI Prototyping & API Integration**

- **Task 7: Web-based UI Prototype Development (Streamlit)**
    - **Main Screen:** Recording start/stop button, real-time transcription display area.
    - **Settings Screen:** UI for users to input keywords (filler words/profanity) for analysis and set a target WPM.
    - **Results Screen:** Dashboard to display real-time analysis results (current WPM, keyword detection count, etc.).
- **Task 8: Backend API Server and Frontend Integration**
    - Call the backend API to send a real-time audio stream when recording starts from the web UI.
    - Fetch and display the STT transcription and analysis results from the backend in the UI in real-time.
- **Task 9: Dialect Analysis Model Development (Proof-of-Concept)**
    - **Model Selection & Data Prep:** Select a pre-trained `Wav2Vec2` model from Hugging Face. Download and preprocess a subset of a Korean dialect dataset (e.g., from AI Hub), focusing on 2-3 distinct dialects plus the standard dialect.
    - **Fine-tuning:** Develop and run a script to fine-tune the `Wav2Vec2` model for the dialect classification task. Evaluate its initial performance.

**Weeks 7-8 (10/16 ~ 10/31): Integration Testing & MVP Completion**

- **Task 10: Post-Session Report Feature Implementation**
    - After a recording session ends, retrieve the complete analysis data for that session from the database.
    - Visualize a summary report (average WPM, total keyword detections, major grammar errors, etc.) on the web dashboard using simple text and charts.
- **Task 11: Integration of Dialect Analysis Module**
    - Integrate the fine-tuned dialect model into the backend, creating a new API endpoint for analysis.
    - Update the Streamlit dashboard to display the dialect analysis results (e.g., "Gyeongsang/Chungcheong Dialect: 85%").
- **Task 12: Contextual Appropriateness Analysis Module**
    - **Backend Logic:** Implement logic to maintain a short-term history of the last 3-5 transcribed sentences for each user session.
    - **LLM Prompting:** Design and test a prompt that sends the conversation history and the latest utterance to an LLM to evaluate its contextual relevance.
    - **API & UI Integration:** Create a new endpoint for this analysis and display the feedback (e.g., "Off-topic") on the dashboard.
- **Task 13: E2E (End-to-End) Integration Testing & Bug Fixes**
    - Test the entire user flow, including all analysis features (WPM, Keywords, Grammar, Dialect, Context).
    - Fix any identified bugs and work on stability improvements.
- **Task 14: Final MVP Cleanup & Documentation**
    - Update the `README.md` file with instructions on how to run the project, its main features, and an architectural overview.
    - Clean up code comments and prepare for deployment.

---

## 3. Post-MVP Considerations (Future Works)

- **Mobile/Wearable App Development:** Start native app development based on the core logic of the web MVP.
- **Advanced Reporting:** Generate and allow downloading of Clinical Reports in PDF format.
- **CI/CD Pipeline:** Set up an automated environment for testing, building, and deployment.
- **User Authentication System:** Add user account and data management features.


