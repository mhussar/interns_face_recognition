NOTE: The code in this repo is only a POC that face recognition works with this implementation. It is not the final code indicated by the project summary below
# 🤖 Face Recognition Kiosk with Voice Interaction

## 📌 Overview

This project implements a smart kiosk capable of recognizing individuals, engaging them through natural conversation, and responding interactively via AI. It’s built initially for a Windows development environment, with deployment targeted on a Raspberry Pi for compact, standalone operation.

---

## 🚀 Features

* **Real-time Face Detection and Recognition**
* **Speech-to-Text (STT)**: Transcribes user's spoken name.
* **Interactive AI Conversation**: Utilizes Google Gemini (primary) and Qwen (fallback).
* **Text-to-Speech (TTS)**: Generates audio responses using ElevenLabs.
* **Visual and Audible Interaction**: Responses displayed and spoken clearly.

---

## 🔄 How It Works

1. **Face Detection**: Webcam detects a face.
2. **Recognition Check**: Determines if face is known.
3. **Prompt**: Politely requests the user’s name if unknown.
4. **Voice Input**: User verbally provides their name.
5. **Transcription**: Voice input converted to text.
6. **Storage**: Face and name saved to local SQLite database.
7. **AI Interaction**: User interaction sent to AI (Gemini/Qwen).
8. **AI Response**: AI-generated conversational reply.
9. **Audio Output**: Response vocalized via ElevenLabs.
10. **Visual Display**: Response simultaneously displayed on-screen.

---

## 🛠️ Tech Stack

| Component         | Technology / Tool                 |
| ----------------- | --------------------------------- |
| Programming Lang  | Python 3.10+                      |
| Webcam Access     | opencv-python                     |
| Face Detection    | opencv-python (DNN Module, Caffe) |
| Face Recognition  | Custom Feature Comparison (numpy) |
| Speech-to-Text    | faster-whisper                    |
| Text-to-Speech    | ElevenLabs API                    |
| LLM (Primary)     | Gemini via Vertex AI              |
| LLM (Fallback)    | Qwen 3 via Ollama                 |
| Storage           | SQLite                            |
| Display           | tkinter with opencv-python feed   |
| Development OS    | Windows 10/11                     |
| Deployment Target | Raspberry Pi 4+ (64-bit OS)       |

---

## 🗂️ Project Workflow

### ✅ **Phase 1: Single-Camera PoC**

* Webcam face detection
* Recognition logic
* Name transcription via Faster-Whisper
* SQLite data storage
* Gemini/Qwen AI conversation
* ElevenLabs audio responses
* Visual and audible feedback

### 🔜 **Phase 2: Multi-Camera Upgrade** *(Planned)*

* Multiple simultaneous webcam inputs
* Enhanced face encoding accuracy
* Expanded UI for multi-angle input
* Improved reliability and user experience

---

## 📁 Deliverables

* `main.py`
* `/modules/` (project modules)
* `requirements.txt`
* `.env.example`
* `README.md`
* Sample SQLite DB with test users
* Raspberry Pi deployment guide

---

## 📖 Getting Started

Refer to the included `README.md` and Raspberry Pi deployment instructions to set up and run the kiosk application smoothly.

 

 
