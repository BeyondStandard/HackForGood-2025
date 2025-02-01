# Red Cross Information Access Platform

## Demonstration

💻 See our website final product at https://hackathon-website-wheat.vercel.app/)!

📞 Call +31 970 102 500 66 to speak with our conversational AI and get the help you need!

(If the phone doesn't pick up then the fellow participants might be overloading the line, please try again later)

---

## Table of Contents
1. Overview
2. Usage
3. Features  
4. Technical Details

---

## Overview
Imagine arriving in a new city, facing an unfamiliar language and needing urgent assistance.
This is a problem that afflicts many people including refugees, asylum seekers and undocumented immigrants.
The current interface is slow, not particularly accessible but still manages to be overwhelming.

Our project seeks to improve this experience by streamlining access to critical resources through a combination of intelligent navigation, UI revamp,  and AI-driven support agents.
Built for the Red Cross, this platform offers an intuitive interface and a multilingual chatbot backed by real-time information from a dynamic content management system (CMS).

---

## Usage

### Accessing the Website
Navigate to [website URL](https://hackathon-website-wheat.vercel.app/) to explore the demo. Key services such as shelter information and frequently asked questions are accessible directly from the homepage.

### Chatbot Interaction
- **Text-based:** Type questions directly into the chatbot interface.
- **Voice-based:** Dial +31 970 102 500 66 to speak to the chatbot and receive spoken responses via Twilio’s integration.

---

## Features

### 1. Simplified Website Structure, Revamped UI
- Reduced the number of clicks required to reach essential information.
- Cleaner navigation focused on high-priority, time-sensitive resources.

### 2. Search Bar  
- Quickly find shelters, services, or other critical information without having to browse.

### 3. Conversational AI Chatbot  
- Natural language interaction: Ask questions in your native language.
- Powered by **Retrieval-Augmented Generation (RAG)** for accurate responses, with data verified through Red Cross resources.

### 4. Multi-Channel Support  
- For users who prefer direct communication, a **phone number** connects to the chatbot using Twilio’s speech processing integration.

### 5. Automated and Updated CMS System  
- Data integrity is maintained through an easy-to-use CMS that automatically updates the entire platform. Adding new content is fast, secure, and requires minimal effort.

---

## Technical Details
This project is built using the following technologies and services:

- **Backend:** FastAPI for handling asynchronous requests efficiently.
- **Twilio Integration:** For speech-based interactions via phone.
- **Sanity:** Content Management Software platform used to store data for the website.
- **Data Management:** Retrieval-Augmented Generation (RAG) for dynamic content extraction.
- **Conversational AI:** LangChain for enhanced context retrieval and natural language understanding.
- **Docker:** Containerized deployment for load managing.
- **Hosting:** Hosted on AWS.
- **LLM:** OpenAI ChatAPI for responses (can be swapped out).

---

This project was developed as part for Hackathon for Good 2025!
