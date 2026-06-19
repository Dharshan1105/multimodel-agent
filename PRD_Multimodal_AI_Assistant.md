# Product Requirements Document (PRD)

## Multimodal Agentic AI Assistant

**Project Codename:** MultiModal-Agent
**Version:** 2.0
**Status:** In Development
**Owner:** Dharshan
**Last Updated:** 2026-06-17

---

## How to Read This Document

This PRD (Product Requirements Document) explains what the project does, how it works, and what each part is responsible for. It is written so that anyone reading the code for the first time, including contributors and collaborators, can understand the purpose and logic of every component before writing or reviewing a single line of code.

Each section maps directly to a Python file or module in the project. When you open a file in VS Code, refer back to the matching section here to understand what that file is supposed to do.

---

## 1. Overview

### 1.1 What This Project Does

This project is a locally running AI assistant that can do four things:

1. Have a text-based conversation with you, remembering what you said earlier in the session.
2. Speak its responses out loud, automatically, without asking you to open an audio file.
3. Look at an image you provide and describe what it sees.
4. Generate a new image based on a text description you give it.

All of this runs on your own machine. No data is sent to the cloud. All AI models are downloaded and run locally using a tool called Ollama.

### 1.2 Why This Project Exists

Most AI assistants with these capabilities either require paid cloud APIs, do not remember what was said earlier in the conversation, or produce audio files that the user has to open manually. This project solves all three problems by building a unified local assistant with shared memory and automatic audio playback.

### 1.3 What We Are Building

A command-line assistant that:
- Reads what you type and decides which AI capability to use
- Remembers the full conversation within a session
- Plays spoken responses automatically without any extra steps from the user
- Stores image analysis and image generation history so you can ask follow-up questions

---

## 2. What Success Looks Like

The project is working correctly when all of the following are true:

| What to Test | How You Know It Works |
|---|---|
| Text conversation with memory | You can tell the assistant your name, ask later what your name is, and it answers correctly |
| Audio auto-playback | The assistant speaks its response and the next prompt appears automatically after the audio finishes |
| Image description | The assistant returns an accurate description of an uploaded image |
| Image generation | The assistant creates and saves an image file from a text prompt |
| Correct routing | Every request goes to the right AI model with no manual switching needed |
| Shared memory across text and audio | A fact mentioned in a text turn is remembered in a later audio turn |

---

## 3. What Is Included and What Is Not

### Included in This Project

- A text conversation module powered by Gemma3
- An audio response module that generates and plays speech automatically
- A vision module that uses LLaVA to describe images
- An image generation module that uses Stable Diffusion
- A router that decides which module to use based on the selected mode
- Shared memory so text and audio turns share the same conversation history
- Separate memory stores for image analysis and image generation history

### Not Included

- A web interface or browser-based UI
- Any cloud API calls or external internet access during inference
- Support for more than one user at a time
- Streaming responses (the full response is generated before being shown or spoken)

---

## 4. What Users Need to Be Able to Do

These are written as user stories so developers know what behavior each piece of code must support.

| ID | As a user, I want to... | So that... |
|---|---|---|
| US-01 | Have a back-and-forth text conversation | I do not have to repeat context in every message |
| US-02 | Hear the assistant speak its answer automatically | I do not have to open or play an audio file myself |
| US-03 | Continue asking questions after the audio finishes | The conversation keeps flowing naturally |
| US-04 | Give the assistant an image and get a description | I can understand visual content through natural language |
| US-05 | Ask follow-up questions about an image I already uploaded | I can explore image details without uploading it again |
| US-06 | Type a description and get a generated image | I can create images using plain language |
| US-07 | Have the assistant remember images it already generated | I can refer back to or adjust a previous generation |

---

## 5. How Each Module Works

Each module described below corresponds to one node in the LangGraph workflow. Think of each node as a self-contained unit that receives input, does its job, and returns output back to the shared state.

---

### 5.1 Text Conversation Module

This module handles plain text conversations. It is the simplest and most foundational part of the project.

**Model used:** Gemma3, running locally through Ollama

**What goes in:** A message typed by the user

**What comes out:** A text response from the model

**Memory:** This module shares its conversation history with the audio module. Both modules read from and write to the same list of messages. This means a fact mentioned during a text turn will still be remembered during an audio turn.

**What this module must do:**
- Accept a user message as input
- Pass the full conversation history to the model so it has context
- Return the model's response as text
- Append both the user message and the model response to the shared conversation history

**Example of correct behavior:**
```
User: My name is Dharshan
Assistant: Nice to meet you, Dharshan.

User: What is my name?
Assistant: Your name is Dharshan.
```

---

### 5.2 Audio Response Module

This module does everything the text module does, but instead of printing a response, it speaks the response out loud and waits for playback to finish before returning control to the user.

**Models used:** Gemma3 for generating the text response, then a TTS engine to convert that text to audio

**What goes in:** A message typed by the user

**What comes out:** A spoken audio response that plays automatically

**Memory:** Same shared conversation history as the text module

**TTS Engine Strategy — Two Phases**

This module uses two different TTS engines depending on the stage of development. Both are fully local and require no internet connection or API key.

Phase 1 — During development and testing:
Use pyttsx3. It requires no model download, no extra setup, and works immediately after installation. It sounds robotic, but that does not matter at this stage. The goal here is only to confirm that the audio node is wired up correctly and that the full flow from Gemma3 to TTS to playback to chat loop is working end to end. Do not spend time on voice quality until the logic is solid.

Phase 2 — Once the audio flow is confirmed working:
Replace pyttsx3 with Kokoro using the kokoro-onnx Python library. Kokoro produces natural sounding speech that is close to a real human voice. The swap requires only changing the tts_client.py file inside the models folder. Nothing else in the project needs to change because all TTS calls are isolated inside that one file. This is the reason the models folder exists as a separate layer.

**How playback works:**

The module generates the text response first using Gemma3. It then passes that text to whichever TTS engine is currently active in tts_client.py. The engine produces audio and plays it on a background thread. The main program waits until that playback is fully finished before showing the next input prompt. The user never has to find or open an audio file.

**Step-by-step flow:**
```
1. Receive the user message
2. Build the full message history and send it to Gemma3
3. Receive the text response from Gemma3
4. Pass the text to tts_client.py (pyttsx3 during dev, Kokoro in production)
5. Start audio playback on a background thread
6. Wait until playback finishes before continuing
7. Return to the chat loop so the user can type again
```

**Example of correct behavior:**
```
You: Explain OCI manifests
Assistant: (Speaking...)
(Audio plays and finishes on its own)
You: What are OCI layers?
Assistant: (Speaking...)
(Audio plays and finishes on its own)
```

No audio files are shown or saved to a location the user has to navigate to. The experience should feel like talking to a voice assistant.

---

### 5.3 Image Understanding Module (Vision)

This module takes an image file and returns a natural language description of what is in it.

**Model used:** LLaVA, running locally through Ollama

**What goes in:** A file path to an image, optionally followed by a specific question about the image

**What comes out:** A text description of the image

**Memory:** This module has its own separate memory store. Each entry in the store contains the image file path, the description that was generated, and a timestamp. This allows the user to ask follow-up questions about an image they already uploaded earlier in the session.

**What this module must do:**
- Accept an image file path as input
- Load the image and send it to LLaVA along with any accompanying question
- Return the description as text
- Save the image path, the description, and the timestamp to the image interaction memory store
- On follow-up questions, retrieve the most recent image from memory and use it as context

**Image types this should handle:**
- Architecture and system design diagrams
- Charts and graphs
- Photographs
- Screenshots
- Technical documentation with diagrams

**Example of correct behavior:**
```
User: Describe docker_architecture.png
Assistant: This image shows a Docker architecture diagram. It contains a Docker Client
           on the left, a Docker Daemon in the center, a Registry on the right, and
           several running containers managed by the Daemon.
```

---

### 5.4 Image Generation Module

This module takes a text description and produces an image from it using Stable Diffusion.

**Model used:** Stable Diffusion, running locally

**What goes in:** A text prompt describing the image to generate

**What comes out:** A saved image file with a generated filename such as generated_001.png

**Memory:** This module has its own separate memory store. Each entry contains the original prompt, the file path of the generated image, and metadata like the timestamp and any generation parameters used. This allows the user to reference or modify a previously generated image later in the session.

**What this module must do:**
- Accept a text prompt as input
- Pass the prompt to Stable Diffusion
- Save the output image to the outputs directory
- Store the prompt, the output file path, and metadata in the image generation memory store
- Return the file path so the user knows where to find the result

**Example of correct behavior:**
```
User: Generate an image of a futuristic city with flying vehicles at sunset
Assistant: Image saved to outputs/generated_001.png
```

---

## 6. Memory Design

There are three separate memory stores in this project. Understanding which module uses which store is important for keeping the code organized.

### 6.1 Shared Conversation Memory

Used by: Text module and Audio module

Stores: A list of messages. Each message has a role (either "user" or "assistant") and the content of the message.

Lifetime: The entire session. When the program exits, this memory is gone.

Why it is shared: The user might switch between typing and listening. The assistant should treat both as part of the same ongoing conversation.

### 6.2 Image Interaction Memory

Used by: Vision module only

Stores per entry:
- The file path of the image that was uploaded
- The description that LLaVA produced
- The timestamp of when it was analyzed

Why it exists: So the user can ask follow-up questions like "What color is the box in the top left?" without having to upload the image again.

### 6.3 Image Generation Memory

Used by: Image Generation module only

Stores per entry:
- The original prompt the user typed
- The file path of the generated image
- Generation metadata such as timestamp and model parameters

Why it exists: So the user can say things like "Generate a similar image but at night" and the module can retrieve what was previously generated to use as context.

---

## 7. How the Router Works

The router is the entry point of the application. Every user message passes through it first. It reads the currently selected mode and directs the message to the correct module.

### 7.1 Flow Diagram

```
User types a message
        |
        v
    LangGraph Router
        |
   Reads current mode
        |
   _____|_____________________________
  |          |          |            |
  v          v          v            v
Text       Audio      Vision     Image Gen
Module    Module      Module      Module
  |          |          |            |
  v          v          v            v
Gemma3     Gemma3    LLaVA      Stable Diffusion
        + pyttsx3
        (dev)
        or Kokoro
        (production)
```

### 7.2 Mode to Module Mapping

| Mode the user selects | Module that handles the request |
|---|---|
| text | Text Conversation Module |
| audio | Audio Response Module |
| vision | Image Understanding Module |
| imagine | Image Generation Module |

### 7.3 Shared State

LangGraph passes a state object between all modules. This state object holds:
- The currently active mode
- The shared conversation history (for text and audio)
- The image interaction memory (for vision)
- The image generation memory (for image generation)

Each module reads what it needs from the state and writes its results back to it before returning.

---

## 8. Technology Stack

| What it does | Technology used |
|---|---|
| Workflow orchestration and routing | LangGraph |
| Connecting models to the workflow | LangChain |
| Text conversation and audio text generation | Gemma3 |
| Image understanding | LLaVA |
| Image generation | Stable Diffusion |
| Text-to-speech synthesis (development) | pyttsx3 — zero setup, runs immediately, used to test the audio flow |
| Text-to-speech synthesis (production) | Kokoro via kokoro-onnx — high quality natural sounding voice, fully local |
| Running all models locally | Ollama |
| Programming language | Python 3.12 |
| User interface | Interactive command line |

---

## 9. Non-Functional Requirements

These are requirements about how the system behaves, not just what it does.

| Area | Requirement |
|---|---|
| Privacy | No data leaves the machine. All inference is local. |
| Audio timing | Audio playback must begin within 5 seconds of the user submitting a query. |
| Routing accuracy | Every request must be sent to the correct module. There should be no misrouting. |
| Extensibility | A developer should be able to add a new module without changing any of the existing modules. |
| Portability | The project should run on any machine that has Python 3.12 and Ollama installed. |

---

## 10. Concepts This Project Demonstrates

Reading this project and its code is a good way to learn about the following topics:

- How to build an agentic AI system where the program decides what to do next
- How to connect multiple AI models in a single workflow
- How to maintain memory in a conversational AI system
- How to use LangGraph to manage state and route between nodes
- How to integrate a vision model for image understanding
- How to use a text-to-speech engine with automatic playback
- How to run large language models locally using Ollama
- How to structure a Python project that uses multiple AI capabilities cleanly

---

## 11. Open Questions and Assumptions

These are things that were not fully decided at the time this document was written. Each assumption should be confirmed before finalizing the implementation.

| Number | Question | Current Assumption |
|---|---|---|
| 1 | Which TTS engine will be used? | pyttsx3 during development, then Kokoro via kokoro-onnx for production. Both are local. |
| 2 | How does the user select a mode? | By typing a keyword or flag at the start of the session |
| 3 | Is session memory saved to disk between runs? | No. Memory only exists while the program is running. |
| 4 | Which image file formats does the vision module support? | PNG, JPG, and JPEG to start |
| 5 | Where are generated images saved? | In an outputs folder at the root of the project |

---

## 12. Example Conversations

These examples show exactly what correct behavior looks like. Use them to verify that your implementation is working as intended.

### Text Memory Across Turns

```
User: I am working on a Rust OCI image puller.
Assistant: That sounds like an interesting project.

User: What project am I working on?
Assistant: You are working on a Rust OCI image puller.
```

### Audio Response Flow

```
You: Explain OCI manifests
Assistant: (Speaking...)
(Audio plays and finishes on its own)
You: What is a layer?
Assistant: (Speaking...)
(Audio plays and finishes on its own)
```

### Image Understanding with Follow-Up

```
User: Describe docker_architecture.png
Assistant: This image shows a Docker architecture diagram...

User: What components are shown?
Assistant: The diagram contains a Docker Client, a Docker Daemon, a Registry, and containers.
```

---

## 13. Notes for GitHub Copilot

If you are reading this file inside VS Code with GitHub Copilot enabled, here is how to get the most out of it:

Place this file in the root of your project and keep it open alongside whatever Python file you are editing. Copilot will use it as context when suggesting code.

Each section in this document maps to a specific Python module:

- Section 5.1 maps to the text conversation module (nodes/text_node.py)
- Section 5.2 maps to the audio response module (nodes/audio_node.py) and the TTS client (models/tts_client.py)
- Section 5.3 maps to the vision module (nodes/vision_node.py)
- Section 5.4 maps to the image generation module (nodes/imagegen_node.py)
- Section 6 maps to the memory classes (memory/)
- Section 7 maps to the router and LangGraph graph definition (router.py)

For the TTS client specifically: start by writing tts_client.py using pyttsx3. Once the audio node is confirmed working, rewrite tts_client.py to use Kokoro. The interface that audio_node.py uses should not change between the two versions. Only the internals of tts_client.py change.

When you start writing a new file, read the matching section first. The behavior descriptions, input and output types, and step-by-step flows are written to help Copilot generate accurate suggestions.

---

*Owner: Dharshan. This document should be updated whenever the design changes.*
