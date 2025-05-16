# RAG-Math Bot: An Intelligent Math Tutoring System

## Overview
RAG-Math Bot is an advanced math tutoring system that uses Retrieval-Augmented Generation (RAG) and Large Language Models (LLMs) to provide intelligent, personalized math tutoring. The system can understand complex mathematical questions, break them down into manageable steps, and provide detailed explanations.

## Problem Statement

### Current Challenges in Math Education:

1. **Accessibility Issues:**
   - Limited access to qualified math tutors
   - High costs of private tutoring
   - Geographic limitations
   - Time constraints for both students and tutors

2. **Learning Gaps:**
   - One-size-fits-all teaching approach
   - Difficulty in identifying individual learning needs
   - Lack of personalized feedback
   - Limited practice opportunities

3. **Resource Limitations:**
   - Scarcity of quality math learning materials
   - Difficulty in finding relevant examples
   - Limited availability of step-by-step solutions
   - Lack of immediate feedback

4. **Engagement Problems:**
   - Traditional methods can be monotonous
   - Limited interaction and engagement
   - Difficulty in maintaining student interest
   - Lack of real-time problem-solving guidance

## How RAG-Math Bot Addresses These Challenges

1. **Personalized Learning:**
   - Breaks down complex problems into manageable steps
   - Provides tailored explanations based on the question
   - Adapts to different learning styles through multiple LLM models
   - Offers step-by-step reasoning for better understanding

2. **Enhanced Accessibility:**
   - Available 24/7 for immediate assistance
   - No geographical limitations
   - Free to use (requires only Ollama setup)
   - Works with various document formats (PDF, TXT, MD)

3. **Comprehensive Learning Support:**
   - Processes and understands mathematical content from uploaded documents
   - Provides detailed explanations with examples
   - Shows reasoning process for each step
   - Maintains context awareness for better understanding

4. **Interactive Learning Experience:**
   - Real-time chat interface for natural interaction
   - Step-by-step solution display
   - Multiple model options for different learning styles
   - Ability to upload and reference multiple learning materials

## Application Flow

### 1. Initial Setup & Document Processing Flow:
```
User Uploads Document → Temporary Storage → Document Processing → Vector Store
```
- User uploads PDF/TXT/MD files through the sidebar
- Files are temporarily stored
- Documents are processed by `VectorstoreManager`:
  - Split into chunks (1200 tokens with 80 token overlap)
  - Converted to embeddings using Nomic-embed-text
  - Stored in FAISS vector store
- Vector store is saved persistently

### 2. Question Processing Flow:
```
User Question → Break Down → Context Retrieval → Calculation → Summary → Response
```

#### A. Question Breakdown:
```python
Input: User's question
Process:
1. Analyze question using LLM
2. Break into 2-3 specific instructions
3. Remove meta-instructions
Output: List of actionable instructions
```

#### B. Context Retrieval:
```python
Input: Question + Instructions
Process:
1. Combine question and instructions into query
2. Search vector store (TOP_K = 3)
3. Retrieve relevant document chunks
Output: Combined context from documents
```

#### C. Calculation:
```python
Input: Context + Instructions
Process:
1. For each instruction:
   - Use LLM to process instruction
   - Show step-by-step work
   - Generate detailed explanation
Output: Dictionary of calculations per instruction
```

#### D. Summary Generation:
```python
Input: Calculations + Question + Chat History
Process:
1. Combine all information
2. Generate comprehensive answer
3. Format in markdown
Output: Final answer with reasoning
```

### 3. UI Flow:
```
Sidebar:
├── Model Selection
│   ├── Qwen 2.5 (7B)
│   ├── Llama 3 (8B)
│   ├── Mistral (7B)
│   └── DeepSeek R1 (8B)
│
└── Document Management
    ├── File Upload
    ├── Processing Status
    └── Success/Error Messages

Main Interface:
├── Chat History
│   ├── User Messages
│   └── Assistant Responses
│
└── Chat Input
    └── Reasoning Process (Expandable)
        ├── Instructions
        └── Calculations
```

### 4. State Management:
```python
State Class:
├── messages: List[Dict[str, str]]
├── current_question: str
├── instructions: List[str]
├── context: str
├── calculations: Dict[str, Any]
└── summary: str
```

### 5. Error Handling Flow:
```
Error Detection → Error Logging → User Notification → Recovery
```
- File upload errors
- Document processing errors
- LLM response errors
- Vector store errors

### 6. Model Selection Flow:
```
Model Change → State Update → Chat History Preservation → New Model Initialization
```

### 7. Document Management Flow:
```
File Upload → Validation → Processing → Vector Store Update → Cleanup
```

### 8. Chat History Management:
```
New Message → State Update → Display → History Preservation
```

## Key Features

1. **Persistence:**
   - Vector store persistence
   - Chat history preservation
   - Model selection memory

2. **Real-time Processing:**
   - Immediate file processing
   - Quick response generation
   - Live status updates

3. **Error Recovery:**
   - Graceful error handling
   - User-friendly error messages
   - Automatic cleanup

4. **User Experience:**
   - Clear progress indicators
   - Expandable reasoning
   - Multiple model options
   - Easy document management

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Set up Ollama:
   - Install Ollama from https://ollama.ai
   - Pull required models:
     ```bash
     ollama pull qwen2.5:7b
     ollama pull llama3:8b
     ollama pull mistral
     ollama pull deepseek-r1:8b
     ```

## Usage

1. Start the application:
```bash
streamlit run streamlit_advanced_rag.py
```

2. Upload documents through the sidebar ( python create_vectorstore.py vector_storage files_storage -d)
3. Select your preferred LLM model
4. Start asking questions!

## Requirements

- Python 3.8+
- Streamlit
- LangChain
- FAISS
- Ollama
- PyPDF2
- Other dependencies listed in requirements.txt

## License

[Your chosen license]

## Contributing

[Contribution guidelines]

## Contact

[Your contact information] 