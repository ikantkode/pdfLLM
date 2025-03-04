# pdfLLM - Chat with your PDFs using Ollama

### About pdfLLM

pdfLLM is a proof-of-concept Core PHP application designed to enable users to interact with their PDFs locally. It supports both PDF-only interactions and PDF + LLM conversations using Ollama. Built entirely with Grok 3, this project demonstrates how accessible AI-powered solutions can be, even for those without extensive coding experience. The goal is to provide a simple yet effective tool for students, professionals, and anyone who needs better insights from their documents.

### Vision

The world is filled with data, but comprehension remains a challenge for many. This project aims to make information more accessible, particularly for students, curious minds, tinkerers, and professionals in industries like construction, where technical expertise may be limited. Existing solutions can be complex; simplicity is the core principle of pdfLLM. If this tool benefits you, simply pay it forward with kindness.

### Change Log

#### 03/01/2025

1. Fixed broken CSS: The "New Chat" button now updates via AJAX without requiring a page refresh. 
2. Uploaded files are now saved either to Backblaze B2 or a local directory.
3. File deletions remove all instances of the file from the database, pgVector, and storage.
4. Updated setup instructions in setupdb.txt to ensure smooth database configuration.

#### 03/02/2025

1. Automatic scrolling to the latest response in chat.
2. Removed "Moderator" account creation in the admin panel.
3. Improved UI for a better user experience.
4. PDF preview functionality now opens in a new tab.

#### 03/04/2025

1. Now able to use embedding model (mxbai-embed-large:latest) - currently hard coded.
2. We (Grok & I), tried to refine the OCR Scanning and Text Extractions. Honestly tried so hard, but we can only get like 90% accuracy.

#### Known Issues

1. Some answers are still not very accurate. But I am not giving up.
2. Grammar errors. I am trying to figure out whether this is an OCR/Text Extraction issue or LLM Hallucination issue. 
3. I noticed the hallucination because I switched from mistral:7b-instruct-v0.3-q4_0 to llama3.2:3b - and immediately felt the difference. Our system prompt is pretty strict, but things still slip.

#### Thoughts

1. I like the DeepSeek's Thinking Approach, so I have independentally thought of and begun implementation of "Grammar Check". The idea is, it is going to stream a response and then grammar check, then give the response again. We will have the same approach as R1 where you have a think bubble and then the actual answer - except it will all be context based.
2. This will make 2 API calls, but at least it will fix the broken words. Today, I understand why PDFs are so yucky. It does not even make sense that this format is alive. The world should just switch to markdown /s (seriously sarcasm, I know PDFs are king in some areas and I wouldn't replace them).

#### Roadmap

pdfLLM will remain open-source. A demo version will be hosted for broader accessibility.

##### Upcoming Milestones:

1. Migrate from Core PHP to a lightweight, easy-to-follow framework.
2. Implement embeddings for improved search and retrieval. - done 03/04/2025 - I worked on this all day and thought, why did i not do anything on 03/03/2025 - its 1:20 AM lol.
3. Enable users to integrate their own API keys (OpenAI, DeepSeek, etc.).
4. Develop an API system.
5. Build a Flutter-based mobile app for iOS and Android.
6. Explore the Sesame project and integrate advanced features when it is released..
7. Implement continuous voice-chat functionality for interacting with personal data/PDFs/other formats in the future pls?


# Prerequisites:

- OS: Ubuntu (or Debian-based distro), WSL for Windows
- Web Server: Apache/Nginx
- Database: PostgreSQL & pgAdmin4
- PHP Package Manager: Composer

#### Installation Steps:

1. Clone the repository: git clone {repo_url}
2. Navigate to the project directory: cd pdfLLM
3. Install dependencies: composer install
4. Configure PostgreSQL:
5. sudo su postgres
6. psql -U postgres
7. Create the database: CREATE DATABASE chatpdf;
8. Import schema: Copy and paste the contents of setupdb.txt into pgAdmin4's SQL editor and execute.
9. Set up environment variables in db.php.

# Usage Guide

#### Initial Setup

Note: Admin Account: First registered user automatically becomes an admin.

1. sudo apt install poppler-utils (this is for pdftotext)
#### Uploading Files:

- Local uploads are stored in /uploads.
- Backblaze B2 integration available for cloud storage.

#### Chat Modes:

- PDF Only: Restricts responses to document content.
- PDF + LLM: Uses vectorized search with LLM assistance.

# Model Recommendations:

- Phi-4 (latest) or 7B models for best balance of performance and accuracy - although, testing was limited to 14B.

- Systems with 4GB VRAM should use 3B models.

- 1B models are not recommended due to limited reasoning capabilities - but feel free to try.

# File Management

- Deleting Files
When a file is deleted:
- It is moved to a "deleted" folder in Backblaze B2 (if enabled) before permanent deletion.
- Corresponding vector embeddings and database records are also removed.

*This feature is still under testing, so use with caution.*

# Future Considerations

The development and updates will depend on community interest and feedback. While pdfLLM is not a polished commercial product, it represents a step toward democratizing AI-assisted document interaction.

For those who find value in this project—use it, improve it, or share it.

With best wishes,
ShakeSpear94