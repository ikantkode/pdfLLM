# pdfLLM - Chat with your PDFs Completely Local!
A core php application meant for a proof of concept. Built completely with Grok 3 without any kind of coding experience. I am literally illiterate in coding. Allows you to chat with your PDFs only or if you want you can chat with PDF + LLM using Ollama. Too poor for APIs so use free stuff.

#### The Actual Goal
There is a lot of data out there in the world that some of us are struggling to comprehend. My vision is to help the world. The students, the curious ones, the tinkerers, the kids, and as a professional in the construction industry - where a lot of non-technical people are, help those people too. The caveat with existing technology? It is not simple. I always like simplicity. I hope I never fade away from it, and I hope I always help people. If this project helps you. Don't do anything. Just be kind to someone.

# Change Log 
*so profressional, ermahgurd*
#### 03/01/2025 - I erased original broken code and uploaded a 100% (i think) fixed code
1. Broken CSS: New Chat button will not require users to refresh. Ajax will take care of that.
2. Uploaded files will now either be saved into b2 or your local drive.
3. Deleting files will delete all instances of that specific file from your database (record entry, pgVector, etc). as well as from your hard drive/b2.
4. Updated set up setupdb.txt and instructions to set up database. Completely forgot that was an important part of sharing this whole thing. lol. Realized and implemented @ 11:00 PM 
# Known issues/thoughts
#### Date: 03/01/2025
1. There is a "loading" button that comes up on the right side of send button. I am being lazy about it.
2. Some texts come back as gibberish/broken words. This is mainly for PDFs that require OCR (we use Tesseract OCR package). Currently LLM is hard-coded to fish out broken words and provide you with proper context. If you have issues, please do share.
3. Text-Embedding. I am not sure what this is, apparently helps you do what we are trying to do much better. I think we are headed towards RAG but meh, I still like this lightweight thing better.
4. Issues with mobile version. Its not even working - I am working on fixing it, I have fixed it 40% - update as of 11:00 PM

# Road Map
I never thought I would be able to do this, but since I can continue, I will - it will remain open source, all of it. I am going to host this app for demo purposes on its own domain.

### Milestones:
1. Move from core-php to something lightweight, but also some framework that is easy to follow.
2. Look into embedding and implement.
3. Allow users to use their APIs (Open AI, DeepSeek).
4. Create an API system
5. Create mobile app with Flutter for both iOS/Android
6. Watch the absolutely amazing and genius Sesame project and implement/play with it when the time arrives
7. Basically add continuous voice-chat model so that users can converse with their own data. Not some generic data from an LLM.

# Who am I? And why did I make this? And how did I make this?
I am shakespear94 from reddit. I used Grok 3 to make this over the course of 2.5 days roughly 30 hours of consecutive work. I have no coding or technical knowledge, but I can follow basic instructions. I found LLMs can provide that. Of course they hallucinate, and of course we cannot make production ready applications just yet. But I don't have time to wait. I don't want to make agentic AI crap, RAG things, stuff I don't even comprehend. I have tried DeepSeek-R1, ChatGPT (Free), and now tried Grok 3 (Free), and Grok 3 was a winner in this case. For a poor man to be able to do this, I feel very hopeful for what lays ahead.

The app was built and tested on Ubuntu 24.04 - 16 GB RAM, a single Nvidia 3060 (Lite Hash - if that matters) 12 GB (VRAM) with an Intel i7 4000 series. Honestly too tired to even care to write the exact model its 5:43 AM as of this. As mentioned somewhere here, I am using Ollama's API, phi4:latest - as of 03/01/2025

# The Future
Depending on hate and rage received, I will continue to update this adventure for giggles. I know there are frameworks that are better, but I am no programmer. I figured someone might find this useful like I did. 

Best of luck everyone, with poor man's love, shakespear. *walk into sunset...*

# Setting Up Postgre
Use your favorite LLM (ChatGPT/DeepSeek, Grok, or whatever. Give it this prompt if you're lazy like me:
**This is the db.php from the code, can you please give me directions to set up postgre, pgadmin4 and pgvector. host is localhost, dbname is pdfchat, user is shake pass is itall1234** - and it should help you out, otherwise start praying to whatver god you believe in. I did the same.

# Set Up Composer
I only know how to do this on linux, so ask an LLM to help you otherwise.

1. Install composer
2. cd into the directory
3. For simplicity run *sudo composer install*
4. It will get you the vendor folder and it is required. Otherwise the app is going to fail.

# Creating Admin User
This is easy. First user is admin. So just go to localhost/index.php (or login.php if you fancy) and use the sign up button. 

# Uploading
*Local uploading is now fully functional (as of 03/01/2025) but you are alternatively able to set up backblaze b2. If you have other requests like S3, or something like that, open an issue and I will have my handy dandy llm help us out. Or if you are a coder, and want to help, feel free.

Local Files are all uploaded to /uploads of your project directory. Be mindful of your hard-drive space. The files are then parsed, saved into pgVector, and you are able to do normal things. Deleting files is going to delete everything related to that file in the database and your local drive.
Go to backblaze, create an account, here is a link: https://www.backblaze.com/sign-up/cloud-storage read up on it. Create a bucket, and then copy it's ID:

![image](https://github.com/user-attachments/assets/9dbc696e-8eff-40ac-b0b1-0d87029617a8)

Then go click the Upload/Download Button, then go to Application Keys. You should be able to generate the keys from there. Copy all the keys required. Paste in settings of your app. There is only one button called Settings.

# Admin Panel
Admin Panel allows you to see and select the model you want all the peasants to use.

# Chat Mode
At the bottom of the app, you are able to select 2 modes, PDF only and PDF + LLM
#### PDF Only 
This mode limits the context to just the PDF. 
#### PDF + LLM**
This mode allows LLM to access the knowledge that is vectorized (some magic through pgVector) and then allows you to converse with it. You can ask it dumb questions like summarize this pdf but how gen-alpha talks. It is so funny lol.

# Set Up / Install
#### Prerequisites
- Use Ubuntu or a debian based flavor. If you are on windows use WSL? Not sure.
- Apache/Nginx
- Composer
- Postgre & pgAdmin4 (web) install.
- General knowledge of how this stuff works.

#### Steps to install
1. Open Terminal -> git clone {git clone repo url}
2. cd pdfLLM
3. composer install (please google how to install it)
4. sudo su postgres (this will login you in as postgres user)
5. psql -U postgres (this will allow you login to postgresSQL, aka psql, and then you can do command magic)
6. CREATE DATABASE chatpdf; (you are able to change the name of the database, just make sure you adjust db.php after)
7. Once the database is created, switch to pgAdmin4 and login.

![418319967-e8d102a7-64ee-45ca-9b0d-1ddba2ce3ae8](https://github.com/user-attachments/assets/95ae55c5-7bd1-423d-8a4b-639b8967e966)

   
8. Paste the entire file from setupdb.txt into the sql center as I call it. Follow my train of thoughts here, if you have never done this before like me, you should do this. Its a good way to learn. Otherwise I dont even know how you got this far.

   ![image](https://github.com/user-attachments/assets/68409207-06ea-4ad3-ae57-716b8963d384)

# Using the App
1. You must upload at least one document. There is an upload button. You can select multiple files.
2. Click New Chat, then refresh your browser. Why? because I cannot code, seriously. Upon refreshing, it will automatically select the chat label.
3. If your PDF is not selected then select it.
4. At the bottom, Chat Mode says PDF Only by default, you can use dropdown menu to select whatever you want.

# Model Recommendation
As if I am intelligent enough in this sector to even write here. Use phi4:latest or any 7B model - but this depends on how much your system can handle. If you have 4GB VRAM, then go for 3B models, or if you have 4GB VRAM = 8 GB RAM, then try 7B models. But truth be told, play with it, its not that serious. I did not try with 1B models because they have an IQ of less than a 5 minute old baby monkey. I know the deal, use case, blah blah. Take your technical knowledge and enjoy. I cry because I don't have it - yet. Kidding, honestly, did not try it, can't tell/recommend.

# Deleting of Files
When a file is deleted, through computer magic, the files are moved to deleted folder in b2, and then after that deleted (i have not fully tested this so consider this a warning, don't go crazy). The vector data and database are wiped of the file's existence as well. But as I am not a coder I cannot test that either. I will try to do this when my brain recovers from doing all this.
