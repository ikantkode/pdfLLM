# pdfLLM
A core php application meant for a proof of concept. Built completely with Grok 3 without any kind of coding experience. I am literally illiterate in coding. Allows you to chat with your PDFs only or if you want you can chat with PDF + LLM using Ollama. Too poor for APIs so use free stuff.

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
*Local Uploading is going to be completed tomorrow. CSS and Deletion aren't working, so I am opting to not include it at the moment. But if you must experiment, there is an index2.php, and process2.php - basically, remove index.php and process.php and rename index2 to index.php and process2 to process.php - Best O' luck, because my brain is fried and dried.
Go to backblaze, create an account, here is a link: https://www.backblaze.com/sign-up/cloud-storage read up on it. Create a bucket, and then copy it's ID:

![image](https://github.com/user-attachments/assets/9dbc696e-8eff-40ac-b0b1-0d87029617a8)

Then go click the Upload/Download Button, then go to Application Keys. You should be able to generate the keys from there. Copy all the keys required. Paste in settings of your app. There is only one button called Settings.

# Admin Panel
Admin Panel allows you to see and select the model you want all the peasants to use.

# Chat Mode
At the bottom of the app, you are able to select 2 modes, PDF only and PDF + LLM
**PDF Only** 
This mode limits the context to just the PDF. 
**PDF + LLM**
This mode allows LLM to access the knowledge that is vectorized (some magic through pgVector) and then allows you to converse with it. You can ask it dumb questions like summarize this pdf but how gen-alpha talks. It is so funny lol.

# Using the App
1. You must upload at least one document. There is an upload button. You can select multiple files.
2. Click New Chat, then refresh your browser. Why? because I cannot code, seriously. Upon refreshing, it will automatically select the chat label.
3. If your PDF is not selected then select it.
4. At the bottom, Chat Mode says PDF Only by default, you can use dropdown menu to select whatever you want.

# Model Recommendation
As if I am intelligent enough in this sector to even write here. Use phi4:latest or any 7B model - but this depends on how much your system can handle. If you have 4GB VRAM, then go for 3B models, or if you have 4GB VRAM = 8 GB RAM, then try 7B models. But truth be told, play with it, its not that serious. I did not try with 1B models because they have an IQ of less than a 5 minute old baby monkey. I know the deal, use case, blah blah. Take your technical knowledge and enjoy. I cry because I don't have it - yet. Kidding, honestly, did not try it, can't tell/recommend.

# Deleting of Files
When a file is deleted, through computer magic, the files are moved to deleted folder in b2, and then after that deleted (i have not fully tested this so consider this a warning, don't go crazy). The vector data and database are wiped of the file's existence as well. But as I am not a coder I cannot test that either. I will try to do this when my brain recovers from doing all this.
