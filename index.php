<?php
session_start();
error_reporting(E_ALL);
ini_set('display_errors', 1);

// Function to check Ollama status
function checkOllamaStatus() {
    $url = 'http://192.168.0.101:11434'; // Ollama server URL
    $ch = curl_init($url);
    curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
    curl_setopt($ch, CURLOPT_TIMEOUT, 2); // Timeout after 2 seconds
    $response = curl_exec($ch);
    $httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
    curl_close($ch);

    // Check if the response is valid
    if ($httpCode === 200) {
        return "Connected";
    } else {
        error_log("Ollama status check failed. HTTP Code: $httpCode, Response: $response");
        return "Not running";
    }
}

// Check Ollama status
$ollamaStatus = checkOllamaStatus();
?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>ChatPDF Clone</title>
    <style>
        .chat-box { height: 300px; overflow-y: scroll; border: 1px solid #ccc; padding: 10px; }
        .message { margin: 5px 0; }
        .status { margin: 10px 0; }
        .input-group { margin-bottom: 10px; }
        #link-input { display: none; }
        pre code {
            background-color: #f4f4f4;
            padding: 10px;
            border-radius: 5px;
            display: block;
            white-space: pre-wrap;
        }
        ul {
            list-style-type: disc;
            margin-left: 20px;
        }
    </style>
</head>
<body>
    <h1>Chat with Your PDF</h1>
    
    <form id="upload-form" enctype="multipart/form-data">
        <div class="input-group">
            <label>
                <input type="radio" name="input-type" value="file" checked> Upload PDF
            </label>
            <label>
                <input type="radio" name="input-type" value="link"> Paste Cloud Link
            </label>
        </div>
        
        <div id="file-input">
            <input type="file" name="pdf" accept=".pdf">
        </div>
        
        <div id="link-input">
            <input type="text" name="cloud-link" placeholder="Paste your cloud storage link (AWS S3, Backblaze B2, etc.)">
        </div>
        
        <button type="submit">Submit</button>
    </form>
    
    <div class="status" id="pdf-status">
        <?php
        if (isset($_SESSION['pdf_name'])) {
            echo "Current PDF: " . htmlspecialchars($_SESSION['pdf_name']);
        } else {
            echo "No PDF uploaded yet.";
        }
        ?>
    </div>
    
    <div class="status" id="ollama-status" style="color: <?php echo $ollamaStatus === 'Connected' ? 'green' : 'red'; ?>">
        Ollama: <?php echo $ollamaStatus; ?>
    </div>
    
    <div class="chat-box" id="chat-box"></div>
    <input type="text" id="user-input" placeholder="Ask something about the PDF">
    <button onclick="sendMessage()">Send</button>

    <script>
        // Toggle between file upload and link input
        document.querySelectorAll('input[name="input-type"]').forEach(input => {
            input.addEventListener('change', function() {
                const fileInput = document.querySelector('input[name="pdf"]');
                const linkInput = document.querySelector('input[name="cloud-link"]');

                if (this.value === 'file') {
                    document.getElementById('file-input').style.display = 'block';
                    document.getElementById('link-input').style.display = 'none';
                    fileInput.setAttribute('required', true); // Add required to file input
                    linkInput.removeAttribute('required');     // Remove required from link input
                } else {
                    document.getElementById('file-input').style.display = 'none';
                    document.getElementById('link-input').style.display = 'block';
                    fileInput.removeAttribute('required');      // Remove required from file input
                    linkInput.setAttribute('required', true); // Add required to link input
                }
            });
        });

        // Handle form submission
        document.getElementById('upload-form').addEventListener('submit', async function(e) {
            e.preventDefault();
            const formData = new FormData(this);
            const pdfStatus = document.getElementById('pdf-status');
            pdfStatus.textContent = "Processing...";

            try {
                const response = await fetch('/chatpdf/process.php', {
                    method: 'POST',
                    body: formData
                });
                const result = await response.json();
                if (result.success) {
                    pdfStatus.textContent = "Current PDF: " + result.pdf_name;
                } else {
                    pdfStatus.textContent = "Upload failed: " + result.error;
                }
            } catch (err) {
                pdfStatus.textContent = "Error: " + err.message;
            }
        });

        // Handle chat messages
        async function sendMessage() {
            const input = document.getElementById('user-input').value;
            const chatBox = document.getElementById('chat-box');
            chatBox.innerHTML += `<div class="message"><b>You:</b> ${input}</div>`;
            try {
                const response = await fetch('/chatpdf/chat.php', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                    body: `message=${encodeURIComponent(input)}`
                });
                const reply = await response.text();
                chatBox.innerHTML += `<div class="message"><b>AI:</b> ${reply}</div>`;
            } catch (err) {
                chatBox.innerHTML += `<div class="message"><b>AI:</b> Error: ${err.message}</div>`;
            }
            chatBox.scrollTop = chatBox.scrollHeight;
            document.getElementById('user-input').value = '';
        }
    </script>
</body>
</html>