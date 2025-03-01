<?php
session_start();
require 'db.php';
require 'vendor/autoload.php';

ini_set('display_errors', 0);
ini_set('log_errors', 1);
ini_set('error_log', '/var/www/html/chatpdf/php_errors.log');

if (!isset($_SESSION['user_id'])) {
    error_log("No user_id in session, redirecting to login.");
    header("Location: login.php");
    exit;
}

$userId = $_SESSION['user_id'];
error_log("User ID: $userId");

try {
    $stmt = $pdo->prepare("SELECT username FROM users WHERE id = ?");
    $stmt->execute([$userId]);
    $user = $stmt->fetch();
    $username = $user ? $user['username'] : 'Unknown User';
    error_log("Username: $username");
} catch (Exception $e) {
    error_log("User fetch error: " . $e->getMessage());
    $username = 'Unknown User';
}

try {
    $stmt = $pdo->prepare("SELECT id, file_name, vectorized FROM pdfs WHERE user_id = ? ORDER BY uploaded_at DESC");
    $stmt->execute([$userId]);
    $pdfs = $stmt->fetchAll(PDO::FETCH_ASSOC);
    error_log("PDFs fetched: " . json_encode($pdfs));
} catch (Exception $e) {
    error_log("PDF fetch error: " . $e->getMessage());
    $pdfs = [];
}

try {
    $stmt = $pdo->prepare("SELECT id, title, selected_pdf_ids FROM chats WHERE user_id = ? ORDER BY created_at DESC");
    $stmt->execute([$userId]);
    $chats = $stmt->fetchAll(PDO::FETCH_ASSOC);
    error_log("Chats fetched: " . json_encode($chats));
} catch (Exception $e) {
    error_log("Chats fetch error: " . $e->getMessage());
    $chats = [];
}

try {
    $stmt = $pdo->prepare("SELECT value FROM settings WHERE key = 'ollama_model' LIMIT 1");
    $stmt->execute();
    $defaultModel = $stmt->fetchColumn() ?: 'mistral:7b-instruct-v0.3-q4_0';
    $stmt = $pdo->prepare("SELECT value FROM settings WHERE key = 'allow_user_model_select' LIMIT 1");
    $stmt->execute();
    $allowUserModelSelect = $stmt->fetchColumn() === 'yes';
    error_log("Default Model: $defaultModel, Allow User Model Select: " . ($allowUserModelSelect ? 'yes' : 'no'));
} catch (Exception $e) {
    error_log("Settings fetch error: " . $e->getMessage());
    $defaultModel = 'mistral:7b-instruct-v0.3-q4_0';
    $allowUserModelSelect = false;
}

$currentModel = $_SESSION['user_selected_model'] ?? $defaultModel;
$currentChatId = $_SESSION['current_chat_id'] ?? ($chats ? $chats[0]['id'] : null);
$selectedPdfIds = $_SESSION['selected_pdf_ids'] ?? array_column($pdfs, 'id');
$chatMode = $_SESSION['chat_mode'] ?? 'pdf-only';
error_log("Current Model: $currentModel, Chat ID: $currentChatId, Selected PDFs: " . json_encode($selectedPdfIds) . ", Chat Mode: $chatMode");

$client = new GuzzleHttp\Client();
$modelNames = [$defaultModel];
try {
    $response = $client->request('GET', 'http://192.168.0.101:11434/api/tags', ['timeout' => 5]);
    $ollamaModels = json_decode($response->getBody(), true)['models'] ?? [];
    $modelNames = array_column($ollamaModels, 'name');
    error_log("Available Ollama models: " . json_encode($modelNames));
} catch (Exception $e) {
    error_log("Ollama models fetch error: " . $e->getMessage());
}

$chatNumbers = array_map(function($chat) {
    return (int)preg_replace('/^Chat (\d+)$/', '$1', $chat['title']);
}, $chats);
$nextChatNumber = $chatNumbers ? max($chatNumbers) + 1 : 1;
?>
<!DOCTYPE html>
<html>
<head>
    <title>ChatPDF</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bulma@1.0.2/css/bulma.min.css">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { margin: 0; height: 100vh; overflow: hidden; }
        .chat-container { display: flex; height: 100vh; }
        .sidebar { position: absolute; width: 320px; height: 100%; padding: 1.5rem; background: #f5f5f5; overflow-y: auto; z-index: 1; }
        .chat-area { margin-left: 320px; flex-grow: 1; display: flex; flex-direction: column; }
        .chat-box { flex-grow: 1; padding: 1.5rem; overflow-y: auto; position: relative; }
        .chat-input { padding: 1rem; background: #fafafa; border-top: 1px solid #dbdbdb; }
        .message { margin: 1rem 0; padding: 1rem; border-radius: 6px; }
        .user-message { background: #3273dc; color: white; margin-left: auto; max-width: 80%; }
        .llm-response { background: #f0f4f8; border-left: 4px solid #3273dc; max-width: 80%; }
        .upload-queue { margin-top: 1rem; }
        .upload-item { display: flex; align-items: center; margin-bottom: 0.5rem; gap: 0.5rem; }
        .pdf-name-button { white-space: normal; text-align: left; overflow-wrap: break-word; word-wrap: break-word; max-width: 200px; height: auto; padding: 0.5rem; }
        .scroll-to-bottom { position: sticky; bottom: 1rem; right: 1rem; float: right; }
        .is-focused { background-color: #e0e0e0; }
        .chat-start { display: flex; justify-content: center; align-items: center; height: 100%; }
        .edit-chat-btn { margin-left: 5px; }
        @media (max-width: 768px) {
            .chat-container { flex-direction: column; }
            .sidebar { position: relative; width: 100%; max-width: 100%; height: auto; }
            .chat-area { margin-left: 0; }
            .chat-box { padding: 1rem; }
            .chat-input { padding: 0.5rem; }
            .pdf-name-button { max-width: 100%; }
        }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="sidebar">
            <h4 class="title is-4 has-text-centered-mobile">Welcome, <?php echo htmlspecialchars($username); ?>!</h4>

            <div class="box">
                <h5 class="title is-5">Upload PDFs</h5>
                <form id="upload-form" enctype="multipart/form-data">
                    <div class="file has-name is-fullwidth">
                        <label class="file-label">
                            <input class="file-input" type="file" name="pdfs[]" accept=".pdf" multiple required>
                            <span class="file-cta">
                                <span class="file-label">Choose PDFs...</span>
                            </span>
                            <span class="file-name">No files selected</span>
                        </label>
                    </div>
                    <button class="button is-primary is-fullwidth mt-2" type="submit">Start Upload</button>
                </form>
                <div id="upload-queue" class="upload-queue"></div>
            </div>

            <div class="box">
                <h5 class="title is-5">Chats</h5>
                <form id="new-chat-form" class="field has-addons">
                    <div class="control is-expanded">
                        <input class="input" type="text" id="chat-title" placeholder="Enter chat title (optional)">
                    </div>
                    <div class="control">
                        <button class="button is-success" id="new-chat">New</button>
                    </div>
                </form>
                <ul id="chat-list" class="mt-2">
                    <?php if (empty($chats)): ?>
                        <li class="notification">No chats available.</li>
                    <?php else: ?>
                        <?php foreach ($chats as $chat): ?>
                            <li class="is-flex is-align-items-center mb-2">
                                <button class="button is-text chat-btn mr-2 is-fullwidth-mobile <?php echo $chat['id'] === $currentChatId ? 'is-focused' : ''; ?>" data-chat-id="<?php echo $chat['id']; ?>">
                                    <?php echo htmlspecialchars($chat['title']); ?>
                                </button>
                                <button class="button is-small edit-chat-btn" data-chat-id="<?php echo $chat['id']; ?>">Edit</button>
                                <button class="button is-danger is-small delete-chat-btn" data-chat-id="<?php echo $chat['id']; ?>">Delete</button>
                            </li>
                        <?php endforeach; ?>
                    <?php endif; ?>
                </ul>
            </div>

            <div class="box">
                <h5 class="title is-5">Your PDFs</h5>
                <ul id="pdf-list">
                    <?php if (empty($pdfs)): ?>
                        <li class="notification">No PDFs available.</li>
                    <?php else: ?>
                        <?php foreach ($pdfs as $pdf): ?>
                            <?php $originalName = str_replace('_', ' ', pathinfo($pdf['file_name'], PATHINFO_FILENAME)) . '.' . pathinfo($pdf['file_name'], PATHINFO_EXTENSION); ?>
                            <li class="is-flex is-align-items-center mb-2 is-flex-wrap-wrap">
                                <input class="checkbox mr-2 pdf-checkbox" type="checkbox" data-pdf-id="<?php echo $pdf['id']; ?>" <?php echo in_array($pdf['id'], $selectedPdfIds) ? 'checked' : ''; ?>>
                                <button class="button is-text mr-2 is-fullwidth-mobile pdf-name-button"><?php echo htmlspecialchars($originalName); ?></button>
                                <?php if ($pdf['vectorized']): ?><span class="tag is-success mr-2">Vectorized</span><?php endif; ?>
                                <button class="button is-danger is-small delete-btn" data-pdf-id="<?php echo $pdf['id']; ?>">Delete</button>
                            </li>
                        <?php endforeach; ?>
                    <?php endif; ?>
                </ul>
            </div>

            <div class="buttons is-centered">
                <a href="logout.php" class="button is-light">Logout</a>
                <a href="settings.php" class="button is-info">Settings</a>
                <?php if ($_SESSION['role'] === 'admin'): ?>
                    <a href="admin.php" class="button is-warning">Admin Panel</a>
                <?php endif; ?>
            </div>
        </div>

        <div class="chat-area" id="chat-area">
            <?php if (empty($chats)): ?>
                <div id="chat-content" class="chat-box">
                    <div class="chat-start">
                        <button class="button is-primary is-large" id="start-new-chat">New Chat</button>
                    </div>
                </div>
            <?php else: ?>
                <div id="chat-content" class="chat-box">
                    <div id="chat-box"></div>
                    <button id="scroll-to-bottom" class="button is-small is-info scroll-to-bottom is-hidden">Go to Latest</button>
                </div>
                <div class="chat-input">
                    <div class="field has-addons">
                        <div class="control is-expanded">
                            <input class="input" type="text" id="user-input" placeholder="Ask something about your PDFs...">
                        </div>
                        <div class="control">
                            <button class="button is-primary" id="send-btn">Send</button>
                        </div>
                        <div class="control">
                            <span id="spinner" class="is-hidden has-text-grey">Loading...</span>
                        </div>
                    </div>
                    <div class="field is-grouped is-grouped-multiline mt-2">
                        <div class="control">
                            <label class="label">Chat Mode:</label>
                            <div class="select">
                                <select id="chat-mode">
                                    <option value="pdf-only" <?php echo $chatMode === 'pdf-only' ? 'selected' : ''; ?>>PDF Only</option>
                                    <option value="mixed" <?php echo $chatMode === 'mixed' ? 'selected' : ''; ?>>PDF + LLM</option>
                                </select>
                            </div>
                        </div>
                        <div class="control">
                            <label class="label">Ollama Connected:</label>
                            <?php if ($allowUserModelSelect): ?>
                                <div class="select">
                                    <select id="model-select">
                                        <?php foreach ($modelNames as $model): ?>
                                            <option value="<?php echo $model; ?>" <?php echo $model === $currentModel ? 'selected' : ''; ?>><?php echo $model; ?></option>
                                        <?php endforeach; ?>
                                    </select>
                                </div>
                            <?php else: ?>
                                <span class="tag is-primary"><?php echo $currentModel; ?></span>
                            <?php endif; ?>
                        </div>
                    </div>
                </div>
            <?php endif; ?>
        </div>
    </div>

    <script>
        let selectedPdfIds = <?php echo json_encode($selectedPdfIds); ?>;
        let uploadQueue = [];
        let isUploading = false;
        let currentChatId = <?php echo json_encode($currentChatId); ?>;
        let isGenerating = false;
        let nextChatNumber = <?php echo $nextChatNumber; ?>;

        document.addEventListener('DOMContentLoaded', () => {
            if (<?php echo count($chats) > 0 ? 'true' : 'false'; ?>) {
                currentChatId = <?php echo json_encode($chats ? $chats[0]['id'] : null); ?>;
                loadChat(currentChatId);
                setupChatEvents();
            } else if (document.getElementById('start-new-chat')) {
                document.getElementById('start-new-chat').addEventListener('click', createNewChat);
            }
            updateChatList();
        });

        document.getElementById('upload-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            const files = e.target.querySelector('input[type="file"]').files;
            if (files.length > 20) {
                alert('You can upload a maximum of 20 files at once.');
                return;
            }
            const queueDiv = document.getElementById('upload-queue');
            queueDiv.innerHTML = '';

            uploadQueue = Array.from(files).map(file => ({
                file,
                cancelled: false,
                originalName: file.name
            }));

            uploadQueue.forEach((item, index) => {
                const div = document.createElement('div');
                div.className = 'upload-item';
                div.innerHTML = `
                    <span class="is-size-6">${item.originalName}</span>
                    <progress class="progress is-primary is-small" id="progress-${index}" value="0" max="100"></progress>
                    <button class="button is-warning is-small cancel-btn" data-index="${index}">Cancel</button>
                    <span id="tick-${index}" class="is-hidden has-text-success">✔</span>
                `;
                queueDiv.appendChild(div);
            });

            if (!isUploading) processQueue();
        });

        async function processQueue() {
            if (uploadQueue.length === 0 || isUploading) return;
            isUploading = true;

            for (let i = 0; i < uploadQueue.length; i++) {
                const item = uploadQueue[i];
                if (item.cancelled) continue;

                const formData = new FormData();
                formData.append('pdf', item.file);
                const progressBar = document.getElementById(`progress-${i}`);
                const statusSpan = document.getElementById(`tick-${i}`);

                try {
                    const response = await fetch('process.php', {
                        method: 'POST',
                        body: formData
                    });
                    const text = await response.text();
                    console.log('Process response:', text);
                    const result = JSON.parse(text);
                    if (result.success) {
                        progressBar.value = 100;
                        progressBar.className = 'progress is-success is-small';
                        statusSpan.className = 'has-text-success';
                        selectedPdfIds.push(result.pdf_id);
                        updatePdfList();
                    } else {
                        progressBar.className = 'progress is-danger is-small';
                        console.error('Upload failed:', result.error);
                    }
                } catch (e) {
                    console.error('Process fetch error:', e);
                    progressBar.className = 'progress is-danger is-small';
                }
            }

            isUploading = false;
            saveChatContext();
        }

        document.querySelectorAll('.cancel-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const index = btn.dataset.index;
                uploadQueue[index].cancelled = true;
                btn.disabled = true;
                document.getElementById(`progress-${index}`).className = 'progress is-danger is-small';
            });
        });

        document.getElementById('pdf-list').addEventListener('change', (e) => {
            if (e.target.classList.contains('pdf-checkbox')) {
                updateSelectedPdfs();
            }
        });

        function updateSelectedPdfs() {
            selectedPdfIds = Array.from(document.querySelectorAll('.pdf-checkbox:checked'))
                .map(cb => cb.dataset.pdfId);
            if (currentChatId) {
                fetch('chat.php', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                    body: `update_pdfs=1&chat_id=${currentChatId}&pdf_ids=${selectedPdfIds.join(',')}`
                }).then(response => response.text())
                  .then(text => console.log('Update PDFs response:', text))
                  .catch(e => console.error('Update PDFs error:', e));
            }
            saveChatContext();
        }

        document.querySelectorAll('.delete-btn').forEach(btn => {
            btn.addEventListener('click', async () => {
                if (confirm('Are you sure you want to delete this PDF?')) {
                    const response = await fetch('process.php', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                        body: `delete_id=${btn.dataset.pdfId}`
                    });
                    const text = await response.text();
                    console.log('Delete PDF response:', text);
                    try {
                        const result = JSON.parse(text);
                        if (result.success) {
                            selectedPdfIds = selectedPdfIds.filter(id => id !== btn.dataset.pdfId);
                            saveChatContext();
                            updatePdfList();
                        } else {
                            console.error('Delete PDF failed:', result.error);
                        }
                    } catch (e) {
                        console.error('Delete PDF JSON parse error:', e, 'Response:', text);
                    }
                }
            });
        });

        async function createNewChat() {
            let title = document.getElementById('chat-title').value.trim();
            if (!title) {
                title = `Chat ${nextChatNumber++}`;
            }
            const response = await fetch('chat.php', {
                method: 'POST',
                headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                body: `new_chat=1&title=${encodeURIComponent(title)}&pdf_ids=${selectedPdfIds.join(',')}`
            });
            const text = await response.text();
            console.log('New chat response:', text);
            try {
                const result = JSON.parse(text);
                if (result.success) {
                    currentChatId = result.chat_id;
                    saveChatContext();

                    // Update the chat list synchronously
                    await updateChatList();

                    // Find and activate the new chat
                    const newChatBtn = document.querySelector(`.chat-btn[data-chat-id="${currentChatId}"]`);
                    if (newChatBtn) {
                        document.querySelectorAll('.chat-btn').forEach(btn => btn.classList.remove('is-focused'));
                        newChatBtn.classList.add('is-focused');
                        newChatBtn.click(); // Trigger the chat load immediately
                    } else {
                        console.error('New chat button not found after creation');
                        // Fallback: manually load the chat
                        loadChat(currentChatId);
                        const chatArea = document.querySelector('.chat-area');
                        chatArea.innerHTML = `
                            <div id="chat-content" class="chat-box">
                                <div id="chat-box"></div>
                                <button id="scroll-to-bottom" class="button is-small is-info scroll-to-bottom is-hidden">Go to Latest</button>
                            </div>
                            <div class="chat-input">
                                <div class="field has-addons">
                                    <div class="control is-expanded">
                                        <input class="input" type="text" id="user-input" placeholder="Ask something about your PDFs...">
                                    </div>
                                    <div class="control">
                                        <button class="button is-primary" id="send-btn">Send</button>
                                    </div>
                                    <div class="control">
                                        <span id="spinner" class="is-hidden has-text-grey">Loading...</span>
                                    </div>
                                </div>
                                <div class="field is-grouped is-grouped-multiline mt-2">
                                    <div class="control">
                                        <label class="label">Chat Mode:</label>
                                        <div class="select">
                                            <select id="chat-mode">
                                                <option value="pdf-only" ${'<?php echo $chatMode; ?>' === 'pdf-only' ? 'selected' : ''}>PDF Only</option>
                                                <option value="mixed" ${'<?php echo $chatMode; ?>' === 'mixed' ? 'selected' : ''}>PDF + LLM</option>
                                            </select>
                                        </div>
                                    </div>
                                    <div class="control">
                                        <label class="label">Ollama Connected:</label>
                                        <?php if ($allowUserModelSelect): ?>
                                            <div class="select">
                                                <select id="model-select">
                                                    <?php foreach ($modelNames as $model): ?>
                                                        <option value="<?php echo $model; ?>" <?php echo $model === $currentModel ? 'selected' : ''; ?>><?php echo $model; ?></option>
                                                    <?php endforeach; ?>
                                                </select>
                                            </div>
                                        <?php else: ?>
                                            <span class="tag is-primary"><?php echo $currentModel; ?></span>
                                        <?php endif; ?>
                                    </div>
                                </div>
                            </div>
                        `;
                        setupChatEvents();
                    }
                } else {
                    console.error('New chat failed:', result.error);
                }
            } catch (e) {
                console.error('New chat JSON parse error:', e, 'Response:', text);
            }
            document.getElementById('chat-title').value = '';
        }

        document.getElementById('new-chat').addEventListener('click', createNewChat);

        async function updateChatList() {
            const response = await fetch('chat.php', {
                method: 'POST',
                headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                body: 'fetch_chats=1'
            });
            const text = await response.text();
            console.log('Fetch chats response:', text);
            try {
                const chats = JSON.parse(text);
                const chatList = document.getElementById('chat-list');
                chatList.innerHTML = '';
                if (chats.length === 0) {
                    chatList.innerHTML = '<li class="notification">No chats available.</li>';
                    const chatArea = document.querySelector('.chat-area');
                    chatArea.innerHTML = `
                        <div id="chat-content" class="chat-box">
                            <div class="chat-start">
                                <button class="button is-primary is-large" id="start-new-chat">New Chat</button>
                            </div>
                        </div>
                    `;
                    document.getElementById('start-new-chat').addEventListener('click', createNewChat);
                } else {
                    chats.forEach(chat => {
                        const li = document.createElement('li');
                        li.className = 'is-flex is-align-items-center mb-2';
                        li.innerHTML = `
                            <button class="button is-text chat-btn mr-2 is-fullwidth-mobile ${chat.id === currentChatId ? 'is-focused' : ''}" data-chat-id="${chat.id}">
                                ${chat.title}
                            </button>
                            <button class="button is-small edit-chat-btn" data-chat-id="${chat.id}">Edit</button>
                            <button class="button is-danger is-small delete-chat-btn" data-chat-id="${chat.id}">Delete</button>
                        `;
                        chatList.appendChild(li);
                    });
                    attachChatEvents();
                }
            } catch (e) {
                console.error('Update chat list JSON parse error:', e, 'Response:', text);
            }
        }

        function attachChatEvents() {
            document.querySelectorAll('.chat-btn').forEach(btn => {
                btn.addEventListener('click', () => {
                    document.querySelectorAll('.chat-btn').forEach(b => b.classList.remove('is-focused'));
                    btn.classList.add('is-focused');
                    loadChat(btn.dataset.chatId);
                    const chatArea = document.querySelector('.chat-area');
                    chatArea.innerHTML = `
                        <div id="chat-content" class="chat-box">
                            <div id="chat-box"></div>
                            <button id="scroll-to-bottom" class="button is-small is-info scroll-to-bottom is-hidden">Go to Latest</button>
                        </div>
                        <div class="chat-input">
                            <div class="field has-addons">
                                <div class="control is-expanded">
                                    <input class="input" type="text" id="user-input" placeholder="Ask something about your PDFs...">
                                </div>
                                <div class="control">
                                    <button class="button is-primary" id="send-btn">Send</button>
                                </div>
                                <div class="control">
                                    <span id="spinner" class="is-hidden has-text-grey">Loading...</span>
                                </div>
                            </div>
                            <div class="field is-grouped is-grouped-multiline mt-2">
                                <div class="control">
                                    <label class="label">Chat Mode:</label>
                                    <div class="select">
                                        <select id="chat-mode">
                                            <option value="pdf-only" ${'<?php echo $chatMode; ?>' === 'pdf-only' ? 'selected' : ''}>PDF Only</option>
                                            <option value="mixed" ${'<?php echo $chatMode; ?>' === 'mixed' ? 'selected' : ''}>PDF + LLM</option>
                                        </select>
                                    </div>
                                </div>
                                <div class="control">
                                    <label class="label">Ollama Connected:</label>
                                    <?php if ($allowUserModelSelect): ?>
                                        <div class="select">
                                            <select id="model-select">
                                                <?php foreach ($modelNames as $model): ?>
                                                    <option value="<?php echo $model; ?>" <?php echo $model === $currentModel ? 'selected' : ''; ?>><?php echo $model; ?></option>
                                                <?php endforeach; ?>
                                            </select>
                                        </div>
                                    <?php else: ?>
                                        <span class="tag is-primary"><?php echo $currentModel; ?></span>
                                    <?php endif; ?>
                                </div>
                            </div>
                        </div>
                    `;
                    setupChatEvents();
                });
            });

            document.querySelectorAll('.delete-chat-btn').forEach(btn => {
                btn.addEventListener('click', async () => {
                    if (confirm('Are you sure you want to delete this chat?')) {
                        const chatId = btn.dataset.chatId;
                        const response = await fetch('chat.php', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                            body: `delete_chat=1&chat_id=${chatId}`
                        });
                        const text = await response.text();
                        console.log('Delete chat response:', text);
                        try {
                            const jsonStart = text.indexOf('{');
                            const jsonText = text.substring(jsonStart);
                            const result = JSON.parse(jsonText);
                            if (result.success) {
                                if (currentChatId === chatId) {
                                    currentChatId = null;
                                    const chatArea = document.querySelector('.chat-area');
                                    chatArea.innerHTML = `
                                        <div id="chat-content" class="chat-box">
                                            <div class="chat-start">
                                                <button class="button is-primary is-large" id="start-new-chat">New Chat</button>
                                            </div>
                                        </div>
                                    `;
                                    document.getElementById('start-new-chat').addEventListener('click', createNewChat);
                                }
                                saveChatContext();
                                updateChatList();
                            } else {
                                console.error('Delete chat failed:', result.error);
                            }
                        } catch (e) {
                            console.error('Delete chat JSON parse error:', e, 'Response:', text);
                        }
                    }
                });
            });

            document.querySelectorAll('.edit-chat-btn').forEach(btn => {
                btn.addEventListener('click', async () => {
                    const chatId = btn.dataset.chatId;
                    const newTitle = prompt('Enter new chat title:', document.querySelector(`.chat-btn[data-chat-id="${chatId}"]`).textContent.trim());
                    if (newTitle && newTitle.trim()) {
                        const response = await fetch('chat.php', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                            body: `update_title=1&chat_id=${chatId}&title=${encodeURIComponent(newTitle.trim())}`
                        });
                        const text = await response.text();
                        console.log('Edit chat response:', text);
                        try {
                            const jsonStart = text.indexOf('{');
                            const jsonText = text.substring(jsonStart);
                            const result = JSON.parse(jsonText);
                            if (result.success) {
                                updateChatList();
                            } else {
                                console.error('Edit chat failed:', result.error);
                            }
                        } catch (e) {
                            console.error('Edit chat JSON parse error:', e, 'Response:', text);
                        }
                    }
                });
            });
        }

        function setupChatEvents() {
            const scrollButton = document.getElementById('scroll-to-bottom');
            const chatBox = document.getElementById('chat-box');
            const sendButton = document.getElementById('send-btn');
            const userInput = document.getElementById('user-input');

            if (scrollButton) {
                scrollButton.addEventListener('click', () => {
                    chatBox.scrollTop = chatBox.scrollHeight;
                    updateScrollButton();
                });
            }
            if (chatBox) {
                chatBox.addEventListener('scroll', updateScrollButton);
            }
            if (sendButton) {
                sendButton.addEventListener('click', sendMessage);
            }
            if (userInput) {
                userInput.addEventListener('keypress', (e) => {
                    if (e.key === 'Enter') sendMessage();
                });
            }
        }

        async function updatePdfList() {
            const pdfList = document.getElementById('pdf-list');
            pdfList.innerHTML = '';
            if (<?php echo count($pdfs) > 0 ? 'false' : 'true'; ?>) {
                pdfList.innerHTML = '<li class="notification">No PDFs available.</li>';
            } else {
                <?php foreach ($pdfs as $pdf): ?>
                    <?php $originalName = str_replace('_', ' ', pathinfo($pdf['file_name'], PATHINFO_FILENAME)) . '.' . pathinfo($pdf['file_name'], PATHINFO_EXTENSION); ?>
                    const li = document.createElement('li');
                    li.className = 'is-flex is-align-items-center mb-2 is-flex-wrap-wrap';
                    li.innerHTML = `
                        <input class="checkbox mr-2 pdf-checkbox" type="checkbox" data-pdf-id="<?php echo $pdf['id']; ?>" ${selectedPdfIds.includes('<?php echo $pdf['id']; ?>') ? 'checked' : ''}>
                        <button class="button is-text mr-2 is-fullwidth-mobile pdf-name-button"><?php echo htmlspecialchars($originalName); ?></button>
                        <?php if ($pdf['vectorized']): ?><span class="tag is-success mr-2">Vectorized</span><?php endif; ?>
                        <button class="button is-danger is-small delete-btn" data-pdf-id="<?php echo $pdf['id']; ?>">Delete</button>
                    `;
                    pdfList.appendChild(li);
                <?php endforeach; ?>
                document.querySelectorAll('.delete-btn').forEach(btn => {
                    btn.addEventListener('click', async () => {
                        if (confirm('Are you sure you want to delete this PDF?')) {
                            const response = await fetch('process.php', {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                                body: `delete_id=${btn.dataset.pdfId}`
                            });
                            const text = await response.text();
                            console.log('Delete PDF response:', text);
                            try {
                                const result = JSON.parse(text);
                                if (result.success) {
                                    selectedPdfIds = selectedPdfIds.filter(id => id !== btn.dataset.pdfId);
                                    saveChatContext();
                                    updatePdfList();
                                } else {
                                    console.error('Delete PDF failed:', result.error);
                                }
                            } catch (e) {
                                console.error('Delete PDF JSON parse error:', e, 'Response:', text);
                            }
                        }
                    });
                });
            }
        }

        async function loadChat(chatId) {
            currentChatId = chatId;
            const response = await fetch('chat.php', {
                method: 'POST',
                headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                body: `load_chat=${currentChatId}`
            });
            const text = await response.text();
            console.log('Load chat response:', text);
            try {
                const messages = JSON.parse(text);
                const chatBox = document.getElementById('chat-box');
                chatBox.innerHTML = '';
                messages.forEach(msg => {
                    chatBox.innerHTML += `<div class="message user-message">${msg.user_message}</div>`;
                    if (msg.llm_response) {
                        const formattedResponse = formatResponse(msg.llm_response);
                        chatBox.innerHTML += `<div class="message llm-response">${formattedResponse}</div>`;
                    }
                });
                chatBox.scrollTop = chatBox.scrollHeight;
                updateScrollButton();

                const chat = <?php echo json_encode($chats); ?>.find(c => c.id === chatId);
                if (chat && chat.selected_pdf_ids) {
                    selectedPdfIds = chat.selected_pdf_ids.split(',').filter(id => id);
                    document.querySelectorAll('.pdf-checkbox').forEach(cb => {
                        cb.checked = selectedPdfIds.includes(cb.dataset.pdfId);
                    });
                } else {
                    selectedPdfIds = <?php echo json_encode(array_column($pdfs, 'id')); ?>;
                    document.querySelectorAll('.pdf-checkbox').forEach(cb => cb.checked = true);
                }
                saveChatContext();
            } catch (e) {
                console.error('Load chat JSON parse error:', e, 'Response:', text);
                const chatBox = document.getElementById('chat-box');
                chatBox.innerHTML = '<div class="message llm-response">Error loading chat: ' + e.message + '</div>';
            }
        }

        async function sendMessage() {
            if (selectedPdfIds.length === 0) {
                alert('Please select at least one PDF to chat with.');
                return;
            }
            const input = document.getElementById('user-input').value;
            if (!input.trim()) return;
            const mode = document.getElementById('chat-mode').value;
            const chatBox = document.getElementById('chat-box');
            const spinner = document.getElementById('spinner');
            chatBox.innerHTML += `<div class="message user-message">${input}</div>`;
            spinner.className = '';
            isGenerating = true;

            const formData = new FormData();
            formData.append('message', input);
            formData.append('pdf_ids', selectedPdfIds.join(','));
            formData.append('mode', mode);
            formData.append('chat_id', currentChatId);

            try {
                const response = await fetch('chat.php', {
                    method: 'POST',
                    body: formData
                });
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let llmResponse = document.createElement('div');
                llmResponse.className = 'message llm-response';
                chatBox.appendChild(llmResponse);
                let fullResponse = '';

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    const chunk = decoder.decode(value, { stream: true });
                    const lines = chunk.split('\n');
                    for (const line of lines) {
                        if (line.startsWith('data: ')) {
                            const data = JSON.parse(line.substring(6));
                            if (data.response) {
                                fullResponse += data.response;
                                llmResponse.innerHTML = formatResponse(fullResponse);
                                const isAtBottom = chatBox.scrollTop + chatBox.clientHeight >= chatBox.scrollHeight - 50;
                                if (isAtBottom) {
                                    chatBox.scrollTop = chatBox.scrollHeight;
                                }
                            } else if (data.error) {
                                llmResponse.innerHTML = `Error: ${data.error}`;
                                break;
                            }
                        }
                    }
                }
                saveChatContext();
                if (mode === 'mixed') {
                    const formData = new FormData();
                    formData.append('chat_mode', 'mixed');
                    await fetch('save_context.php', { method: 'POST', body: formData });
                }
            } catch (e) {
                console.error('Send message fetch error:', e);
                const llmResponse = document.createElement('div');
                llmResponse.className = 'message llm-response';
                llmResponse.textContent = 'Error: Failed to connect to the server.';
                chatBox.appendChild(llmResponse);
            }

            spinner.className = 'is-hidden';
            document.getElementById('user-input').value = '';
            isGenerating = false;
            updateScrollButton();
        }

        function updateScrollButton() {
            const chatBox = document.getElementById('chat-box');
            const scrollButton = document.getElementById('scroll-to-bottom');
            if (chatBox && scrollButton) {
                const isScrolledUp = chatBox.scrollTop + chatBox.clientHeight < chatBox.scrollHeight - 50;
                scrollButton.className = `button is-small is-info scroll-to-bottom ${isScrolledUp && (isGenerating || chatBox.children.length > 0) ? '' : 'is-hidden'}`;
            }
        }

        <?php if ($allowUserModelSelect): ?>
            document.getElementById('model-select').addEventListener('change', async () => {
                const selectedModel = document.getElementById('model-select').value;
                const formData = new FormData();
                formData.append('user_selected_model', selectedModel);
                await fetch('settings.php', {
                    method: 'POST',
                    body: formData
                });
                location.reload();
            });
        <?php endif; ?>

        async function saveChatContext() {
            const formData = new FormData();
            formData.append('chat_id', currentChatId || '');
            formData.append('pdf_ids', selectedPdfIds.join(','));
            await fetch('save_context.php', {
                method: 'POST',
                body: formData
            });
            console.log('Saved chat context: chatId=' + currentChatId + ', pdfIds=' + selectedPdfIds.join(','));
        }

        function formatResponse(text) {
            const lines = text.split('\n');
            let html = '';
            let inList = false;

            lines.forEach(line => {
                line = line.trim();
                if (line.startsWith('- ') || line.startsWith('* ')) {
                    if (!inList) {
                        html += '<ul>';
                        inList = true;
                    }
                    html += `<li>${line.substring(2)}</li>`;
                } else if (line) {
                    if (inList) {
                        html += '</ul>';
                        inList = false;
                    }
                    html += `<p>${line}</p>`;
                }
            });

            if (inList) html += '</ul>';
            return html;
        }
    </script>
</body>
</html>