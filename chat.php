<?php
session_start();
require 'db.php';
require 'vendor/autoload.php';
use GuzzleHttp\Client;

ini_set('display_errors', 0);
ini_set('log_errors', 1);
ini_set('error_log', '/var/www/html/chatpdf/php_errors.log');

header('Content-Type: text/event-stream');
header('Cache-Control: no-cache');
header('Connection: keep-alive');
ob_implicit_flush(true);
ob_end_flush();

if (!isset($_SESSION['user_id'])) {
    error_log("No user_id in session.");
    echo "data: " . json_encode(['success' => false, 'error' => 'Not authorized']) . "\n\n";
    exit;
}

$userId = $_SESSION['user_id'];
error_log("Processing request for user_id: $userId, POST: " . json_encode($_POST));

try {
    $stmt = $pdo->prepare("SELECT value FROM settings WHERE key = 'ollama_model' LIMIT 1");
    $stmt->execute();
    $currentModel = $stmt->fetchColumn() ?: 'llama3.2:3b'; // Updated to llama3.2:3b
    error_log("Current Ollama Model: $currentModel");
} catch (Exception $e) {
    error_log("Error fetching model: " . $e->getMessage());
    echo "data: " . json_encode(['success' => false, 'error' => 'Database error: ' . $e->getMessage()]) . "\n\n";
    exit;
}

if (isset($_POST['new_chat'])) {
    $title = $_POST['title'] ?? 'New Chat';
    $pdfIds = $_POST['pdf_ids'] ?? '';
    try {
        $stmt = $pdo->prepare("INSERT INTO chats (user_id, title, selected_pdf_ids) VALUES (?, ?, ?)");
        $stmt->execute([$userId, $title, $pdfIds]);
        $chatId = $pdo->lastInsertId();
        error_log("New chat created: ID $chatId, Title: $title");
        echo "data: " . json_encode(['success' => true, 'chat_id' => $chatId]) . "\n\n";
    } catch (Exception $e) {
        error_log("New chat error: " . $e->getMessage());
        echo "data: " . json_encode(['success' => false, 'error' => 'Failed to create chat: ' . $e->getMessage()]) . "\n\n";
    }
    exit;
}

if (isset($_POST['update_title']) && isset($_POST['chat_id']) && isset($_POST['title'])) {
    $chatId = $_POST['chat_id'];
    $title = $_POST['title'];
    try {
        $stmt = $pdo->prepare("UPDATE chats SET title = ? WHERE id = ? AND user_id = ?");
        $stmt->execute([$title, $chatId, $userId]);
        error_log("Chat $chatId title updated to: $title");
        echo "data: " . json_encode(['success' => true]) . "\n\n";
    } catch (Exception $e) {
        error_log("Update title error: " . $e->getMessage());
        echo "data: " . json_encode(['success' => false, 'error' => 'Failed to update title: ' . $e->getMessage()]) . "\n\n";
    }
    exit;
}

if (isset($_POST['update_pdfs']) && isset($_POST['chat_id']) && isset($_POST['pdf_ids'])) {
    $chatId = $_POST['chat_id'];
    $pdfIds = $_POST['pdf_ids'];
    try {
        $stmt = $pdo->prepare("UPDATE chats SET selected_pdf_ids = ? WHERE id = ? AND user_id = ?");
        $stmt->execute([$pdfIds, $chatId, $userId]);
        error_log("Chat $chatId PDFs updated: $pdfIds");
        echo "data: " . json_encode(['success' => true]) . "\n\n";
    } catch (Exception $e) {
        error_log("Update PDFs error: " . $e->getMessage());
        echo "data: " . json_encode(['success' => false, 'error' => 'Failed to update PDFs: ' . $e->getMessage()]) . "\n\n";
    }
    exit;
}

if (isset($_POST['delete_chat']) && isset($_POST['chat_id'])) {
    $chatId = $_POST['chat_id'];
    try {
        $stmt = $pdo->prepare("DELETE FROM chat_messages WHERE chat_id = ? AND user_id = ?");
        $stmt->execute([$chatId, $userId]);
        $stmt = $pdo->prepare("DELETE FROM chats WHERE id = ? AND user_id = ?");
        $stmt->execute([$chatId, $userId]);
        error_log("Chat $chatId deleted for user $userId");
        echo "data: " . json_encode(['success' => true]) . "\n\n";
    } catch (Exception $e) {
        error_log("Delete chat error: " . $e->getMessage());
        echo "data: " . json_encode(['success' => false, 'error' => 'Failed to delete chat: ' . $e->getMessage()]) . "\n\n";
    }
    exit;
}

if (isset($_POST['fetch_chats'])) {
    try {
        $stmt = $pdo->prepare("SELECT id, title FROM chats WHERE user_id = ? ORDER BY created_at DESC");
        $stmt->execute([$userId]);
        $chats = $stmt->fetchAll(PDO::FETCH_ASSOC);
        echo json_encode($chats);
    } catch (Exception $e) {
        error_log("Fetch chats error: " . $e->getMessage());
        echo "data: " . json_encode(['success' => false, 'error' => 'Failed to fetch chats: ' . $e->getMessage()]) . "\n\n";
    }
    exit;
}

if (isset($_POST['fetch_chat'])) {
    $chatId = $_POST['fetch_chat'];
    try {
        $stmt = $pdo->prepare("SELECT selected_pdf_ids FROM chats WHERE id = ? AND user_id = ?");
        $stmt->execute([$chatId, $userId]);
        $chat = $stmt->fetch(PDO::FETCH_ASSOC);
        echo json_encode($chat);
    } catch (Exception $e) {
        error_log("Fetch chat error: " . $e->getMessage());
        echo "data: " . json_encode(['success' => false, 'error' => 'Failed to fetch chat: ' . $e->getMessage()]) . "\n\n";
    }
    exit;
}

if (isset($_POST['load_chat'])) {
    $chatId = $_POST['load_chat'];
    try {
        $stmt = $pdo->prepare("SELECT user_message, llm_response, pdf_ids, created_at FROM chat_messages WHERE chat_id = ? AND user_id = ? ORDER BY created_at");
        $stmt->execute([$chatId, $userId]);
        $messages = $stmt->fetchAll(PDO::FETCH_ASSOC);
        error_log("Loaded chat $chatId for user $userId: " . json_encode($messages));
        echo json_encode($messages);
    } catch (Exception $e) {
        error_log("Load chat error: " . $e->getMessage());
        echo "data: " . json_encode(['success' => false, 'error' => 'Failed to load chat: ' . $e->getMessage()]) . "\n\n";
    }
    exit;
}
if (isset($_POST['message']) && isset($_POST['pdf_ids'])) {
    $pdfIds = explode(',', $_POST['pdf_ids']);
    $context = $_POST['context'] ?? '';

    $mode = $_POST['mode'] ?? 'pdf-only';
    $prompt = $mode === 'pdf-only'
        ? "Based solely on this PDF content: \"{$context}\"\n\nUser asked: \"{$_POST['message']}\"\n\nProvide a response using only the PDF content, without adding external knowledge."
        : "Based on this PDF content: \"{$context}\"\n\nUser asked: \"{$_POST['message']}\"\n\nProvide a helpful response combining information from the PDFs with your general knowledge. If grammar or words are broken or incomplete, use your best guess silently.";

    $chatId = $_POST['chat_id'];
    if (!$chatId) {
        error_log("No chat_id provided for message.");
        echo "data: " . json_encode(['success' => false, 'error' => 'No chat selected']) . "\n\n";
        exit;
    }

    $stmt = $pdo->prepare("INSERT INTO chat_messages (chat_id, user_id, user_message, pdf_ids) VALUES (?, ?, ?, ?)");
    $stmt->execute([$chatId, $userId, $_POST['message'], $_POST['pdf_ids']]);
    $messageId = $pdo->lastInsertId();
    error_log("Message inserted: ID $messageId, Chat ID: $chatId");

    $client = new Client();
    try {
        // First Pass: Stream raw response
        $response = $client->post('http://localhost:11434/api/generate', [
            'json' => [
                'model' => $currentModel,
                'prompt' => "[INST] {$prompt} [/INST]",
                'stream' => true
            ],
            'timeout' => 120,
            'stream' => true
        ]);

        $stream = $response->getBody();
        $rawResponse = '';
        while (!$stream->eof()) {
            $chunk = $stream->read(1024);
            $lines = explode("\n", $chunk);
            foreach ($lines as $line) {
                if (trim($line)) {
                    $data = json_decode($line, true);
                    if (isset($data['response'])) {
                        $rawResponse .= $data['response'];
                        echo "data: " . json_encode(['response' => $data['response'], 'type' => 'raw']) . "\n\n";
                        flush();
                    } elseif (isset($data['error'])) {
                        echo "data: " . json_encode(['error' => $data['error'], 'type' => 'raw']) . "\n\n";
                        flush();
                        exit;
                    }
                }
            }
        }
        $stream->close();
        error_log("Raw response for message $messageId: " . $rawResponse);

        // Second Pass: Refine response
        $refinePrompt = "Refine this response for proper grammar, spelling, and punctuation, preserving all names and numbers exactly as they appear: \"$rawResponse\"";
        $refineResponse = $client->post('http://localhost:11434/api/generate', [
            'json' => [
                'model' => $currentModel,
                'prompt' => "[INST] {$refinePrompt} [/INST]",
                'stream' => true
            ],
            'timeout' => 120,
            'stream' => true
        ]);

        $refinedStream = $refineResponse->getBody();
        $refinedResponse = '';
        while (!$refinedStream->eof()) {
            $chunk = $refinedStream->read(1024);
            $lines = explode("\n", $chunk);
            foreach ($lines as $line) {
                if (trim($line)) {
                    $data = json_decode($line, true);
                    if (isset($data['response'])) {
                        $refinedResponse .= $data['response'];
                        echo "data: " . json_encode(['response' => $data['response'], 'type' => 'refined']) . "\n\n";
                        flush();
                    } elseif (isset($data['error'])) {
                        echo "data: " . json_encode(['error' => $data['error'], 'type' => 'refined']) . "\n\n";
                        flush();
                        exit;
                    }
                }
            }
        }
        $refinedStream->close();

        $stmt = $pdo->prepare("UPDATE chat_messages SET llm_response = ? WHERE id = ? AND user_id = ?");
        $stmt->execute([$refinedResponse, $messageId, $userId]);
        error_log("Refined response saved for message $messageId: " . $refinedResponse);
    } catch (Exception $e) {
        error_log("Ollama request error: " . $e->getMessage());
        echo "data: " . json_encode(['error' => 'Failed to get response from Ollama: ' . $e->getMessage()]) . "\n\n";
    }
    exit;
}

error_log("Invalid request received.");
echo "data: " . json_encode(['success' => false, 'error' => 'Invalid request']) . "\n\n";
exit;
?>