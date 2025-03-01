<?php
session_start();
require 'db.php';

if (!isset($_SESSION['user_id'])) {
    exit;
}

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $chatId = $_POST['chat_id'] ?? null;
    $pdfIds = $_POST['pdf_ids'] ?? '';
    $chatMode = $_POST['chat_mode'] ?? null;

    $_SESSION['current_chat_id'] = $chatId;
    $_SESSION['selected_pdf_ids'] = array_filter(explode(',', $pdfIds));
    if ($chatMode) $_SESSION['chat_mode'] = $chatMode;
    error_log("Saved context: Chat ID: $chatId, PDF IDs: $pdfIds, Chat Mode: " . ($chatMode ?? 'unchanged'));
}
?>