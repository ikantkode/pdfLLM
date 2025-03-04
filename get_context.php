<?php
session_start();
require 'db.php';

if (!isset($_SESSION['user_id'])) die(json_encode(['error' => 'Not logged in']));

if ($_SERVER['REQUEST_METHOD'] === 'POST' && isset($_POST['query_embedding'])) {
    $queryEmbedding = json_decode($_POST['query_embedding'], true);
    $pdfIds = explode(',', $_POST['pdf_ids']);
    
    $placeholders = implode(',', array_fill(0, count($pdfIds), '?'));
    $stmt = $pdo->prepare("
        SELECT chunk_text
        FROM pdf_chunks
        WHERE pdf_id IN ($placeholders)
        ORDER BY embedding <=> ?::vector LIMIT 3
    ");
    $params = array_merge($pdfIds, ['[' . implode(',', $queryEmbedding) . ']']);
    $stmt->execute($params);
    $chunks = $stmt->fetchAll(PDO::FETCH_COLUMN);

    $context = implode("\n", $chunks);
    echo json_encode(['context' => $context]);
}
?>