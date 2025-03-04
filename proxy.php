<?php
header('Content-Type: application/json');
require 'vendor/autoload.php';
use GuzzleHttp\Client;

$client = new Client();
try {
    $response = $client->post('http://localhost:11434/api/embeddings', [ // Localhost
        'json' => json_decode(file_get_contents('php://input'), true)
    ]);
    echo $response->getBody();
} catch (Exception $e) {
    http_response_code(500);
    echo json_encode(['error' => $e->getMessage()]);
}
?>