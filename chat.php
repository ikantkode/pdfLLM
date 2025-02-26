<?php
require 'vendor/autoload.php';
use GuzzleHttp\Client;

session_start();

error_reporting(E_ALL);
ini_set('display_errors', 1);

if ($_SERVER['REQUEST_METHOD'] === 'POST' && isset($_POST['message'])) {
    $userMessage = $_POST['message'];
    $pdfText = $_SESSION['pdf_text'] ?? 'No PDF content available yet.';
    $pdfText = substr($pdfText, 0, 10000);

    $prompt = "[INST] You are an assistant helping a user understand a PDF document. Below is the extracted content from the PDF:\n\n"
            . "PDF Content: \"$pdfText\"\n\n"
            . "The user asked: \"$userMessage\"\n\n"
            . "Provide a concise, helpful response based on the PDF content. If the content doesn’t contain relevant information, say so clearly. [/INST]";

    $client = new Client();
    try {
        $response = $client->post('http://192.168.0.101:11434/api/generate', [
            'json' => [
                'model' => 'phi4:latest',
                'prompt' => $prompt,
                'stream' => false,
                'max_tokens' => 500,
                'temperature' => 0.7
            ]
        ]);
        $data = json_decode($response->getBody(), true);
        echo $data['response'] ?? 'No response generated.';
    } catch (Exception $e) {
        echo "Error: " . $e->getMessage();
    }
} else {
    echo "Invalid request.";
}
?>