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

    // Determine the response format based on the user's request
    $responseFormat = 'text'; // Default format
    if (stripos($userMessage, 'bullet points') !== false) {
        $responseFormat = 'bullets';
    } elseif (stripos($userMessage, 'json') !== false) {
        $responseFormat = 'json';
    }

    // Add instructions for the LLM based on the desired format
    $formatInstruction = '';
    if ($responseFormat === 'bullets') {
        $formatInstruction = "Provide the response in bullet points.";
    } elseif ($responseFormat === 'json') {
        $formatInstruction = "Provide the response in valid JSON format.";
    }

    $prompt = "[INST] You are an assistant helping a user understand a PDF document. Below is the extracted content from the PDF:\n\n"
            . "PDF Content: \"$pdfText\"\n\n"
            . "The user asked: \"$userMessage\"\n\n"
            . "$formatInstruction\n\n"
            . "Provide a concise, helpful response based on the PDF content. If the content doesn’t contain relevant information, say so clearly. [/INST]";

    $client = new Client();
    try {
        $response = $client->post('http://192.168.0.101:11434/api/generate', [
            'json' => [
                'model' => 'mistral:7b-instruct-v0.3-q4_0',
                'prompt' => $prompt,
                'stream' => false,
                'max_tokens' => 500,
                'temperature' => 0.7
            ]
        ]);
        $data = json_decode($response->getBody(), true);
        $llmResponse = $data['response'] ?? 'No response generated.';

        // Format the response based on the requested format
        if ($responseFormat === 'bullets') {
            $llmResponse = "<ul>" . implode('', array_map(function($line) {
                return "<li>" . htmlspecialchars(trim($line)) . "</li>";
            }, explode("\n", $llmResponse))) . "</ul>";
        } elseif ($responseFormat === 'json') {
            $llmResponse = "<pre><code>" . htmlspecialchars(json_encode(json_decode($llmResponse, true), JSON_PRETTY_PRINT) . "</code></pre>";
        } else {
            $llmResponse = nl2br(htmlspecialchars($llmResponse));
        }

        echo $llmResponse;
    } catch (Exception $e) {
        echo "Error: " . $e->getMessage();
    }
} else {
    echo "Invalid request.";
}
?>