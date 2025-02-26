<?php
require 'vendor/autoload.php';
use Smalot\PdfParser\Parser;
use thiagoalessio\TesseractOCR\TesseractOCR;

session_start();
error_reporting(E_ALL);
ini_set('display_errors', 0);
ini_set('log_errors', 1);
ini_set('error_log', '/var/www/html/chatpdf/php_errors.log');

header('Content-Type: application/json');

$response = ['success' => false, 'error' => 'Unknown error'];

try {
    if ($_SERVER['REQUEST_METHOD'] === 'POST') {
        $uploadDir = 'uploads/';
        if (!is_dir($uploadDir)) {
            mkdir($uploadDir, 0755, true);
        }

        if (isset($_POST['cloud-link']) && !empty($_POST['cloud-link'])) {
            // Handle cloud storage link
            $cloudLink = $_POST['cloud-link'];
            $fileName = basename($cloudLink);
            $filePath = $uploadDir . $fileName;

            // Download the file
            $fileContent = file_get_contents($cloudLink);
            if ($fileContent === false) {
                throw new Exception("Failed to download file from the provided link.");
            }

            if (file_put_contents($filePath, $fileContent)) {
                error_log("File downloaded from cloud: $filePath");
            } else {
                throw new Exception("Failed to save downloaded file.");
            }
        } elseif (isset($_FILES['pdf']) && $_FILES['pdf']['error'] === UPLOAD_ERR_OK) {
            // Handle file upload
            $file = $_FILES['pdf'];
            $filePath = $uploadDir . basename($file['name']);

            error_log("Attempting to move uploaded file to: $filePath");

            if (move_uploaded_file($file['tmp_name'], $filePath)) {
                error_log("File moved successfully to: $filePath");
            } else {
                $error = "Failed to move uploaded file.";
                error_log($error);
                throw new Exception($error);
            }
        } else {
            throw new Exception("No file or link provided.");
        }

        // Process the file (text extraction or OCR)
        $text = '';
        try {
            $parser = new Parser();
            $pdf = $parser->parseFile($filePath);
            $text = $pdf->getText();
            error_log("Parser text: " . substr($text, 0, 500));
        } catch (Exception $e) {
            error_log("Parser error: " . $e->getMessage());
        }

        if (empty(trim($text))) {
            // OCR logic here (same as before)
        }

        $_SESSION['pdf_text'] = $text;
        $_SESSION['pdf_name'] = basename($filePath);

        $response = [
            'success' => true,
            'pdf_name' => basename($filePath)
        ];
    } else {
        $response['error'] = "Invalid request method.";
    }
} catch (Exception $e) {
    error_log("Unhandled exception: " . $e->getMessage());
    $response['error'] = $e->getMessage();
}

error_log("Response: " . json_encode($response));
echo json_encode($response);
?>