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

            if (!move_uploaded_file($file['tmp_name'], $filePath)) {
                throw new Exception("Failed to move uploaded file: " . $file['error']);
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

        // If text is empty, try OCR
        if (empty(trim($text))) {
            error_log("Text is empty, attempting OCR...");

            $imagePrefix = $uploadDir . 'temp_image';
            $imagePath = $imagePrefix . '-1.png';
            $command = "pdftoppm -png -f 1 -l 1 " . escapeshellarg($filePath) . " " . escapeshellarg($imagePrefix);

            error_log("Running pdftoppm: $command");
            exec($command . " 2>&1", $output, $returnVar);
            error_log("pdftoppm output: " . (empty($output) ? "No output" : implode("\n", $output)));
            error_log("pdftoppm return code: $returnVar");

            if (file_exists($imagePath)) {
                error_log("Image created: $imagePath");
                try {
                    $tesseract = new TesseractOCR($imagePath);
                    $text = $tesseract->run();
                    error_log("OCR text: " . substr($text, 0, 500));
                    unlink($imagePath); // Clean up the image file
                } catch (Exception $e) {
                    error_log("Tesseract error: " . $e->getMessage());
                    $text = ''; // Default to empty if OCR fails
                }
            } else {
                error_log("Image not found at: $imagePath");
                throw new Exception("Failed to convert PDF to image. Return code: $returnVar, Output: " . (empty($output) ? "None" : implode(", ", $output)));
            }
        }

        if (empty(trim($text))) {
            error_log("Warning: Text still empty after processing.");
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