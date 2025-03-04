<?php
session_start();
require 'db.php';
require 'vendor/autoload.php';
require_once 'b2_utils.php';

ini_set('display_errors', 0);
ini_set('log_errors', 1);
ini_set('error_log', '/var/www/html/chatpdf/php_errors.log');

header('Content-Type: application/json');

if (!isset($_SESSION['user_id'])) {
    error_log("No user_id in session.");
    echo json_encode(['success' => false, 'error' => 'Not authorized']);
    exit;
}

$userId = $_SESSION['user_id'];

if (isset($_FILES['pdf'])) {
    $file = $_FILES['pdf'];
    $storage = $_POST['storage'] ?? 'local';
    $originalName = $file['name'];
    $sanitizedName = preg_replace('/[^A-Za-z0-9\-_\.]/', '_', $originalName);
    $tempPath = $file['tmp_name'];

    if ($file['error'] !== UPLOAD_ERR_OK) {
        error_log("Upload error for $originalName: " . $file['error']);
        echo json_encode(['success' => false, 'error' => 'Upload failed']);
        exit;
    }

    try {
        $pdfText = extractTextFromPDF($tempPath);
        if ($pdfText === false || $pdfText === 'No text extracted') {
            $pdfText = extractTextWithOCR($tempPath);
            if ($pdfText === false) {
                throw new Exception("Failed to extract text from PDF with OCR");
            }
        }
        file_put_contents('/var/www/html/chatpdf/extracted_text.txt', $pdfText); // Save to file
        error_log("Extracted text length for $originalName: " . strlen($pdfText));
        error_log("Extracted text for $originalName: " . substr($pdfText, 0, 1000)); // Debug

        $uniqueName = $userId . '_' . uniqid() . '_' . $sanitizedName;
        $fileUrl = '';

        if ($storage === 'local') {
            $uploadDir = __DIR__ . '/uploads/';
            if (!is_dir($uploadDir)) {
                mkdir($uploadDir, 0755, true);
            }
            $destination = $uploadDir . $uniqueName;
            if (!move_uploaded_file($tempPath, $destination)) {
                throw new Exception("Failed to move file to local storage");
            }
            $fileUrl = 'uploads/' . $uniqueName;
        } else if ($storage === 'b2') {
            $b2File = uploadToB2($tempPath, $uniqueName);
            if (!$b2File || !isset($b2File['fileId'])) {
                throw new Exception("Failed to upload to B2");
            }
            $fileUrl = getB2DownloadUrl($b2File['fileId']);
        } else {
            throw new Exception("Invalid storage option");
        }

        $stmt = $pdo->prepare("INSERT INTO pdfs (user_id, file_name, file_url, extracted_text, vectorized) VALUES (?, ?, ?, ?, TRUE)");
        $stmt->execute([$userId, $uniqueName, $fileUrl, $pdfText]);
        $pdfId = $pdo->lastInsertId();
        error_log("PDF inserted: ID $pdfId, Name: $uniqueName");

        $client = new GuzzleHttp\Client(['timeout' => 30]);
        $chunks = chunkText($pdfText);
        error_log("Number of chunks for $uniqueName: " . count($chunks));
        if (empty($chunks)) {
            error_log("No chunks generated for $uniqueName—text might be empty or invalid.");
        }

        // Updated INSERT with b2_path
        $stmt = $pdo->prepare("INSERT INTO pdf_chunks (pdf_id, chunk_text, embedding, chunk_number, b2_path) VALUES (?, ?, ?, ?, ?)");

        foreach ($chunks as $index => $chunk) {
            try {
                error_log("Processing chunk $index for PDF $pdfId: " . substr($chunk, 0, 50) . "...");
                $response = $client->post('http://localhost/chatpdf/proxy.php', [
                    'json' => ['model' => 'mxbai-embed-large:latest', 'prompt' => $chunk],
                    'headers' => ['Content-Type' => 'application/json']
                ]);
                $body = $response->getBody()->getContents();
                $embedData = json_decode($body, true);
                
                if (!isset($embedData['embedding']) || !is_array($embedData['embedding']) || count($embedData['embedding']) !== 1024) {
                    throw new Exception("Invalid embedding response: " . $body);
                }
                
                $embedding = $embedData['embedding'];
                error_log("Chunk $index embedded for PDF $pdfId, dims: " . count($embedding));
                $b2Path = ($storage === 'b2') ? $fileUrl : ''; // Use fileUrl for B2, empty for local
                $stmt->execute([$pdfId, $chunk, '[' . implode(',', $embedding) . ']', $index, $b2Path]);
                error_log("Chunk $index stored for PDF $pdfId with b2_path: $b2Path");
            } catch (Exception $e) {
                error_log("Embedding failed for chunk $index of PDF $pdfId: " . $e->getMessage());
                throw new Exception("Embedding process failed: " . $e->getMessage());
            }
        }

        error_log("PDF uploaded and vectorized: $uniqueName for user $userId to $storage");
        echo json_encode(['success' => true, 'pdf_id' => $pdfId]);
    } catch (Exception $e) {
        error_log("Error processing PDF $originalName: " . $e->getMessage());
        echo json_encode(['success' => false, 'error' => $e->getMessage()]);
        if ($storage === 'local' && isset($destination) && file_exists($destination)) {
            unlink($destination);
        }
    }
    exit;
}

if (isset($_POST['delete_id'])) {
    $pdfId = $_POST['delete_id'];
    error_log("Attempting to delete PDF ID: $pdfId for user $userId");
    try {
        $stmt = $pdo->prepare("SELECT file_name, file_url FROM pdfs WHERE id = ? AND user_id = ?");
        $stmt->execute([$pdfId, $userId]);
        $pdf = $stmt->fetch();

        if ($pdf) {
            $fileUrl = $pdf['file_url'];
            error_log("Found PDF with file_url: $fileUrl");

            if (strpos($fileUrl, 'uploads/') === 0) {
                $filePath = __DIR__ . '/' . $fileUrl;
                error_log("Checking local file path: $filePath");
                if (file_exists($filePath)) {
                    if (unlink($filePath)) {
                        error_log("Successfully deleted local file: $filePath");
                    } else {
                        error_log("Failed to delete local file: $filePath - Permission or path issue");
                    }
                } else {
                    error_log("Local file not found at: $filePath");
                }
            } else {
                try {
                    deleteFromB2($pdf['file_name']);
                    error_log("Successfully deleted B2 file: " . $pdf['file_name']);
                } catch (Exception $b2e) {
                    error_log("Failed to delete B2 file " . $pdf['file_name'] . ": " . $b2e->getMessage());
                }
            }

            $stmt = $pdo->prepare("DELETE FROM pdfs WHERE id = ? AND user_id = ?");
            $stmt->execute([$pdfId, $userId]);
            error_log("Deleted PDF ID $pdfId from database for user $userId");

            echo json_encode(['success' => true]);
        } else {
            error_log("PDF ID $pdfId not found or not authorized for user $userId");
            echo json_encode(['success' => false, 'error' => 'PDF not found or not authorized']);
        }
    } catch (Exception $e) {
        error_log("Error deleting PDF ID $pdfId: " . $e->getMessage());
        echo json_encode(['success' => false, 'error' => $e->getMessage()]);
    }
    exit;
}

if (isset($_POST['fetch_pdfs'])) {
    try {
        $stmt = $pdo->prepare("SELECT id, file_name, file_url, vectorized FROM pdfs WHERE user_id = ? ORDER BY uploaded_at DESC");
        $stmt->execute([$userId]);
        $pdfs = $stmt->fetchAll(PDO::FETCH_ASSOC);
        error_log("Fetched PDFs for user $userId: " . json_encode($pdfs));
        echo json_encode($pdfs);
    } catch (Exception $e) {
        error_log("Error fetching PDFs: " . $e->getMessage());
        echo json_encode(['success' => false, 'error' => $e->getMessage()]);
    }
    exit;
}

echo json_encode(['success' => false, 'error' => 'Invalid request']);
exit;

function extractTextFromPDF($filePath) {
    try {
        $pdftotext = shell_exec("pdftotext " . escapeshellarg($filePath) . " -");
        error_log("pdftotext output: " . substr($pdftotext, 0, 1000));
        if ($pdftotext && strlen($pdftotext) > 10) {
            return $pdftotext;
        }
        $parser = new \Smalot\PdfParser\Parser();
        $pdf = $parser->parseFile($filePath);
        $text = $pdf->getText();
        error_log("Smalot output: " . substr($text, 0, 1000));
        return $text ?: 'No text extracted';
    } catch (Exception $e) {
        error_log("PDF extraction failed: " . $e->getMessage());
        return false;
    }
}
function extractTextWithOCR($filePath) {
    try {
        $ocr = new \thiagoalessio\TesseractOCR\TesseractOCR();
        $ocr->image($filePath)->dpi(300)->psm(1); // PSM 1 for full page
        $text = $ocr->run();
        error_log("Tesseract output: " . substr($text, 0, 1000));
        return $text ?: 'No text extracted via OCR';
    } catch (Exception $e) {
        error_log("Tesseract OCR failed: " . $e->getMessage());
        return false;
    }
}

function chunkText($text) {
    $chunks = [];
    $sentences = preg_split('/(?<=[.!?])\s+/', $text, -1, PREG_SPLIT_NO_EMPTY);
    $currentChunk = '';
    foreach ($sentences as $sentence) {
        if (strlen($currentChunk) + strlen($sentence) <= 1000) { // Up from 500
            $currentChunk .= ' ' . $sentence;
        } else {
            if ($currentChunk) $chunks[] = trim($currentChunk);
            $currentChunk = $sentence;
        }
    }
    if ($currentChunk) $chunks[] = trim($currentChunk);
    return $chunks;
}
?>