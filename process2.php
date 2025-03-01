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
        if ($pdfText === false) {
            throw new Exception("Failed to extract text from PDF");
        }

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

        $stmt = $pdo->prepare("INSERT INTO pdfs (user_id, file_name, file_url, extracted_text) VALUES (?, ?, ?, ?)");
        $stmt->execute([$userId, $uniqueName, $fileUrl, $pdfText]);
        $pdfId = $pdo->lastInsertId();

        error_log("PDF uploaded: $uniqueName for user $userId to $storage");
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
            error_log("Found PDF: $fileUrl");
            if (strpos($fileUrl, 'uploads/') === 0) {
                $filePath = __DIR__ . '/' . $fileUrl;
                if (file_exists($filePath)) {
                    if (unlink($filePath)) {
                        error_log("Successfully deleted local file: $filePath");
                    } else {
                        error_log("Failed to delete local file: $filePath");
                    }
                } else {
                    error_log("Local file not found: $filePath");
                }
            } else {
                deleteFromB2($pdf['file_name']);
                error_log("Deleted B2 file: " . $pdf['file_name']);
            }

            $stmt = $pdo->prepare("DELETE FROM pdfs WHERE id = ? AND user_id = ?");
            $stmt->execute([$pdfId, $userId]);
            error_log("Deleted PDF ID $pdfId from DB for user $userId");
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
        $stmt = $pdo->prepare("SELECT id, file_name, vectorized FROM pdfs WHERE user_id = ? ORDER BY uploaded_at DESC");
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
        $parser = new \Smalot\PdfParser\Parser();
        $pdf = $parser->parseFile($filePath);
        $text = $pdf->getText();
        return $text ?: 'No text extracted';
    } catch (Exception $e) {
        error_log("PDF text extraction failed: " . $e->getMessage());
        return false;
    }
}
?>
