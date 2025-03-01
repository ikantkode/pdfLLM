<?php
session_start();
require 'db.php';
require 'b2_utils.php';
require 'vendor/autoload.php';
use Smalot\PdfParser\Parser;
use TesseractOCR;

header('Content-Type: application/json');
ini_set('display_errors', 0);
ini_set('log_errors', 1);
ini_set('error_log', '/var/www/html/chatpdf/php_errors.log');

if (!isset($_SESSION['user_id'])) {
    echo json_encode(['success' => false, 'error' => 'Not logged in']);
    exit;
}

$stmt = $pdo->prepare("SELECT username FROM users WHERE id = ?");
$stmt->execute([$_SESSION['user_id']]);
$userFolder = "userFiles/" . $stmt->fetchColumn();
error_log("User folder: $userFolder");

if (isset($_POST['delete_id'])) {
    $pdfId = $_POST['delete_id'];
    try {
        $stmt = $pdo->prepare("SELECT file_name, b2_path FROM pdfs WHERE id = ? AND user_id = ?");
        $stmt->execute([$pdfId, $_SESSION['user_id']]);
        $pdf = $stmt->fetch();

        if ($pdf) {
            moveB2File($pdf['file_name'], $userFolder);
            $deletedFolder = "userFiles/deleted/";
            $deletedPath = $deletedFolder . $pdf['file_name'];
            error_log("Moved file to deleted folder: $deletedPath");

            deleteB2File($deletedPath);
            error_log("Deleted file from B2: $deletedPath");

            $pdo->prepare("DELETE FROM pdfs WHERE id = ?")->execute([$pdfId]);
            error_log("Deleted PDF $pdfId from database");
            echo json_encode(['success' => true]);
        } else {
            error_log("PDF $pdfId not found for user {$_SESSION['user_id']}");
            echo json_encode(['success' => false, 'error' => 'PDF not found']);
        }
    } catch (Exception $e) {
        error_log("Delete PDF error: " . $e->getMessage());
        echo json_encode(['success' => false, 'error' => 'Failed to delete PDF: ' . $e->getMessage()]);
    }
    exit;
}

if ($_SERVER['REQUEST_METHOD'] === 'POST' && isset($_FILES['pdf'])) {
    $file = $_FILES['pdf'];
    $fileSize = $file['size'];
    $fileTmp = $file['tmp_name'];
    $originalName = basename($file['name']);
    $fileNameParts = pathinfo($originalName);
    $friendlyName = strtolower(preg_replace('/[\s-]+/', '_', $fileNameParts['filename'])) . '.' . $fileNameParts['extension'];
    $maxSize = 10 * 1024 * 1024;

    error_log("Processing file: $originalName as $friendlyName ($fileSize bytes)");

    if ($fileSize > $maxSize) {
        $chunkSize = 5 * 1024 * 1024;
        $totalChunks = ceil($fileSize / $chunkSize);
        $chunkFolder = "$userFolder/chunked_" . uniqid();
        $text = '';
        for ($i = 0; $i < $totalChunks; $i++) {
            $offset = $i * $chunkSize;
            $chunk = file_get_contents($fileTmp, false, null, $offset, $chunkSize);
            $chunkPath = sys_get_temp_dir() . "/chunk_$i.pdf";
            if (!file_put_contents($chunkPath, $chunk)) {
                error_log("Failed to write chunk $i for $friendlyName");
                echo json_encode(['success' => false, 'error' => "Failed to process chunk $i"]);
                exit;
            }
            $b2Path = "$chunkFolder/chunk_$i.pdf";
            uploadB2File($chunkPath, "chunk_$i.pdf", $chunkFolder);
            $parser = new Parser();
            try {
                $pdf = $parser->parseFile($chunkPath);
                $chunkText = $pdf->getText();
            } catch (Exception $e) {
                error_log("Parser error for chunk $i of $friendlyName: " . $e->getMessage());
                $chunkText = '';
            }
            if (empty(trim($chunkText))) {
                $imagePath = sys_get_temp_dir() . "/temp_image-$i-1.png";
                exec("pdftoppm -png -f 1 -l 1 " . escapeshellarg($chunkPath) . " " . escapeshellarg(sys_get_temp_dir() . "/temp_image-$i") . " 2>&1", $output, $returnVar);
                if ($returnVar !== 0) {
                    error_log("pdftoppm failed for $friendlyName: " . implode("\n", $output));
                }
                if (file_exists($imagePath)) {
                    try {
                        $tesseract = new TesseractOCR($imagePath);
                        $chunkText = $tesseract->run();
                    } catch (Exception $e) {
                        error_log("Tesseract error for $friendlyName: " . $e->getMessage());
                    }
                    unlink($imagePath);
                }
            }
            $text .= $chunkText;
            unlink($chunkPath);
        }

        $stmt = $pdo->prepare("INSERT INTO pdfs (user_id, file_name, b2_path, extracted_text) VALUES (?, ?, ?, ?)");
        $stmt->execute([$_SESSION['user_id'], $friendlyName, "$userFolder/$friendlyName", $text]);
        $pdfId = $pdo->lastInsertId();

        $stmt = $pdo->prepare("INSERT INTO pdf_chunks (pdf_id, chunk_number, b2_path, extracted_text) VALUES (?, ?, ?, ?)");
        for ($i = 0; $i < $totalChunks; $i++) {
            $stmt->execute([$pdfId, $i, "$chunkFolder/chunk_$i.pdf", substr($text, $i * 5000, 5000)]);
        }
    } else {
        try {
            uploadB2File($fileTmp, $friendlyName, $userFolder);
            $parser = new Parser();
            $pdf = $parser->parseFile($fileTmp);
            $text = $pdf->getText();
            if (empty(trim($text))) {
                $imagePath = sys_get_temp_dir() . '/temp_image-1.png';
                exec("pdftoppm -png -f 1 -l 1 " . escapeshellarg($fileTmp) . " " . escapeshellarg(sys_get_temp_dir() . "/temp_image") . " 2>&1", $output, $returnVar);
                if ($returnVar !== 0) {
                    error_log("pdftoppm failed for $friendlyName: " . implode("\n", $output));
                }
                if (file_exists($imagePath)) {
                    try {
                        $tesseract = new TesseractOCR($imagePath);
                        $text = $tesseract->run();
                    } catch (Exception $e) {
                        error_log("Tesseract error for $friendlyName: " . $e->getMessage());
                    }
                    unlink($imagePath);
                }
            }

            $stmt = $pdo->prepare("INSERT INTO pdfs (user_id, file_name, b2_path, extracted_text) VALUES (?, ?, ?, ?)");
            $stmt->execute([$_SESSION['user_id'], $friendlyName, "$userFolder/$friendlyName", $text]);
            $pdfId = $pdo->lastInsertId();
        } catch (Exception $e) {
            error_log("Upload error for $friendlyName: " . $e->getMessage());
            echo json_encode(['success' => false, 'error' => $e->getMessage()]);
            exit;
        }
    }

    error_log("Uploaded PDF: ID $pdfId, Name: $friendlyName");
    echo json_encode(['success' => true, 'pdf_name' => $friendlyName, 'pdf_id' => $pdfId]);
    exit;
}

error_log("No PDF uploaded or invalid request.");
echo json_encode(['success' => false, 'error' => 'No PDF uploaded']);
exit;
?>