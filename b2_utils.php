<?php
require 'vendor/autoload.php';
use GuzzleHttp\Client;

function getB2Credentials() {
    global $pdo;
    try {
        $stmt = $pdo->query("SELECT key, value FROM settings WHERE key IN ('b2_bucket_id', 'b2_application_key_id', 'b2_application_key')");
        $settings = [];
        while ($row = $stmt->fetch(PDO::FETCH_ASSOC)) {
            $settings[$row['key']] = $row['value'];
        }
        $bucketId = $settings['b2_bucket_id'] ?? null;
        $applicationKeyId = $settings['b2_application_key_id'] ?? null;
        $applicationKey = $settings['b2_application_key'] ?? null;

        if (!$bucketId || !$applicationKeyId || !$applicationKey) {
            throw new Exception("Backblaze B2 credentials not configured. Please set them in Settings.");
        }

        return [
            'bucketId' => $bucketId,
            'applicationKeyId' => $applicationKeyId,
            'applicationKey' => $applicationKey
        ];
    } catch (Exception $e) {
        error_log("B2 credentials fetch error: " . $e->getMessage());
        throw $e; // Re-throw to be caught by calling functions
    }
}

function getB2Auth($applicationKeyId, $applicationKey) {
    $client = new Client();
    $authResponse = $client->request('GET', 'https://api.backblazeb2.com/b2api/v2/b2_authorize_account', [
        'headers' => ['Authorization' => 'Basic ' . base64_encode("$applicationKeyId:$applicationKey")]
    ]);
    $authData = json_decode($authResponse->getBody(), true);
    return [
        'authToken' => $authData['authorizationToken'],
        'apiUrl' => $authData['apiUrl']
    ];
}

function uploadB2File($filePath, $fileName, $folder = 'userFiles') {
    try {
        $creds = getB2Credentials();
        $bucketId = $creds['bucketId'];
        $applicationKeyId = $creds['applicationKeyId'];
        $applicationKey = $creds['applicationKey'];

        $b2Path = "$folder/$fileName";
        $auth = getB2Auth($applicationKeyId, $applicationKey);
        $authToken = $auth['authToken'];
        $apiUrl = $auth['apiUrl'];

        $client = new Client();
        $uploadUrlResponse = $client->request('POST', "$apiUrl/b2api/v2/b2_get_upload_url", [
            'headers' => ['Authorization' => $authToken],
            'json' => ['bucketId' => $bucketId]
        ]);
        $uploadData = json_decode($uploadUrlResponse->getBody(), true);
        $uploadUrl = $uploadData['uploadUrl'];
        $uploadToken = $uploadData['authorizationToken'];

        $fileContent = file_get_contents($filePath);
        $response = $client->request('POST', $uploadUrl, [
            'headers' => [
                'Authorization' => $uploadToken,
                'Content-Type' => 'b2/x-auto',
                'X-Bz-File-Name' => $b2Path,
                'X-Bz-Content-Sha1' => sha1_file($filePath)
            ],
            'body' => $fileContent
        ]);

        $result = json_decode($response->getBody(), true);
        error_log("Uploaded to B2: $b2Path, File ID: " . $result['fileId']);
        return $result['fileId'];
    } catch (Exception $e) {
        error_log("Upload to B2 failed: " . $e->getMessage());
        throw $e;
    }
}

function moveB2File($fileName, $sourceFolder, $destFolder = 'deleted') {
    try {
        $creds = getB2Credentials();
        $bucketId = $creds['bucketId'];
        $applicationKeyId = $creds['applicationKeyId'];
        $applicationKey = $creds['applicationKey'];
        $sourcePath = "$sourceFolder/$fileName";
        $destPath = "$sourceFolder/$destFolder/$fileName";

        $auth = getB2Auth($applicationKeyId, $applicationKey);
        $authToken = $auth['authToken'];
        $apiUrl = $auth['apiUrl'];

        $client = new Client();
        $listResponse = $client->request('POST', "$apiUrl/b2api/v2/b2_list_file_names", [
            'headers' => ['Authorization' => $authToken],
            'json' => ['bucketId' => $bucketId, 'startFileName' => $sourcePath, 'maxFileCount' => 1]
        ]);
        $listData = json_decode($listResponse->getBody(), true);

        if (empty($listData['files'])) {
            error_log("File not found in B2: $sourcePath");
            return false;
        }

        $fileId = $listData['files'][0]['fileId'];
        $copyResponse = $client->request('POST', "$apiUrl/b2api/v2/b2_copy_file", [
            'headers' => ['Authorization' => $authToken],
            'json' => [
                'sourceFileId' => $fileId,
                'fileName' => $destPath
            ]
        ]);
        $copyData = json_decode($copyResponse->getBody(), true);

        if ($copyData && isset($copyData['fileId'])) {
            $client->request('POST', "$apiUrl/b2api/v2/b2_delete_file_version", [
                'headers' => ['Authorization' => $authToken],
                'json' => ['fileName' => $sourcePath, 'fileId' => $fileId]
            ]);
            error_log("Moved B2 file from $sourcePath to $destPath and deleted original");
            return true;
        }

        error_log("Failed to move B2 file: " . json_encode($copyResponse->getBody()));
        return false;
    } catch (Exception $e) {
        error_log("Move B2 file failed: " . $e->getMessage());
        throw $e;
    }
}

function deleteB2File($filePath) {
    try {
        $creds = getB2Credentials();
        $bucketId = $creds['bucketId'];
        $applicationKeyId = $creds['applicationKeyId'];
        $applicationKey = $creds['applicationKey'];

        $auth = getB2Auth($applicationKeyId, $applicationKey);
        $authToken = $auth['authToken'];
        $apiUrl = $auth['apiUrl'];

        $client = new Client();
        $listResponse = $client->request('POST', "$apiUrl/b2api/v2/b2_list_file_names", [
            'headers' => ['Authorization' => $authToken],
            'json' => ['bucketId' => $bucketId, 'startFileName' => $filePath, 'maxFileCount' => 1]
        ]);
        $listData = json_decode($listResponse->getBody(), true);

        if (empty($listData['files'])) {
            error_log("File not found in B2 for deletion: $filePath");
            return false;
        }

        $fileId = $listData['files'][0]['fileId'];
        $client->request('POST', "$apiUrl/b2api/v2/b2_delete_file_version", [
            'headers' => ['Authorization' => $authToken],
            'json' => ['fileName' => $filePath, 'fileId' => $fileId]
        ]);
        error_log("Deleted file from B2: $filePath, File ID: $fileId");
        return true;
    } catch (Exception $e) {
        error_log("Delete B2 file error: " . $e->getMessage());
        throw $e;
    }
}
?>