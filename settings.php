<?php
session_start();
require 'db.php';
require 'vendor/autoload.php';

ini_set('display_errors', 0);
ini_set('log_errors', 1);
ini_set('error_log', '/var/www/html/chatpdf/php_errors.log');

if (!isset($_SESSION['user_id'])) {
    header("Location: login.php");
    exit;
}

$isAdmin = $_SESSION['role'] === 'admin';
$userId = $_SESSION['user_id'];

try {
    $stmt = $pdo->query("SELECT key, value FROM settings");
    $settings = [];
    while ($row = $stmt->fetch(PDO::FETCH_ASSOC)) {
        $settings[$row['key']] = $row['value'];
    }
    $defaultModel = $settings['ollama_model'] ?? 'mistral:7b-instruct-v0.3-q4_0';
    $allowUserModelSelect = ($settings['allow_user_model_select'] ?? 'no') === 'yes';
    $b2BucketId = $settings['b2_bucket_id'] ?? '';
    $b2ApplicationKeyId = $settings['b2_application_key_id'] ?? '';
    $b2ApplicationKey = $settings['b2_application_key'] ?? '';
    $currentModel = $_SESSION['user_selected_model'] ?? $defaultModel;
    error_log("Settings loaded: " . json_encode($settings));
} catch (Exception $e) {
    error_log("Settings fetch error: " . $e->getMessage());
    $defaultModel = 'mistral:7b-instruct-v0.3-q4_0';
    $allowUserModelSelect = false;
    $b2BucketId = '';
    $b2ApplicationKeyId = '';
    $b2ApplicationKey = '';
    $currentModel = $defaultModel;
}

$client = new GuzzleHttp\Client();
try {
    $response = $client->request('GET', 'http://192.168.0.101:11434/api/tags', ['timeout' => 5]);
    $ollamaModels = json_decode($response->getBody(), true)['models'] ?? [];
    $modelNames = array_column($ollamaModels, 'name');
    error_log("Available Ollama models: " . json_encode($modelNames));
} catch (Exception $e) {
    error_log("Ollama models fetch error: " . $e->getMessage());
    $modelNames = [$defaultModel];
}

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    try {
        if ($isAdmin) {
            if (isset($_POST['ollama_model'])) {
                $stmt = $pdo->prepare("INSERT INTO settings (key, value) VALUES ('ollama_model', ?) ON CONFLICT (key) DO UPDATE SET value = ?");
                $stmt->execute([$_POST['ollama_model'], $_POST['ollama_model']]);
                error_log("Admin updated default model: " . $_POST['ollama_model']);
            }
            if (isset($_POST['allow_user_model_select'])) {
                $stmt = $pdo->prepare("INSERT INTO settings (key, value) VALUES ('allow_user_model_select', ?) ON CONFLICT (key) DO UPDATE SET value = ?");
                $stmt->execute([$_POST['allow_user_model_select'], $_POST['allow_user_model_select']]);
                error_log("Admin updated allow_user_model_select: " . $_POST['allow_user_model_select']);
            }
            if (isset($_POST['b2_bucket_id']) && !empty($_POST['b2_bucket_id'])) {
                $stmt = $pdo->prepare("INSERT INTO settings (key, value) VALUES ('b2_bucket_id', ?) ON CONFLICT (key) DO UPDATE SET value = ?");
                $stmt->execute([$_POST['b2_bucket_id'], $_POST['b2_bucket_id']]);
                error_log("Admin updated B2 bucket ID: " . $_POST['b2_bucket_id']);
            }
            if (isset($_POST['b2_application_key_id']) && !empty($_POST['b2_application_key_id'])) {
                $stmt = $pdo->prepare("INSERT INTO settings (key, value) VALUES ('b2_application_key_id', ?) ON CONFLICT (key) DO UPDATE SET value = ?");
                $stmt->execute([$_POST['b2_application_key_id'], $_POST['b2_application_key_id']]);
                error_log("Admin updated B2 application key ID: " . $_POST['b2_application_key_id']);
            }
            if (isset($_POST['b2_application_key']) && !empty($_POST['b2_application_key'])) {
                $stmt = $pdo->prepare("INSERT INTO settings (key, value) VALUES ('b2_application_key', ?) ON CONFLICT (key) DO UPDATE SET value = ?");
                $stmt->execute([$_POST['b2_application_key'], $_POST['b2_application_key']]);
                error_log("Admin updated B2 application key");
            }
        }
        if (isset($_POST['user_selected_model']) && $allowUserModelSelect) {
            $_SESSION['user_selected_model'] = $_POST['user_selected_model'];
            error_log("User selected model: " . $_POST['user_selected_model']);
        }
        header("Location: settings.php");
        exit;
    } catch (Exception $e) {
        error_log("POST processing error: " . $e->getMessage());
        $errorMessage = "An error occurred while saving settings: " . $e->getMessage();
    }
}
?>
<!DOCTYPE html>
<html>
<head>
    <title>Settings - ChatPDF</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bulma@1.0.2/css/bulma.min.css">
    <meta name="viewport" content="width=device-width, initial-scale=1">
</head>
<body>
    <section class="section">
        <div class="container">
            <h1 class="title">Settings</h1>
            <?php if (isset($errorMessage)): ?>
                <div class="notification is-danger"><?php echo htmlspecialchars($errorMessage); ?></div>
            <?php endif; ?>
            <div class="box">
                <?php if ($isAdmin): ?>
                    <h2 class="subtitle">Admin Settings</h2>
                    <form method="POST">
                        <div class="field">
                            <label class="label">Default Ollama Model</label>
                            <div class="control">
                                <div class="select">
                                    <select name="ollama_model">
                                        <?php foreach ($modelNames as $model): ?>
                                            <option value="<?php echo $model; ?>" <?php echo $model === $defaultModel ? 'selected' : ''; ?>><?php echo $model; ?></option>
                                        <?php endforeach; ?>
                                    </select>
                                </div>
                            </div>
                        </div>
                        <div class="field">
                            <label class="label">Allow Users to Select Models</label>
                            <div class="control">
                                <div class="select">
                                    <select name="allow_user_model_select">
                                        <option value="yes" <?php echo $allowUserModelSelect ? 'selected' : ''; ?>>Yes</option>
                                        <option value="no" <?php echo !$allowUserModelSelect ? 'selected' : ''; ?>>No</option>
                                    </select>
                                </div>
                            </div>
                        </div>
                        <div class="field">
                            <label class="label">B2 Bucket ID</label>
                            <div class="control">
                                <input class="input" type="text" name="b2_bucket_id" value="<?php echo htmlspecialchars($b2BucketId); ?>" placeholder="Enter your B2 Bucket ID" required>
                            </div>
                        </div>
                        <div class="field">
                            <label class="label">B2 Application Key ID</label>
                            <div class="control">
                                <input class="input" type="text" name="b2_application_key_id" value="<?php echo htmlspecialchars($b2ApplicationKeyId); ?>" placeholder="Enter your B2 Application Key ID" required>
                            </div>
                        </div>
                        <div class="field">
                            <label class="label">B2 Application Key</label>
                            <div class="control">
                                <input class="input" type="text" name="b2_application_key" value="<?php echo htmlspecialchars($b2ApplicationKey); ?>" placeholder="Enter your B2 Application Key" required>
                            </div>
                        </div>
                        <div class="field">
                            <div class="control">
                                <button class="button is-primary" type="submit">Save Admin Settings</button>
                            </div>
                        </div>
                    </form>
                <?php endif; ?>

                <h2 class="subtitle">User Settings</h2>
                <form method="POST">
                    <?php if ($allowUserModelSelect): ?>
                        <div class="field">
                            <label class="label">Your Ollama Model</label>
                            <div class="control">
                                <div class="select">
                                    <select name="user_selected_model">
                                        <?php foreach ($modelNames as $model): ?>
                                            <option value="<?php echo $model; ?>" <?php echo $model === $currentModel ? 'selected' : ''; ?>><?php echo $model; ?></option>
                                        <?php endforeach; ?>
                                    </select>
                                </div>
                            </div>
                        </div>
                        <div class="field">
                            <div class="control">
                                <button class="button is-primary" type="submit">Save Model</button>
                            </div>
                        </div>
                    <?php else: ?>
                        <p>Your current model is set by the admin: <span class="tag is-primary"><?php echo $currentModel; ?></span></p>
                    <?php endif; ?>
                </form>
            </div>
            <a href="index.php" class="button is-link mt-2">Back to Dashboard</a>
        </div>
    </section>
</body>
</html>