<?php
session_start();
require 'db.php';
if (!isset($_SESSION['user_id']) || $_SESSION['role'] !== 'admin') {
    header("Location: index.php");
    exit;
}

if ($_SERVER['REQUEST_METHOD'] === 'POST' && isset($_POST['create_moderator'])) {
    $username = $_POST['username'];
    $password = password_hash($_POST['password'], PASSWORD_DEFAULT);
    $stmt = $pdo->prepare("INSERT INTO users (username, password, role) VALUES (?, ?, 'moderator')");
    $stmt->execute([$username, $password]);
}

if ($_SERVER['REQUEST_METHOD'] === 'POST' && isset($_POST['ollama_model'])) {
    $model = $_POST['ollama_model'];
    $stmt = $pdo->prepare("INSERT INTO settings (key, value) VALUES ('ollama_model', ?) ON CONFLICT (key) DO UPDATE SET value = ?");
    $stmt->execute([$model, $model]);
}

$users = $pdo->query("SELECT id, username, role FROM users WHERE role != 'admin'")->fetchAll();

// Fetch Ollama models
$ch = curl_init('http://192.168.0.101:11434/api/tags');
curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
$ollamaResponse = curl_exec($ch);
curl_close($ch);
$ollamaModels = json_decode($ollamaResponse, true)['models'] ?? [];

$stmt = $pdo->prepare("SELECT value FROM settings WHERE key = 'ollama_model' LIMIT 1");
$stmt->execute();
$currentModel = $stmt->fetchColumn() ?: 'mistral:7b-instruct-v0.3-q4_0';
?>
<!DOCTYPE html>
<html>
<head>
    <title>Admin Panel</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body class="bg-light">
    <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
        <div class="container">
            <a class="navbar-brand" href="#">ChatPDF Admin</a>
            <div class="navbar-nav ms-auto">
                <a class="nav-link" href="index.php">Dashboard</a>
                <a class="nav-link" href="logout.php">Logout</a>
            </div>
        </div>
    </nav>
    <div class="container mt-4">
        <h1>Admin Panel</h1>
        <h3>Create Moderator</h3>
        <form method="POST" class="mb-4">
            <div class="mb-3"><input type="text" name="username" class="form-control" placeholder="Username" required></div>
            <div class="mb-3"><input type="password" name="password" class="form-control" placeholder="Password" required></div>
            <button type="submit" name="create_moderator" class="btn btn-success">Create</button>
        </form>
        <h3>Users</h3>
        <table class="table table-striped">
            <thead><tr><th>ID</th><th>Username</th><th>Role</th></tr></thead>
            <tbody>
                <?php foreach ($users as $user): ?>
                    <tr><td><?php echo $user['id']; ?></td><td><?php echo htmlspecialchars($user['username']); ?></td><td><?php echo $user['role']; ?></td></tr>
                <?php endforeach; ?>
            </tbody>
        </table>
        <h3>Ollama Model Selection</h3>
        <form method="POST" class="mb-4">
            <div class="mb-3">
                <label for="ollama_model" class="form-label">Select Model (Current: <?php echo htmlspecialchars($currentModel); ?>)</label>
                <select name="ollama_model" id="ollama_model" class="form-select">
                    <?php foreach ($ollamaModels as $model): ?>
                        <option value="<?php echo $model['name']; ?>" <?php echo $model['name'] === $currentModel ? 'selected' : ''; ?>><?php echo htmlspecialchars($model['name']); ?></option>
                    <?php endforeach; ?>
                </select>
            </div>
            <button type="submit" class="btn btn-primary">Set Model</button>
        </form>
    </div>
</body>
</html>