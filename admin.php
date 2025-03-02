<?php
session_start();
require 'db.php';
require 'vendor/autoload.php';

ini_set('display_errors', 0);
ini_set('log_errors', 1);
ini_set('error_log', '/var/www/html/chatpdf/php_errors.log');

if (!isset($_SESSION['user_id']) || $_SESSION['role'] !== 'admin') {
    error_log("Unauthorized access attempt to admin.php by user_id: " . ($_SESSION['user_id'] ?? 'unknown'));
    header("Location: login.php");
    exit;
}

$userId = $_SESSION['user_id'];
$username = $_SESSION['username'] ?? 'Admin';

// Handle user creation
if (isset($_POST['create_user'])) {
    $newUsername = trim($_POST['username']);
    $newPassword = password_hash(trim($_POST['password']), PASSWORD_DEFAULT);
    $newRole = $_POST['role'] === 'admin' ? 'admin' : 'user'; // Only 'user' or 'admin'

    try {
        $stmt = $pdo->prepare("INSERT INTO users (username, password, role) VALUES (?, ?, ?)");
        $stmt->execute([$newUsername, $newPassword, $newRole]);
        error_log("User created: $newUsername with role: $newRole");
    } catch (Exception $e) {
        error_log("User creation error: " . $e->getMessage());
    }
}

// Handle user deletion
if (isset($_POST['delete_user'])) {
    $deleteUserId = $_POST['user_id'];
    try {
        $stmt = $pdo->prepare("DELETE FROM users WHERE id = ? AND id != ?"); // Prevent self-deletion
        $stmt->execute([$deleteUserId, $userId]);
        error_log("User deleted: ID $deleteUserId");
    } catch (Exception $e) {
        error_log("User deletion error: " . $e->getMessage());
    }
}

// Handle user role update
if (isset($_POST['update_role'])) {
    $updateUserId = $_POST['user_id'];
    $newRole = $_POST['new_role'] === 'admin' ? 'admin' : 'user'; // Only 'user' or 'admin'
    try {
        $stmt = $pdo->prepare("UPDATE users SET role = ? WHERE id = ? AND id != ?"); // Prevent self-role change
        $stmt->execute([$newRole, $updateUserId, $userId]);
        error_log("User role updated: ID $updateUserId to $newRole");
    } catch (Exception $e) {
        error_log("User role update error: " . $e->getMessage());
    }
}

// Fetch all users
try {
    $stmt = $pdo->prepare("SELECT id, username, role, created_at FROM users ORDER BY created_at DESC");
    $stmt->execute();
    $users = $stmt->fetchAll(PDO::FETCH_ASSOC);
} catch (Exception $e) {
    error_log("Users fetch error: " . $e->getMessage());
    $users = [];
}

// Fetch all PDFs
try {
    $stmt = $pdo->prepare("SELECT id, user_id, file_name, file_url, uploaded_at FROM pdfs ORDER BY uploaded_at DESC");
    $stmt->execute();
    $pdfs = $stmt->fetchAll(PDO::FETCH_ASSOC);
} catch (Exception $e) {
    error_log("PDFs fetch error: " . $e->getMessage());
    $pdfs = [];
}
?>

<!DOCTYPE html>
<html>
<head>
    <title>ChatPDF - Admin Panel</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bulma@1.0.2/css/bulma.min.css">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link rel="icon" href="data:,">
    <style>
        body { margin: 0; padding: 2rem; }
        .container { max-width: 1200px; margin: 0 auto; }
        .box { margin-bottom: 2rem; }
        .table-container { overflow-x: auto; }
        .button.is-small { margin: 0.25rem; }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="title is-1">Admin Panel</h1>
        <p class="subtitle">Welcome, <?php echo htmlspecialchars($username); ?>!</p>
        <div class="buttons">
            <a href="index.php" class="button is-primary">Back to Chat</a>
            <a href="logout.php" class="button is-light">Logout</a>
        </div>

        <!-- User Management -->
        <div class="box">
            <h2 class="title is-3">Manage Users</h2>
            <form method="POST" class="field has-addons">
                <div class="control">
                    <input class="input" type="text" name="username" placeholder="New Username" required>
                </div>
                <div class="control">
                    <input class="input" type="password" name="password" placeholder="Password" required>
                </div>
                <div class="control">
                    <div class="select">
                        <select name="role">
                            <option value="user">User</option>
                            <option value="admin">Admin</option>
                        </select>
                    </div>
                </div>
                <div class="control">
                    <button class="button is-success" name="create_user">Create User</button>
                </div>
            </form>

            <div class="table-container">
                <table class="table is-striped is-fullwidth">
                    <thead>
                        <tr>
                            <th>ID</th>
                            <th>Username</th>
                            <th>Role</th>
                            <th>Created At</th>
                            <th>Actions</th>
                        </tr>
                    </thead>
                    <tbody>
                        <?php foreach ($users as $user): ?>
                            <tr>
                                <td><?php echo $user['id']; ?></td>
                                <td><?php echo htmlspecialchars($user['username']); ?></td>
                                <td><?php echo htmlspecialchars($user['role']); ?></td>
                                <td><?php echo $user['created_at']; ?></td>
                                <td>
                                    <?php if ($user['id'] != $userId): // Prevent self-action ?>
                                        <form method="POST" style="display:inline;">
                                            <input type="hidden" name="user_id" value="<?php echo $user['id']; ?>">
                                            <div class="select is-small">
                                                <select name="new_role" onchange="this.form.submit()">
                                                    <option value="user" <?php echo $user['role'] === 'user' ? 'selected' : ''; ?>>User</option>
                                                    <option value="admin" <?php echo $user['role'] === 'admin' ? 'selected' : ''; ?>>Admin</option>
                                                </select>
                                            </div>
                                            <input type="hidden" name="update_role" value="1">
                                        </form>
                                        <form method="POST" style="display:inline;">
                                            <input type="hidden" name="user_id" value="<?php echo $user['id']; ?>">
                                            <button class="button is-danger is-small" name="delete_user" onclick="return confirm('Are you sure?');">Delete</button>
                                        </form>
                                    <?php else: ?>
                                        <span class="tag is-info">You</span>
                                    <?php endif; ?>
                                </td>
                            </tr>
                        <?php endforeach; ?>
                    </tbody>
                </table>
            </div>
        </div>

        <!-- PDF Overview -->
        <div class="box">
            <h2 class="title is-3">All PDFs</h2>
            <div class="table-container">
                <table class="table is-striped is-fullwidth">
                    <thead>
                        <tr>
                            <th>ID</th>
                            <th>User ID</th>
                            <th>File Name</th>
                            <th>File URL</th>
                            <th>Uploaded At</th>
                        </tr>
                    </thead>
                    <tbody>
                        <?php foreach ($pdfs as $pdf): ?>
                            <tr>
                                <td><?php echo $pdf['id']; ?></td>
                                <td><?php echo $pdf['user_id']; ?></td>
                                <td><?php echo htmlspecialchars($pdf['file_name']); ?></td>
                                <td><a href="<?php echo htmlspecialchars($pdf['file_url']); ?>" target="_blank">View</a></td>
                                <td><?php echo $pdf['uploaded_at']; ?></td>
                            </tr>
                        <?php endforeach; ?>
                    </tbody>
                </table>
            </div>
        </div>
    </div>
</body>
</html>