<?php
session_start();
require 'db.php';
require 'b2_utils.php';

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $username = trim($_POST['username']);
    if (preg_match('/\s/', $username)) {
        $error = "Username cannot contain spaces.";
    } else {
        $password = password_hash($_POST['password'], PASSWORD_DEFAULT);
        $stmt = $pdo->query("SELECT COUNT(*) FROM users");
        $userCount = $stmt->fetchColumn();
        $role = $userCount == 0 ? 'admin' : 'user';

        $stmt = $pdo->prepare("INSERT INTO users (username, password, role) VALUES (?, ?, ?)");
        $stmt->execute([$username, $password, $role]);
        $userId = $pdo->lastInsertId();

        $dummyFile = sys_get_temp_dir() . '/dummy.txt';
        file_put_contents($dummyFile, 'dummy');
        uploadB2File($dummyFile, 'dummy.txt', "userFiles/$username");
        unlink($dummyFile);

        header("Location: login.php");
        exit;
    }
}
?>
<!DOCTYPE html>
<html>
<head>
    <title>Sign Up</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body class="bg-light">
    <div class="container mt-5">
        <h1 class="text-center">Sign Up</h1>
        <?php if (isset($error)) echo "<p class='text-danger text-center'>$error</p>"; ?>
        <form method="POST" class="col-md-6 mx-auto">
            <div class="mb-3"><input type="text" name="username" class="form-control" placeholder="Username (no spaces)" required></div>
            <div class="mb-3"><input type="password" name="password" class="form-control" placeholder="Password" required></div>
            <button type="submit" class="btn btn-primary w-100">Sign Up</button>
        </form>
        <p class="text-center mt-3">Already have an account? <a href="login.php">Log in</a></p>
    </div>
</body>
</html>