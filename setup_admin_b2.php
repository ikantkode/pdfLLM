<?php
require 'b2_utils.php';

// Create a dummy file to initialize the folder
$dummyFile = sys_get_temp_dir() . '/dummy.txt';
file_put_contents($dummyFile, 'dummy');

// Set up folder for admin (user ID 1)
$adminUserId = 1;
$adminFolder = "chatpdf/user_$adminUserId";
uploadB2File($dummyFile, 'dummy.txt', $adminFolder);

unlink($dummyFile);
echo "Admin folder $adminFolder created in B2.";
?>