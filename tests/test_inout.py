import os

from doors.inout import get_md5_hash


def test_get_md5_hash():
    test_file_path = "test_file.txt"
    with open(test_file_path, "wb") as f:
        f.write(b"Hello, world!")

    expected_hash = "6cd3556deb0da54bca060b4c39479839"
    assert (
        get_md5_hash(test_file_path) == expected_hash
    ), "MD5 hash does not match expected value"

    os.remove(test_file_path)


def test_get_md5_hash_file_not_found():
    non_existent_file = "non_existent_file.txt"
    assert (
        get_md5_hash(non_existent_file) == "File not found."
    ), "Error handling for non-existent file failed"
