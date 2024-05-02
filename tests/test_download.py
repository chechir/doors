import pathlib
from unittest import mock

from doors.download import SFTPClient

http_files_cfg = {
    "main_file": {
        "filename_pattern": "foo.txt",
        "type": "csv",
        "fields": [
            {
                "name": "col_a",
            },
            {
                "name": "col_b",
                "nullable": True,
                "type": "int",
                "multiple_values": True,
            },
        ],
    },
    "another_file": {
        "filename_pattern": "bar.txt",
        "alternative_http": "https://alternative.foo.bar.co.uk/",
        "type": "csv",
        "fields": [
            {
                "name": "col_a",
            },
            {
                "name": "col_b",
                "nullable": True,
                "type": "int",
                "multiple_values": True,
            },
        ],
    },
}


ftp_access_cfg = {"url": "sftp://foo.bar.co.uk", "username": "bla", "password": "bla"}

ftp_files_cfg = {
    "a": {
        "filename_pattern": "a_{}.txt",
    },
    "b": {
        "filename_pattern": "b_{}.txt",
    },
}


@mock.patch("doors.download.SSHClient")
def test_ftpclient_download(mock_sshclient):
    cfg = ftp_access_cfg
    with SFTPClient(cfg) as downloader:
        downloader.download_all_files("2023-08-01", pathlib.Path("/local_dir"))
    assert mock_sshclient.call_count == 1
    assert mock_sshclient().connect.call_args_list[0].args == ("sftp://foo.bar.co.uk",)
    assert mock_sshclient().connect.call_count == 1
