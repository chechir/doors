import pathlib
from unittest import mock

import pytest

from doors.download import BaseFtpClient, BaseHttpClient

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


ftp_access_cfg = {"ftp": {"url": "ftp://foo.bar.co.uk", "remote_directory": "/files/"}}


ftp_files_cfg = {
    "a": {
        "filename_pattern": "a_{}.txt",
    },
    "b": {
        "filename_pattern": "b_{}.txt",
    },
}


@pytest.mark.parametrize(
    "url,credentials,expected",
    [
        (
            "https://foo.bar.co.uk/files/{token}",
            {"token": "abc12345"},
            "https://foo.bar.co.uk/files/abc12345/foo.txt",
        ),
        (
            "https://foo.bar.co.uk/files/{username}:{password}",
            {"username": "foo", "password": "bar"},
            "https://foo.bar.co.uk/files/foo:bar/foo.txt",
        ),
    ],
)
@mock.patch("doors.download.urlretrieve")
def test_httpclient_download(mock_urlretrieve, url, credentials, expected):
    cfg = {"access": {"http": {"url": url}}, "files": http_files_cfg}
    with BaseHttpClient(cfg, credentials) as downloader:
        downloader.download_all_files(None, pathlib.Path("/local_dir"))
    assert mock_urlretrieve.call_count == 2
    assert [item.args for item in mock_urlretrieve.call_args_list] == [
        (expected, pathlib.Path("/local_dir/foo.txt")),
        (
            "https://alternative.foo.bar.co.uk/bar.txt",
            pathlib.Path("/local_dir/bar.txt"),
        ),
    ]


@pytest.mark.parametrize(
    "credentials,expected_ftp_kwargs",
    [
        ({"username": "foo", "password": "bar"}, {"user": "foo", "passwd": "bar"}),
        (None, {}),
    ],
)
@mock.patch("builtins.open")
@mock.patch("doors.download.FTP")
def test_ftpclient_download(mock_ftp, mock_open, credentials, expected_ftp_kwargs):
    cfg = {"access": ftp_access_cfg, "files": ftp_files_cfg}
    with BaseFtpClient(cfg, credentials=credentials) as downloader:
        downloader.download_all_files("2023-08-01", pathlib.Path("/local_dir"))
    assert mock_open.call_count == 2
    assert mock_ftp.call_count == 1
    assert mock_ftp.call_args_list[0].args == ("ftp://foo.bar.co.uk",)
    assert mock_ftp.call_args_list[0].kwargs == expected_ftp_kwargs
    assert mock_ftp().retrbinary.call_count == 2
    mock_retrbinary_calls = [item.args for item in mock_ftp().retrbinary.mock_calls]
    assert [item[0] for item in mock_retrbinary_calls] == [
        "RETR a_2023-08-01.txt",
        "RETR b_2023-08-01.txt",
    ]
