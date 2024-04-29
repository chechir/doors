"""
This module contains the basic functionality for downloading the raw data from the original source.
"""

import abc
import logging
import os
import pathlib
import re
import typing as tp
from abc import ABC, abstractmethod
from ftplib import FTP
from functools import partial
from itertools import chain
from urllib.request import urlretrieve

LOGGER = logging.getLogger(__name__)


class ConfigurationError(Exception):
    pass


class VersionHandlerMissing(Exception):
    pass


class Client(ABC):
    """
    Abstract base class to be used as a template for downloading clients
    E.g. FTP, HTTP
    """

    def __init__(
        self,
        config: tp.Dict,
        logger: logging.Logger = LOGGER,
    ):
        self.config = config
        self.logger = logger

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    @abstractmethod
    def download_all_files(
        self, target_version: str, local_directory: pathlib.Path
    ) -> None:
        """
        Download all files defined in the configuration.

        :param target_version: the version related to all the files we want to download
        :param local_directory: the local storage path
        """

    @abstractmethod
    def download_one_file(
        self, target_file: str, local_directory: pathlib.Path
    ) -> None:
        """
        Download one file that corresponds to the given filename.

        :param target_file: the file name for the file we want to download
        :param local_directory: the local storage path
        """


class BaseFtpClient(Client):
    """The base class for an FTP client"""

    def __init__(
        self,
        config: tp.Dict,
        credentials: tp.Optional[tp.Dict] = None,
        logger: logging.Logger = LOGGER,
    ):
        super().__init__(config, logger)

        self.files_config = config["files"]

        kwargs = {}
        ftp = self.config["access"]["ftp"]

        if credentials:
            if "username" not in credentials:
                raise ConfigurationError("There is no `username` information")
            kwargs["user"] = credentials["username"]

            if "password" not in credentials:
                raise ConfigurationError("There is no `password` information")
            kwargs["passwd"] = credentials["password"]

        self.ftp = FTP(ftp["url"], **kwargs)
        self.ftp.login()
        self.ftp.cwd(ftp["remote_directory"])

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.ftp.close()

    def get_all_file_names(self) -> tp.List[str]:
        """Execute the NLST method of the FTP server to get a list of all the files"""
        return self.ftp.nlst()

    def _concoct_target_filenames_list(self, target_version: str) -> tp.List[str]:
        """Create a list of target filenames to be downloaded from the FTP server."""
        target_filenames = []
        for file_info in self.files_config.values():
            filename = file_info["filename_pattern"].format(target_version)
            target_filenames.append(filename)

        return target_filenames

    def download_one_file(
        self, target_file: str, local_directory: pathlib.Path
    ) -> None:
        """
        Execute the RETR method of the FTP server in order to download one file that corresponds
        to the given filename.

        :param target_file: the file name for the file we want to download
        :param local_directory: the local storage path
        """
        self.logger.info(f"Downloading {target_file}...")
        with open(local_directory.joinpath(target_file), "wb") as tf:
            self.ftp.retrbinary(f"RETR {target_file}", tf.write)

    def download_all_files(
        self, target_version: str, local_directory: pathlib.Path
    ) -> None:
        """
        Download all files defined in the configuration

        :param target_version: the version related to all the files we want to download
        :param local_directory: the local storage path
        """
        for target_file in self._concoct_target_filenames_list(target_version):
            self.download_one_file(target_file, local_directory)


class BaseHttpClient(Client):
    """The base class for an HTTP client"""

    def __init__(
        self,
        config: tp.Dict,
        credentials: tp.Optional[tp.Dict] = None,
        logger: logging.Logger = LOGGER,
    ):
        super().__init__(config, logger=logger)

        if credentials:
            self.config["access"]["http"]["url"] = self.config["access"]["http"][
                "url"
            ].format(**credentials)

    def _concoct_target_filepaths_list(
        self, target_version: tp.Union[str, None]
    ) -> tp.List[str]:
        """
        Create a list of target filepaths to be downloaded from the HTTP server.

        :param target_version: the version related to all the files we want to download
        """
        main_url = self.config["access"]["http"]["url"]

        target_filepaths = []
        for file in self.config["files"].values():
            filename = file["filename_pattern"].format(target_version)
            if alternative_http := file.get("alternative_http"):
                filepath = os.path.join(alternative_http, filename)
            else:
                filepath = os.path.join(main_url, filename)
            target_filepaths.append(filepath)

        return target_filepaths

    def download_one_file(
        self, target_file: str, local_directory: pathlib.Path
    ) -> None:
        """
        Download one file that corresponds to the given filename.

        :param target_file: the file name for the file we want to download
        :param local_directory: the local storage path
        """
        urlretrieve(
            target_file, local_directory.joinpath(os.path.basename(target_file))
        )

    def download_all_files(
        self, target_version: tp.Union[str, None], local_directory: pathlib.Path
    ) -> None:
        """
        Download all files defined in the configuration.

        :param target_version: the version related to all the files we want to download
        :param local_directory: the local storage path
        """
        for target_file in self._concoct_target_filepaths_list(target_version):
            self.download_one_file(target_file, local_directory)
