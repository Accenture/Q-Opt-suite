"""
Copyright (c) 2023 Objectivity Ltd.
"""

import logging
import os
from typing import Any

import boto3  # type: ignore
from botocore.config import Config  # type: ignore
from botocore.exceptions import ProfileNotFound  # type: ignore
from braket.aws import AwsSession  # type: ignore

from platforms.platform import Platform


class AWSBraket(Platform):
    """
    The `Platform` implementation for Amazon AWS Braket.
    """

    def __init__(self, config: dict) -> None:
        """
        Perform AWS Braket authentication with the given configuration taken from the yaml file.

        :param config: the model section of the configuration file
        """

        def get_config(name: str, default: Any = None) -> Any:
            # get configuration information from the yml file with the environment as a fallback
            if name in config:
                return config[name]
            elif name.upper() in os.environ:
                return os.environ[name.upper()]
            else:
                logging.info("AWS: defaulting %s to %s", name, default)
                return default

        def get_proxy_definitions():
            # get proxy information
            http_proxy = get_config("http_proxy")
            if http_proxy:
                proxies = {
                    "http": http_proxy,
                    "https": get_config("https_proxy", http_proxy),
                }
                os.environ["HTTP_PROXY"] = proxies["http"]
                os.environ["HTTPS_PROXY"] = proxies["https"]
                return proxies
            else:
                logging.warning(
                    "AWS: no http proxy configured, this could cause trouble if a VPN is being used"
                )
                return None

        super().__init__(config)
        region_name = get_config("aws_region", "eu-west-2")
        my_config = Config(region_name=region_name, proxies=get_proxy_definitions())
        profile_name = get_config("aws_profile")
        if not profile_name:
            raise ValueError(
                "AWS: no profile specified in config or environment. Create one using the CLI,"
                "see:\nhttps://medium.com/ivymobility-developers/"
                "configure-named-aws-profile-usage-in-applications-aws-cli-60ea7f6f7b40"
            )

        try:
            self.boto_session = boto3.Session(
                profile_name=profile_name, region_name=region_name
            )
            self.aws_session = AwsSession(
                boto_session=self.boto_session, config=my_config
            )
        except ProfileNotFound as ex:
            raise ValueError(f"AWS: profile {profile_name} could not be found!") from ex
