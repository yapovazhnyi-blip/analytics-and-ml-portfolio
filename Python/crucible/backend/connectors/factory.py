"""
ConnectorFactory for Crucible.

Reads a Connector ORM record from the database, decrypts its credentials,
and returns the right concrete connector instance. This is the single
place where the connector type string maps to a class — no other layer
needs to know about this dispatch logic.

Why a factory instead of a method on the Connector model:
  - The ORM model lives in the models/ layer and should not import
    from the connectors/ layer (would create a circular dependency).
  - Decryption logic belongs here, not in the model.
  - The factory can be swapped in tests with a test double without
    touching the ORM.
"""

from __future__ import annotations

from typing import Optional

from models.connector import Connector
from connectors.file_connector import FileConnector
from connectors.sql_connector import SQLConnector
from connectors.rest_connector import (
    PaginatedRestConnector,
    PaginationConfig,
    RateLimitConfig,
)
from connectors.auth.oauth2 import (
    AuthCodeConnector,
    ClientCredentialsConnector,
    OAuth2Config,
    OAuth2Token,
)
from storage.encryption import decrypt
from config import settings

import httpx


class ConnectorFactory:

    @staticmethod
    def file(storage_dir: Optional[str] = None) -> FileConnector:
        return FileConnector(storage_dir or settings.dataset_storage_path)

    @staticmethod
    def sql(db_record: Connector) -> SQLConnector:
        """
        Returns a SQLConnector with the connection URL decrypted.
        Raises ValueError if the record has no database URL.
        """
        raw_url = decrypt(db_record.db_url_encrypted)
        if not raw_url:
            raise ValueError(f"Connector '{db_record.name}' has no database URL configured")

        return SQLConnector(
            db_url=raw_url,
            storage_dir=settings.dataset_storage_path,
        )

    @staticmethod
    def rest(
        db_record: Connector,
        pagination: Optional[PaginationConfig] = None,
    ) -> PaginatedRestConnector:
        """
        Returns a PaginatedRestConnector. If the connector has OAuth2
        credentials, builds the appropriate OAuth2 sub-connector and
        passes it as the auth layer.

        The auth sub-connector is responsible for token caching and
        refresh — the REST connector just asks for a valid token before
        each request.
        """
        base_url = db_record.base_url
        if not base_url:
            raise ValueError(f"Connector '{db_record.name}' has no base URL configured")

        http_client = httpx.AsyncClient(timeout=30.0)
        auth_connector: Optional[object] = None

        if db_record.client_id and db_record.client_secret_encrypted:
            client_secret = decrypt(db_record.client_secret_encrypted) or ""

            # Use the explicitly stored token_url and auth_url — these are
            # almost never the same as base_url. For GitHub for example,
            # base_url is https://api.github.com, token_url is
            # https://github.com/login/oauth/access_token, and auth_url is
            # https://github.com/login/oauth/authorize. The old code used
            # base_url for all three, which is the bug being fixed here.
            oauth_cfg = OAuth2Config(
                client_id=db_record.client_id,
                client_secret=client_secret,
                token_url=db_record.token_url or "",
                auth_url=db_record.auth_url or "",
                scopes=[],
                token_body_format=db_record.token_body_format or "json",
                response_format=db_record.response_format or "json",
                client_auth=db_record.client_auth or "body",
                use_pkce=bool(db_record.use_pkce),
            )

            if db_record.oauth2_flow == "client_credentials":
                auth_connector = ClientCredentialsConnector(oauth_cfg, http_client)
                # Inject stored token if we have one (avoids an unnecessary token request)
                _inject_stored_token(auth_connector, db_record)

            elif db_record.oauth2_flow == "authorization_code":
                auth_connector = AuthCodeConnector(oauth_cfg, http_client)
                _inject_stored_token(auth_connector, db_record)

        return PaginatedRestConnector(
            base_url=base_url,
            storage_dir=settings.dataset_storage_path,
            auth_connector=auth_connector,  # type: ignore[arg-type]
            pagination=pagination or PaginationConfig(),
        )

    @staticmethod
    def from_record(
        db_record: Connector,
        pagination: Optional[PaginationConfig] = None,
    ):
        """
        Dispatches to the right factory method based on connector_type.
        This is the primary entry point for all router code.
        """
        ct = db_record.connector_type

        if ct in ("csv", "parquet"):
            return ConnectorFactory.file()

        elif ct in ("sql_postgres", "sql_sqlite", "sql_bigquery"):
            return ConnectorFactory.sql(db_record)

        elif ct in ("rest_oauth2", "rest_api_key"):
            return ConnectorFactory.rest(db_record, pagination=pagination)

        else:
            raise ValueError(f"Unknown connector type: {ct!r}")


def _inject_stored_token(connector, db_record: Connector) -> None:
    """
    If the connector record has a stored access token, inject it into
    the auth connector's token cache. This avoids a token request on
    every connector use — the token is only refreshed when it expires.
    """
    import time
    if not db_record.access_token_encrypted:
        return

    access_token = decrypt(db_record.access_token_encrypted)
    refresh_token = decrypt(db_record.refresh_token_encrypted)

    if not access_token:
        return

    from connectors.auth.oauth2 import OAuth2Token
    connector._token = OAuth2Token(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=None if not db_record.token_expires_at else int(
            max(0, db_record.token_expires_at - time.time())
        ),
        issued_at=time.time(),
    )
