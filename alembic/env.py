from logging.config import fileConfig
from pathlib import Path
import sys

from alembic import context
from sqlalchemy import engine_from_config, pool


# =========================================================
# Project Path
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# =========================================================
# Application Imports
# =========================================================

from app.database import Base, DATABASE_URL
import app.db_models


# =========================================================
# Alembic Configuration
# =========================================================

config = context.config


# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------

if config.config_file_name is not None:
    fileConfig(
        config.config_file_name
    )


# =========================================================
# Database URL
#
# Use the application's database configuration instead
# of maintaining a second database URL in alembic.ini.
# =========================================================

config.set_main_option(
    "sqlalchemy.url",
    DATABASE_URL,
)


# =========================================================
# SQLAlchemy Metadata
#
# Alembic compares this metadata with the actual database
# when --autogenerate is used.
# =========================================================

target_metadata = Base.metadata


# =========================================================
# Offline Migrations
# =========================================================

def run_migrations_offline() -> None:
    """
    Run migrations without creating a live database
    connection.
    """

    url = config.get_main_option(
        "sqlalchemy.url"
    )

    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={
            "paramstyle": "named"
        },
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


# =========================================================
# Online Migrations
# =========================================================

def run_migrations_online() -> None:
    """
    Run migrations using a live database connection.
    """

    connectable = engine_from_config(
        config.get_section(
            config.config_ini_section,
            {}
        ),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:

        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )

        with context.begin_transaction():
            context.run_migrations()


# =========================================================
# Entrypoint
# =========================================================

if context.is_offline_mode():
    run_migrations_offline()

else:
    run_migrations_online()