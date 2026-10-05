-- Runs once on first start of the Postgres container.
-- Each MCP server owns its own database (service boundaries); the API owns `samadhan`.
CREATE DATABASE commerce OWNER samadhan;
CREATE DATABASE helpdesk OWNER samadhan;
\connect samadhan
CREATE EXTENSION IF NOT EXISTS vector;
