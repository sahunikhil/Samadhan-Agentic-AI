-- Runs once on first start of the Postgres container.
-- Each MCP server owns its own database (service boundaries); the API owns `caseflow`.
CREATE DATABASE commerce OWNER caseflow;
CREATE DATABASE helpdesk OWNER caseflow;
\connect caseflow
CREATE EXTENSION IF NOT EXISTS vector;
