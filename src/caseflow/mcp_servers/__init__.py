"""CaseFlow MCP servers.

Each server is an independent FastMCP application that can be deployed, scaled and
secured on its own - the same way a real company exposes its order system, help
desk and knowledge base as separate services:

* ``commerce``  - orders, shipments, returns, refunds (authenticated, per-customer).
* ``helpdesk``  - tickets, escalations and case analytics (authenticated).
* ``knowledge`` - hybrid-search over the help center (public, read-only).
"""
