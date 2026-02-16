"""Unit tests for the persistent key-value store module."""


import pytest

from puffinflow.core.agent import Agent
from puffinflow.core.store import Item, MemoryStore
from puffinflow.core.store.base import BaseStore

# ---------------------------------------------------------------------------
# 1. test_put_get -- put a value, get it back, verify Item fields
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_put_get():
    store = MemoryStore()
    ns = ("users", "alice")

    await store.put(ns, "profile", {"name": "Alice", "age": 30})

    item = await store.get(ns, "profile")
    assert item is not None
    assert isinstance(item, Item)
    assert item.namespace == ns
    assert item.key == "profile"
    assert item.value == {"name": "Alice", "age": 30}
    assert isinstance(item.created_at, float)
    assert isinstance(item.updated_at, float)
    assert item.created_at <= item.updated_at
    assert item.metadata == {}


# ---------------------------------------------------------------------------
# 2. test_namespaces -- items in different namespaces are isolated
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_namespaces():
    store = MemoryStore()
    ns_alice = ("users", "alice")
    ns_bob = ("users", "bob")

    await store.put(ns_alice, "color", "blue")
    await store.put(ns_bob, "color", "red")

    alice_item = await store.get(ns_alice, "color")
    bob_item = await store.get(ns_bob, "color")

    assert alice_item is not None
    assert bob_item is not None
    assert alice_item.value == "blue"
    assert bob_item.value == "red"

    # Getting a key from the wrong namespace should return None
    assert await store.get(("users", "charlie"), "color") is None


# ---------------------------------------------------------------------------
# 3. test_delete -- delete returns True and item is gone; missing returns False
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete():
    store = MemoryStore()
    ns = ("temp",)

    await store.put(ns, "ephemeral", "data")
    assert await store.get(ns, "ephemeral") is not None

    result = await store.delete(ns, "ephemeral")
    assert result is True
    assert await store.get(ns, "ephemeral") is None

    # Deleting a non-existent key returns False
    result = await store.delete(ns, "ephemeral")
    assert result is False

    # Deleting a key that was never inserted also returns False
    result = await store.delete(ns, "never_existed")
    assert result is False


# ---------------------------------------------------------------------------
# 4. test_list -- list items in a namespace with limit and offset
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list():
    store = MemoryStore()
    ns = ("items",)

    # Insert 5 items with slight time gaps so ordering is deterministic
    for i in range(5):
        await store.put(ns, f"item_{i}", f"value_{i}")

    # List all -- should return 5 items, ordered by updated_at descending
    all_items = await store.list(ns)
    assert len(all_items) == 5
    # Most recently updated first
    assert all_items[0].key == "item_4"

    # Limit
    limited = await store.list(ns, limit=2)
    assert len(limited) == 2

    # Offset
    offset_items = await store.list(ns, limit=2, offset=2)
    assert len(offset_items) == 2
    assert offset_items[0].key == all_items[2].key

    # Items in a different namespace should not appear
    other_items = await store.list(("other",))
    assert len(other_items) == 0


# ---------------------------------------------------------------------------
# 5. test_search -- search finds items by query string match
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search():
    store = MemoryStore()
    ns = ("docs",)

    await store.put(ns, "readme", "Installation guide for puffinflow")
    await store.put(ns, "changelog", "Version 2.0 released")
    await store.put(ns, "license", "MIT License")

    # Search by value content
    results = await store.search(ns, query="puffinflow")
    assert len(results) == 1
    assert results[0].key == "readme"

    # Search by key content
    results = await store.search(ns, query="changelog")
    assert len(results) == 1
    assert results[0].key == "changelog"

    # Case-insensitive search
    results = await store.search(ns, query="MIT")
    assert len(results) == 1
    assert results[0].key == "license"

    # No matches
    results = await store.search(ns, query="nonexistent")
    assert len(results) == 0

    # Empty query returns all items under namespace
    results = await store.search(ns, query="")
    assert len(results) == 3


# ---------------------------------------------------------------------------
# 6. test_context_access -- agent with store= kwarg, context accesses ctx.store
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_context_access():
    store = MemoryStore()
    agent = Agent("store_agent", store=store)

    accessed_store = None

    async def check_store(ctx):
        nonlocal accessed_store
        accessed_store = ctx.store
        await ctx.store.put(("test",), "key1", "hello")
        return None  # End agent

    agent.add_state("start", check_store)

    await agent.run()

    assert accessed_store is store
    # Verify data persisted through the store
    item = await store.get(("test",), "key1")
    assert item is not None
    assert item.value == "hello"


# ---------------------------------------------------------------------------
# 7. test_cross_state_persistence -- put in one state, get in another state
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cross_state_persistence():
    store = MemoryStore()
    agent = Agent("persist_agent", store=store)

    async def state_write(ctx):
        await ctx.store.put(("session",), "counter", 42)
        return "state_read"

    async def state_read(ctx):
        item = await ctx.store.get(("session",), "counter")
        ctx.set_variable("retrieved_value", item.value)
        return None  # End agent

    agent.add_state("state_write", state_write)
    agent.add_state("state_read", state_read)

    result = await agent.run()

    # The value written in state_write was available in state_read
    assert result.variables.get("retrieved_value") == 42


# ---------------------------------------------------------------------------
# 8. test_protocol_conformance -- MemoryStore is instance of BaseStore
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_protocol_conformance():
    store = MemoryStore()
    # BaseStore is a runtime_checkable Protocol
    assert isinstance(store, BaseStore)

    # Verify all protocol methods exist and are callable
    assert callable(getattr(store, "put", None))
    assert callable(getattr(store, "get", None))
    assert callable(getattr(store, "delete", None))
    assert callable(getattr(store, "list", None))
    assert callable(getattr(store, "search", None))


# ---------------------------------------------------------------------------
# 9. test_sqlite_basic -- skip if aiosqlite not installed, basic put/get
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sqlite_basic():
    try:
        import aiosqlite  # noqa: F401
    except ImportError:
        pytest.skip("aiosqlite is not installed")

    from puffinflow.core.store.sqlite import SqliteStore

    store = SqliteStore(":memory:")
    try:
        ns = ("db", "test")

        await store.put(ns, "row1", {"col": "value1"})
        item = await store.get(ns, "row1")

        assert item is not None
        assert item.namespace == ns
        assert item.key == "row1"
        assert item.value == {"col": "value1"}
        assert isinstance(item.created_at, float)
        assert isinstance(item.updated_at, float)
        assert item.metadata == {}

        # Verify delete works
        deleted = await store.delete(ns, "row1")
        assert deleted is True
        assert await store.get(ns, "row1") is None
    finally:
        await store.close()


# ---------------------------------------------------------------------------
# 10. test_metadata -- put with metadata dict, verify it persists
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_metadata():
    store = MemoryStore()
    ns = ("meta",)

    meta = {"author": "alice", "version": 3, "tags": ["important", "reviewed"]}
    await store.put(ns, "doc1", "content", metadata=meta)

    item = await store.get(ns, "doc1")
    assert item is not None
    assert item.metadata == meta
    assert item.metadata["author"] == "alice"
    assert item.metadata["version"] == 3
    assert "important" in item.metadata["tags"]

    # Update the item value -- metadata should be merged on update
    await store.put(ns, "doc1", "updated content", metadata={"version": 4})
    item = await store.get(ns, "doc1")
    assert item.value == "updated content"
    assert item.metadata["version"] == 4
    # Original metadata keys are preserved via update
    assert item.metadata["author"] == "alice"
    assert "important" in item.metadata["tags"]
