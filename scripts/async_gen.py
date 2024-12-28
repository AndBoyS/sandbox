import asyncio
from collections.abc import AsyncIterator, Generator
from typing import Any


class ReusableFuture:
    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop
        self._future = loop.create_future()

    def __await__(self) -> Generator[Any, None, Any]:
        return self._await_future().__await__()

    async def _await_future(self) -> None:
        await self._future
        self._future = self.loop.create_future()

    def set_result(self, val: Any) -> None:
        self._future.set_result(val)


async def long_func(fut: ReusableFuture) -> None:
    await asyncio.sleep(1)
    fut.set_result(1)
    await asyncio.sleep(1)
    fut.set_result(1)


async def gen() -> AsyncIterator[int]:
    loop = asyncio.get_running_loop()
    fut = ReusableFuture(loop)

    loop.create_task(long_func(fut))
    for _ in range(2):
        await fut
        yield 1


async def main() -> None:
    g = gen()
    async for val in g:
        print(val)


if __name__ == "__main__":
    asyncio.run(main())
