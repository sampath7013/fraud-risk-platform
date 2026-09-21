import asyncio
import time

import httpx


BASE_URL = "http://127.0.0.1:8000"

# The blocking test can take about 15 seconds because
# three 5-second blocking requests execute sequentially.
# Give the client enough time to wait for all responses.
TIMEOUT = httpx.Timeout(30.0)


async def send_request(
    client: httpx.AsyncClient,
    endpoint: str,
    request_number: int,
):
    start = time.perf_counter()

    try:
        response = await client.get(
            f"{BASE_URL}{endpoint}"
        )

        duration = time.perf_counter() - start

        print(
            f"Request {request_number}: "
            f"status={response.status_code}, "
            f"time={duration:.2f}s"
        )

    except httpx.TimeoutException:
        duration = time.perf_counter() - start

        print(
            f"Request {request_number}: "
            f"TIMEOUT after {duration:.2f}s"
        )


async def run_test(endpoint: str):
    print()
    print(f"Testing {endpoint}")
    print("-" * 50)

    start = time.perf_counter()

    async with httpx.AsyncClient(
        timeout=TIMEOUT
    ) as client:
        await asyncio.gather(
            send_request(client, endpoint, 1),
            send_request(client, endpoint, 2),
            send_request(client, endpoint, 3),
        )

    total_duration = time.perf_counter() - start

    print("-" * 50)
    print(
        f"Total time: {total_duration:.2f}s"
    )


async def main():
    await run_test("/blocking-demo")
    await run_test("/async-demo")


if __name__ == "__main__":
    asyncio.run(main())