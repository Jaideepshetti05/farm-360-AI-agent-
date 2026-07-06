import os
import sys
import time
import asyncio
from loguru import logger

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.streaming.stream_manager import StreamManager

async def run_concurrent_users(count: int):
    manager = StreamManager()
    context = {
        "user_profile": {"location": "Assam"},
        "language": "en",
        "session_id": "load_test_session"
    }
    
    async def single_user(user_id: int):
        start = time.time()
        chunks_count = 0
        async for event in manager.stream_query(f"User {user_id}: How to grow rice?", context):
            chunks_count += 1
        latency = (time.time() - start) * 1000.0
        return latency, chunks_count

    logger.info(f"Simulating {count} concurrent users executing streaming requests...")
    start_all = time.time()
    
    tasks = [single_user(i) for i in range(count)]
    results = await asyncio.gather(*tasks)
    
    total_time = (time.time() - start_all) * 1000.0
    latencies = [r[0] for r in results]
    avg_latency = sum(latencies) / len(latencies)
    
    logger.success(f"--- Load Test Summary ({count} users) ---")
    logger.info(f"Total Execution Time: {total_time:.2f} ms")
    logger.info(f"Average Request Latency: {avg_latency:.2f} ms")
    logger.info(f"P95 Latency: {percentile(latencies, 95):.2f} ms")

def percentile(data, percent):
    if not data:
        return 0.0
    sorted_data = sorted(data)
    index = (len(sorted_data) - 1) * percent / 100.0
    floor_idx = int(index)
    ceil_idx = floor_idx + 1 if floor_idx < len(sorted_data) - 1 else floor_idx
    return sorted_data[floor_idx] + (sorted_data[ceil_idx] - sorted_data[floor_idx]) * (index - floor_idx)

if __name__ == "__main__":
    asyncio.run(run_concurrent_users(20))
