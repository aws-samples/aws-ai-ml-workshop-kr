#!/usr/bin/env python3
"""
Test script for new graph.stream_async() functionality
"""
import asyncio
from src.graph.builder import build_graph

async def test_stream_async():
    """Test the new graph.stream_async method"""
    print("🧪 Testing graph.stream_async()...")
    
    graph = build_graph()
    
    test_task = {
        "request": "Hello, can you help me test the streaming functionality?",
        "request_prompt": "Here is a user request: <user_request>Hello, can you help me test the streaming functionality?</user_request>"
    }
    
    print("📡 Starting streaming...")
    event_count = 0
    
    try:
        async for event in graph.stream_async(test_task):
            event_count += 1
            print(f"📥 Event #{event_count}: {event}")
            
            # Limit events for testing
            if event_count >= 10:
                print("🔄 Stopping after 10 events for testing...")
                break
                
    except Exception as e:
        print(f"❌ Error during streaming: {e}")
        
    print(f"✅ Streaming test completed! Received {event_count} events")

if __name__ == "__main__":
    asyncio.run(test_stream_async())