#!/usr/bin/env python3
"""
Enhanced run script for XAUUSD AI Analysis Backend
"""
import uvicorn
import asyncio
import os
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def main():
    """Main application runner"""
    try:
        config = uvicorn.Config(
            "app.main:app",
            host="0.0.0.0",
            port=int(os.getenv("PORT", 8000)),
            reload=os.getenv("DEBUG", "true").lower() == "true",
            log_level="info",
            access_log=True
        )
        
        server = uvicorn.Server(config)
        await server.serve()
        
    except Exception as e:
        logging.error(f"Failed to start server: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
