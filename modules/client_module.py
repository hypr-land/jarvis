#!/usr/bin/env python3

import os
import sys
import json
import asyncio
import logging
import websockets
import requests
from typing import Optional, Dict, Any
from rich.console import Console
from rich.prompt import Prompt
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout

logger = logging.getLogger(__name__)

class JarvisClient:
    """JARVIS client for interacting with the server"""
    
    def __init__(self, host: str = "localhost", port: int = 5000):
        """Initialize the JARVIS client
        
        Args:
            host: Server hostname
            port: Server port
        """
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.ws_url = f"ws://{host}:{port}"
        self.console = Console()
        self.stats: Dict[str, Any] = {}
        self.running = False
        
    async def connect(self):
        """Connect to the JARVIS server"""
        try:
            # Test HTTP connection
            response = requests.get(f"{self.base_url}/api/config")
            response.raise_for_status()
            config = response.json()
            
            self.console.print(f"[bold blue]Connected to {config.get('bot_name', 'JARVIS')} server[/]")
            self.console.print("[dim]Type your messages below. Use Ctrl+C to exit.[/]")
            print()  # Empty line for spacing
            
            return True
        except Exception as e:
            self.console.print(f"[bold red]Error connecting to server: {e}[/]")
            return False
    
    async def _update_stats(self, websocket):
        """Background task to receive and update stats"""
        try:
            while True:
                message = await websocket.recv()
                data = json.loads(message)
                if 'type' in data and data['type'] == 'stats_update':
                    self.stats = data['stats']
        except websockets.exceptions.ConnectionClosed:
            logger.info("Stats WebSocket connection closed")
        except Exception as e:
            logger.error(f"Error in stats update: {e}")
    
    def _render_stats(self) -> Panel:
        """Render current stats in a nice format"""
        if not self.stats:
            return Panel("No stats available", title="System Stats")
        
        table = Table(show_header=False, show_edge=False, box=None)
        table.add_column("Metric", style="blue")
        table.add_column("Value", style="green")
        
        stats = self.stats
        table.add_row("CPU Usage", f"{stats.get('cpu_percent', 0):.1f}%")
        table.add_row("Memory Usage", f"{stats.get('memory_percent', 0):.1f}%")
        table.add_row("Threads", str(stats.get('threads', 0)))
        table.add_row("Connections", str(stats.get('connections', 0)))
        table.add_row("Status", stats.get('status', 'unknown'))
        
        return Panel(table, title="System Stats")
    
    async def run(self):
        """Run the client interface"""
        if not await self.connect():
            return
        
        self.running = True
        
        try:
            async with websockets.connect(f"{self.ws_url}/socket.io/?EIO=4&transport=websocket") as websocket:
                # Start stats update task
                asyncio.create_task(self._update_stats(websocket))
                
                # Main interaction loop
                while self.running:
                    try:
                        # Create layout
                        layout = Layout()
                        layout.split_column(
                            Layout(name="stats", size=10),
                            Layout(name="input")
                        )
                        
                        # Update stats display
                        with Live(layout, refresh_per_second=4, screen=False):
                            layout["stats"].update(self._render_stats())
                            
                            # Get user input
                            query = Prompt.ask("\nYou")
                            if not query:
                                continue
                            
                            # Send query
                            await websocket.send(json.dumps({
                                "type": "query",
                                "data": {"query": query}
                            }))
                            
                            # Wait for response
                            response = await websocket.recv()
                            data = json.loads(response)
                            
                            if data.get('type') == 'response':
                                self.console.print(f"\n[bold blue]JARVIS:[/] {data['data']['response']}")
                            elif data.get('type') == 'error':
                                self.console.print(f"\n[bold red]Error:[/] {data['data']['message']}")
                    
                    except KeyboardInterrupt:
                        self.running = False
                        break
                    except Exception as e:
                        self.console.print(f"[bold red]Error:[/] {e}")
                        continue
        
        except websockets.exceptions.ConnectionClosed:
            self.console.print("[bold red]Connection to server closed[/]")
        except Exception as e:
            self.console.print(f"[bold red]Error:[/] {e}")
    
    def stop(self):
        """Stop the client"""
        self.running = False


async def main():
    """Main entry point for JARVIS client"""
    import argparse
    
    parser = argparse.ArgumentParser(description="JARVIS Client")
    parser.add_argument("--host", default="localhost", help="Server hostname")
    parser.add_argument("--port", type=int, default=5000, help="Server port")
    
    args = parser.parse_args()
    
    client = JarvisClient(host=args.host, port=args.port)
    await client.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nGoodbye!") 