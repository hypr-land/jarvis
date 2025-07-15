#!/usr/bin/env python3

import os
import json
import asyncio
import logging
from typing import Optional, Dict, Any
try:
    from flask import Flask, request, jsonify, render_template
    from flask_socketio import SocketIO, emit
except ImportError:
    raise ImportError("Flask and Flask-SocketIO are required. Install with: pip install flask flask-socketio")

from dataclasses import asdict
from threading import Thread, Lock
import psutil
import queue

from modules.config_manager import BotConfig, ConfigManager
from modules.process_monitor import ProcessMonitor

logger = logging.getLogger(__name__)

class JarvisServer:
    """JARVIS Server implementation with Flask and WebSocket support"""
    
    def __init__(self, bot, config: BotConfig, host: str = "0.0.0.0", port: int = 5000):
        """Initialize the JARVIS server
        
        Args:
            bot: JarvisBot instance
            config: Bot configuration
            host: Host to bind to
            port: Port to listen on
        """
        self.bot = bot
        self.config = config
        self.host = host
        self.port = port
        
        # Initialize Flask and SocketIO
        self.app = Flask(__name__,
                        template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
                        static_folder=os.path.join(os.path.dirname(__file__), 'static'))
        self.socketio = SocketIO(self.app, cors_allowed_origins="*", async_mode='threading')
        
        # Process monitoring
        self.monitor = ProcessMonitor(os.getpid(), interval=config.monitoring_interval)
        self.monitor_lock = Lock()
        self.last_stats: Dict[str, Any] = {}
        
        # Response queue for async handling
        self.response_queue = queue.Queue()
        
        # Register routes
        self._register_routes()
        self._register_socketio_handlers()
        
        # Start background tasks
        self.start_background_tasks()
    
    def _register_routes(self):
        """Register HTTP routes"""
        
        @self.app.route('/')
        def index():
            """Render the main web interface"""
            return render_template('index.html', bot_name=self.config.bot_name)
        
        @self.app.route('/api/query', methods=['POST'])
        def query():
            """Handle JARVIS queries via HTTP"""
            data = request.get_json()
            if not data or 'query' not in data:
                return jsonify({'error': 'No query provided'}), 400
            
            try:
                # Run async code in a new thread
                def run_query():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        response = loop.run_until_complete(self.bot.generate_response(data['query']))
                        self.response_queue.put(response)
                    except Exception as e:
                        self.response_queue.put(e)
                    finally:
                        loop.close()
                
                thread = Thread(target=run_query)
                thread.start()
                thread.join(timeout=30)  # Wait up to 30 seconds
                
                result = self.response_queue.get(timeout=1)
                if isinstance(result, Exception):
                    raise result
                    
                return jsonify({'response': result})
            except Exception as e:
                logger.error(f"Error generating response: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/debug')
        def debug():
            """Get debug information"""
            with self.monitor_lock:
                return jsonify(self.last_stats)
        
        @self.app.route('/api/config')
        def get_config():
            """Get current configuration"""
            # Filter out sensitive information
            config_dict = asdict(self.config)
            sensitive_fields = ['api_key', 'access_key']
            for field in sensitive_fields:
                if field in config_dict:
                    config_dict[field] = None
            return jsonify(config_dict)
    
    def _register_socketio_handlers(self):
        """Register WebSocket event handlers"""
        
        @self.socketio.on('connect')
        def handle_connect():
            """Handle client connection"""
            logger.info("Client connected")
            # Send initial stats
            with self.monitor_lock:
                emit('stats_update', self.last_stats)
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection"""
            logger.info("Client disconnected")
        
        @self.socketio.on('query')
        def handle_query(data):
            """Handle JARVIS queries via WebSocket"""
            if not data or 'query' not in data:
                emit('error', {'type': 'error', 'data': {'message': 'No query provided'}})
                return
            
            def run_query():
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        response = loop.run_until_complete(self.bot.generate_response(data['query']))
                        self.socketio.emit('response', {'type': 'response', 'data': {'response': response}})
                    finally:
                        loop.close()
                except Exception as e:
                    logger.error(f"Error handling query: {e}")
                    self.socketio.emit('error', {'type': 'error', 'data': {'message': str(e)}})
            
            # Run in a new thread
            Thread(target=run_query).start()
    
    def start_background_tasks(self):
        """Start background tasks for monitoring and updates"""
        def update_stats():
            while True:
                try:
                    # Get process-specific stats
                    process = psutil.Process()
                    cpu_count = psutil.cpu_count() or 1  # Default to 1 if None
                    process_cpu = process.cpu_percent(interval=1) / cpu_count
                    process_memory = process.memory_percent()
                    
                    stats = {
                        'cpu_percent': process_cpu,  # Per-core CPU usage
                        'memory_percent': process_memory,
                        'memory_info': dict(process.memory_info()._asdict()),
                        'connections': len(process.connections()),
                        'threads': process.num_threads(),
                        'status': 'running'
                    }
                    
                    with self.monitor_lock:
                        self.last_stats = stats
                        
                    # Emit stats update to all connected clients
                    self.socketio.emit('stats_update', stats)
                    
                except Exception as e:
                    logger.error(f"Error updating stats: {e}")
                finally:
                    # Update every 5 seconds
                    self.socketio.sleep(5)
        
        # Start the stats update thread
        Thread(target=update_stats, daemon=True).start()
    
    def run(self):
        """Run the server"""
        logger.info(f"Starting JARVIS server on {self.host}:{self.port}")
        self.socketio.run(self.app, host=self.host, port=self.port, debug=False)
    
    def stop(self):
        """Stop the server"""
        # Cleanup tasks
        self.monitor.stop() 