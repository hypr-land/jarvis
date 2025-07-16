#!/usr/bin/env python3


import dbus
import dbus.service
import dbus.mainloop.glib
from gi.repository import GLib
import logging
import threading
import time
import os
import tempfile
import subprocess
import signal
import atexit
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Any, Set

dbus.mainloop.glib.threads_init()
dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)

class JarvisDBusService(dbus.service.Object):
    """DBus service for JARVIS screenshot functionality"""
    
    def __init__(self, bus_name, vision_module, jarvis_bot):
        self.vision = vision_module
        self.jarvis = jarvis_bot
        self.logger = logging.getLogger(__name__)
        self._temp_files: Set[str] = set()
        self._cleanup_lock = threading.Lock()
        atexit.register(self._cleanup_temp_files)
        super().__init__(bus_name, '/org/jarvis/Service')
        
    @dbus.service.method('org.jarvis.Service', in_signature='', out_signature='s')
    def TestService(self):
        """Test method to verify service is working"""
        self.logger.info("TestService called")
        return "JARVIS DBus service is working!"
    
    @dbus.service.method('org.jarvis.Service', in_signature='', out_signature='s')
    def TriggerScreenshotAnalysis(self):
        """Trigger screenshot analysis"""
        try:
            self.logger.info("TriggerScreenshotAnalysis called")
            
            # Take screenshot using system tools
            screenshot_path = self.take_screenshot()
            
            if screenshot_path and os.path.exists(screenshot_path):
                self.logger.info(f"Screenshot saved: {screenshot_path}")
                
                # Process the screenshot through Jarvis's vision module
                if self.jarvis and self.jarvis.vision_module:
                    self.logger.info("Starting screenshot analysis...")
                    result = self.jarvis.vision_module.process_image(screenshot_path)
                    self.logger.debug(f"Analysis complete: {result}")
                    
                    # Display result in Jarvis console
                    if hasattr(self.jarvis, 'colors'):
                        try:
                            self.jarvis.colors.print_success(f"Screenshot Analysis: {result}")
                        except Exception as e:
                            self.logger.error(f"Failed to print analysis result: {e}")
                    
                    return f"Screenshot analyzed: {result}"
                else:
                    self.logger.error("Vision module not available")
                    return "Error: Vision module not available"
            else:
                self.logger.error("Failed to take screenshot")
                return "Failed to take screenshot"
                
        except Exception as e:
            self.logger.error(f"Error in TriggerScreenshotAnalysis: {e}", exc_info=True)
            return f"Error: {str(e)}"
    
    @dbus.service.method('org.jarvis.Service', in_signature='s', out_signature='s')
    def ProcessImageFile(self, image_path):
        """Process an existing image file"""
        try:
            self.logger.info(f"ProcessImageFile called with: {image_path}")
            
            if not os.path.exists(image_path):
                return f"File not found: {image_path}"
            
            result = self.process_screenshot(image_path)
            return f"Image processed: {result}"
            
        except Exception as e:
            self.logger.error(f"Error in ProcessImageFile: {e}")
            return f"Error: {str(e)}"
    
    def _cleanup_temp_files(self):
        """Clean up any temporary files created by this service"""
        with self._cleanup_lock:
            for file_path in list(self._temp_files):
                try:
                    if os.path.exists(file_path):
                        os.unlink(file_path)
                    self._temp_files.remove(file_path)
                except Exception as e:
                    self.logger.warning(f"Failed to clean up temp file {file_path}: {e}")

    def _register_temp_file(self, file_path: str) -> None:
        """Register a temporary file for cleanup"""
        with self._cleanup_lock:
            self._temp_files.add(file_path)

    def take_screenshot(self):
        """Take a screenshot using system tools"""
        try:
            # Create temp file with cleanup on exit
            fd, screenshot_path = tempfile.mkstemp(suffix='.png', prefix='jarvis_screenshot_')
            os.close(fd)
            self._register_temp_file(screenshot_path)
            
            # Use spectacle with rectangular selection saved directly to file
            if shutil.which('spectacle'):
                # -r interactive region, -b background (no GUI), -o output file
                cmd = ['spectacle', '-r', '-b', '-o', screenshot_path]
                subprocess.run(cmd, check=True)
                # Wait up to 10 s for the user to finish region selection and for Spectacle to actually write the file.
                for _ in range(100):  # 100 × 0.1 s = 10 s
                    if os.path.exists(screenshot_path) and os.path.getsize(screenshot_path) > 0:
                        self.logger.info("Successfully took screenshot using spectacle")
                        return screenshot_path
                    time.sleep(0.1)
                
                self.logger.error(f"Screenshot file empty or not created: {screenshot_path}")
            else:
                self.logger.error("spectacle not found, falling back to other tools")
                # Try other screenshot tools
                if shutil.which('grim'):
                    cmd = ['grim', screenshot_path]
                    subprocess.run(cmd, check=True)
                elif shutil.which('scrot'):
                    cmd = ['scrot', '-s', screenshot_path]
                    subprocess.run(cmd, check=True)
                else:
                    raise RuntimeError("No screenshot tool found")
            
            if os.path.exists(screenshot_path):
                return screenshot_path
            else:
                self.logger.error("Screenshot file not created")
                return None
            
        except Exception as e:
            self.logger.error(f"Error taking screenshot: {e}")
            return None
    
    def process_screenshot(self, image_path):
        """Process screenshot with vision module"""
        try:
            if self.jarvis and self.jarvis.vision_module:
                result = self.jarvis.vision_module.analyze_image(image_path)
                self.logger.info(f"Vision analysis result: {result}")
                return result
            else:
                return "Vision module not available"
                
        except Exception as e:
            self.logger.error(f"Error processing screenshot: {e}", exc_info=True)
            return f"Processing error: {str(e)}"


class DesktopIntegration:
    """Desktop integration for JARVIS with fixed DBus service"""
    
    def __init__(self, config, vision_module, jarvis_bot):
        self.config = config
        self.vision = vision_module
        self.jarvis = jarvis_bot
        self.logger = logging.getLogger(__name__)
        
        self.bus = None
        self.service = None
        self.loop = None
        self.loop_thread = None
        self.running = False
        self._shutdown_event = threading.Event()
        self._dbus_warnings_suppressed = False
        
        # Suppress DBus warnings for specific errors we handle
        if not self._dbus_warnings_suppressed:
            import warnings
            warnings.filterwarnings('ignore', category=Warning)
            self._dbus_warnings_suppressed = True
            
        # Register cleanup on exit after everything is initialized
        atexit.register(self.stop)
        


    def start(self):
        """Start desktop integration with fixed DBus service"""
        if self.running:
            self.logger.warning("Desktop integration is already running")
            return True
            
        try:
            # Test DBus connection
            self.logger.debug("Testing DBus connection...")
            self.bus = dbus.SessionBus()
            self.logger.debug("Connected to DBus session bus")
            
            # Request bus name with error handling
            try:
                bus_name = dbus.service.BusName('org.jarvis.Service', self.bus)
                self.logger.debug("Successfully registered DBus service name")
            except dbus.exceptions.NameExistsException:
                self.logger.error("DBus service name already exists")
                return False
            except Exception as e:
                self.logger.error(f"Failed to register DBus service: {e}", exc_info=True)
                return False
            
            # Create the service
            self.service = JarvisDBusService(bus_name, self.vision, self.jarvis)
            self.logger.debug("Created DBus service object")
            
            # Initialize DBus main loop
            self.loop = GLib.MainLoop()
            self._shutdown_event.clear()
            self.logger.debug("Initialized DBus main loop")
            
            # Start the main loop in a separate thread
            self.loop_thread = threading.Thread(
                target=self.run_loop,
                name="DBusMainLoop",
                daemon=True
            )
            self.loop_thread.start()
            self.logger.debug("Started DBus main loop thread")
            
            # Wait for the service to be fully initialized
            # Wait briefly for the service thread to come up (max 1 s)
            for _ in range(10):  # 10 × 0.1 s = 1 s
                if self.loop_thread.is_alive():
                    break
                time.sleep(0.1)
            
            # Verify the service thread is running
            if not self.loop_thread.is_alive():
                self.logger.error("DBus service thread failed to start")
                self.stop()
                return False
            
            self.running = True
            self.logger.info("Desktop integration started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start desktop integration: {e}", exc_info=True)
            self.stop()
            return False
    
    def _check_shutdown(self) -> bool:
        """Check if we should shut down the main loop"""
        if self._shutdown_event.is_set():
            if self.loop and self.loop.is_running():
                self.loop.quit()
            return False
        return True

    def stop(self) -> None:
        """Stop the desktop integration service"""
        if not self.running:
            return
            
        self.logger.info("Stopping desktop integration...")
        self._shutdown_event.set()
        
        if self.loop and self.loop.is_running():
            self.loop.quit()
            
        if self.loop_thread and self.loop_thread.is_alive():
            self.loop_thread.join(timeout=5.0)
            
        self.running = False
        self.logger.info("Desktop integration stopped")
    
    def run_loop(self):
        """Run the DBus main loop"""
        try:
            self.logger.info("Starting DBus main loop")
            # Add a timeout to periodically check for shutdown
            GLib.timeout_add_seconds(1, self._check_shutdown)
            self.loop.run()
        except Exception as e:
            self.logger.error(f"Error in DBus main loop: {e}", exc_info=True)
        finally:
            self.logger.info("DBus main loop stopped")
            self.running = False
            self._shutdown_event.set()
            
            # Clean up the loop
            if hasattr(self, 'loop') and self.loop:
                if self.loop.is_running():
                    self.loop.quit()
                self.loop = None
            
            # Clean up the service
            if hasattr(self, 'service') and self.service is not None:
                try:
                    if hasattr(self.service, '_cleanup_temp_files'):
                        self.service._cleanup_temp_files()
                except Exception as e:
                    self.logger.warning(f"Error cleaning up service: {e}")
            
            # Clean up DBus connection
            if hasattr(self, 'bus') and self.bus:
                try:
                    self.bus.close()
                except Exception as e:
                    self.logger.warning(f"Error closing DBus connection: {e}")
                finally:
                    self.bus = None


# Quick test script
def test_dbus_service():
    """Test the DBus service directly"""
    import sys
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize DBus main loop
        dbus.mainloop.glib.threads_init()
        dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
        
        # Get session bus
        bus = dbus.SessionBus()
        
        # Create bus name
        bus_name = dbus.service.BusName('org.jarvis.Service', bus)

        class MockJarvisBot:
            def __init__(self):
                self.vision_module = None
                self.colors = type('Colors', (), {
                    'print_success': lambda _, x: print(f"SUCCESS: {x}")
                })()
        
        # Create service object with mock objects
        service = JarvisDBusService(bus_name, None, MockJarvisBot())
        
        logger.info("DBus service created successfully")
        logger.info("Testing service methods...")
        
        # Test the service
        test_result = service.TestService()
        logger.info(f"TestService result: {test_result}")
        
        # Start the main loop
        logger.info("Starting main loop... Press Ctrl+C to stop")
        loop = GLib.MainLoop()
        
        try:
            loop.run()
        except KeyboardInterrupt:
            logger.info("Stopping service...")
            loop.quit()
        finally:
            # Clean up resources
            if service is not None and hasattr(service, '_cleanup_temp_files'):
                try:
                    service._cleanup_temp_files()
                except Exception as e:
                    logger.warning(f"Error cleaning up temp files: {e}")
            if bus is not None:
                try:
                    bus.close()
                except Exception as e:
                    logger.warning(f"Error closing bus: {e}")
                finally:
                    # Ensure we don't try to use a closed bus
                    bus = None
            
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    test_dbus_service()
