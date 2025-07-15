#!/usr/bin/env python3


import os
import logging
import tempfile
from pathlib import Path
from typing import Optional, Protocol, Any

# Optional imports for DBus
try:
    import dbus  # type: ignore
    import dbus.service  # type: ignore
    import dbus.mainloop.glib  # type: ignore
    from gi.repository import GLib  # type: ignore
    dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
except ImportError as e:
    raise ImportError(f"Required DBus packages not found: {e}. Install with: pip install dbus-python PyGObject")

logger = logging.getLogger(__name__)

class ConfigProtocol(Protocol):
    """Protocol for config object"""
    vision_enabled: bool
    api_key: str
    vision_model: str

class DBusHandler:
    """DBus handler for JARVIS"""

    def __init__(self, config: Optional[ConfigProtocol] = None, bot: Optional[Any] = None):
        """Initialize DBus handler
        
        Args:
            config: Configuration object with vision_enabled, api_key, and vision_model attributes
        """
        try:
            # Initialize DBus
            self.bus = dbus.SessionBus()
            
            # Get KGlobalAccel interface
            kglobalaccel = self.bus.get_object(
                'org.kde.kglobalaccel',
                '/kglobalaccel'
            )
            self.kglobalaccel_interface = dbus.Interface(
                kglobalaccel,
                dbus_interface='org.kde.KGlobalAccel'
            )
            
            # Keep a reference to the Jarvis bot (if provided) so we can hand off analyses
            self.bot: Any = bot  # type: ignore  # May be None when running standalone

            # Initialize vision module if config is provided
            self.vision_module = None
            if config and config.vision_enabled:
                try:
                    from groq import Groq
                    from modules.vision_module import VisionModule
                    
                    # Initialize Groq client
                    api_key = config.api_key or os.getenv("GROQ_API_KEY")
                    if api_key:
                        client = Groq(api_key=api_key)
                        self.vision_module = VisionModule(client, config.vision_model)
                        logger.info("Vision module initialized successfully")
                    else:
                        logger.error("No API key found for vision module")
                except ImportError as e:
                    logger.error(f"Failed to import vision dependencies: {e}")
                except Exception as e:
                    logger.error(f"Failed to initialize vision module: {e}")
            
            # Register global shortcut
            self._register_shortcut()

            # NEW: Listen for Spectacle "ScreenshotTaken" signal so we can react instantly
            try:
                self.bus.add_signal_receiver(
                    handler_function=self._handle_screenshot_taken,
                    dbus_interface="org.kde.Spectacle",
                    signal_name="ScreenshotTaken",
                    path="/org/kde/Spectacle",
                )
                logger.info("Connected to Spectacle ScreenshotTaken signal")
            except Exception as sig_e:
                logger.warning(f"Could not connect to Spectacle ScreenshotTaken signal: {sig_e}")
            
        except Exception as e:
            logger.error(f"Failed to initialize DBus handler: {e}")
            raise

    def _register_shortcut(self):
        """Register Super+Shift+C shortcut with KDE"""
        if not self.kglobalaccel_interface:
            return
        
        try:
            # Register the shortcut with correct types
            # Signature: asaiu = array of strings, array of strings, unsigned int, unsigned int
            component = ["jarvis"]  # Component name (array of strings)
            action = "Screenshot Analysis"  # Action name (string)
            keys = ["Meta+Shift+C"]  # Shortcut keys (array of strings)
            action_type = dbus.UInt32(0)  # Action type (unsigned integer)
            flags = dbus.UInt32(0)  # Flags (unsigned integer)
            
            # First set the shortcut
            self.kglobalaccel_interface.setShortcut(
                component,
                action,
                keys,
                action_type,
                flags
            )
            
            # Then connect the shortcut to our handler
            self.kglobalaccel_interface.connect(
                "jarvis",  # Component name
                action,  # Action name
                "org.jarvis.Service",  # Service name
                "/org/jarvis/Service",  # Object path
                "TriggerScreenshotAnalysis"  # Method name
            )
            
            logger.info("Successfully registered global shortcut Meta+Shift+C")
            
        except dbus.exceptions.DBusException:
            pass
    
    def _handle_screenshot_taken(self, filename: str):
        """Handle the ScreenshotTaken signal from Spectacle"""
        try:
            # Spectacle sometimes sends the filename as a QByteArray / dbus string – make sure we have a plain str
            path = str(filename)
            logger.info(f"ScreenshotTaken signal received: {path}")

            if not os.path.exists(path):
                logger.warning(f"Screenshot file does not exist yet: {path}")
                return

            # Process directly for the fastest response
            self._process_screenshot(path)
        except Exception as e:
            logger.error(f"Error handling ScreenshotTaken signal: {e}")
    
    @dbus.service.method("org.jarvis.Service", in_signature="", out_signature="")
    def TriggerScreenshotAnalysis(self):
        """DBus method called when the shortcut is triggered"""
        logger.info("Screenshot analysis triggered")
        try:
            spectacle = None
            try:
                spectacle = self.bus.get_object(
                    'org.kde.Spectacle',
                    '/org/kde/Spectacle'
                )
                spectacle_interface = dbus.Interface(
                    spectacle,
                    dbus_interface='org.kde.Spectacle'
                )
            except dbus.exceptions.DBusException:
                # Spectacle is not running, start it
                logger.info("Spectacle not running, starting it...")
                os.system("spectacle --background &")
                
                # Wait a bit for Spectacle to start
                GLib.timeout_add(1000, self._delayed_capture)
                return
            
            # If Spectacle is already running, capture immediately
            self._capture_region(spectacle_interface)
            
        except Exception as e:
            logger.error(f"Failed to trigger screenshot: {e}")
    
    def _delayed_capture(self):
        """Delayed capture after starting Spectacle"""
        try:
            spectacle = self.bus.get_object(
                'org.kde.Spectacle',
                '/org/kde/Spectacle'
            )
            spectacle_interface = dbus.Interface(
                spectacle,
                dbus_interface='org.kde.Spectacle'
            )
            self._capture_region(spectacle_interface)
        except dbus.exceptions.DBusException as e:
            logger.error(f"Failed to capture after delay: {e}")
        
        return False  # Don't repeat the timeout
    
    def _capture_region(self, spectacle_interface):
        """Capture region using Spectacle interface"""
        try:
            # Try different capture methods
            methods = [
                'CaptureRegion',
                'captureRegion',
                'takeNewScreenshot',
                'StartAgent'
            ]
            
            for method in methods:
                try:
                    if hasattr(spectacle_interface, method):
                        getattr(spectacle_interface, method)()
                        logger.info(f"Successfully triggered {method}")
                        return
                except dbus.exceptions.DBusException as e:
                    logger.debug(f"Method {method} failed: {e}")
                    continue
            
            # If all methods fail, try launching Spectacle directly
            logger.info("All DBus methods failed, launching Spectacle directly")
            os.system("spectacle --region --background")
            
        except Exception as e:
            logger.error(f"Failed to capture region: {e}")

    def _process_screenshot(self, image_path: str) -> None:
        """Process a screenshot using the vision module"""
        try:
            if self.vision_module is None:
                logger.error("Vision module not initialized")
                return
            
            # 1) Obtain objective description via vision model
            objective_desc = self.vision_module.analyze_image(
                image_source=image_path,
                prompt="Describe this screenshot objectively.",
                temperature=0.5,
                max_tokens=512,
            )

            logger.info(f"Objective screenshot description: {objective_desc}")

            final_reply = objective_desc

            # 2) Prefer feeding description into Jarvis's normal input loop (ensures full tone, history, etc.)
            if self.bot is not None and getattr(self.bot, "loop", None):
                try:
                    import asyncio

                    async def _inject():
                        await self.bot.process_command_line(objective_desc)  # type: ignore

                    asyncio.run_coroutine_threadsafe(_inject(), self.bot.loop)  # type: ignore
                    final_reply = ""  # Message will be printed by Jarvis core
                except Exception as exc:
                    logger.error(f"Failed to inject description into Jarvis loop: {exc}")

            # If that still didn't work, fallback to crafting with main LLM (no Jarvis pipe)
            if final_reply == objective_desc and self.bot is not None and hasattr(self.bot, "client"):
                try:
                    sys_prompt = getattr(self.bot.config, "system_prompt", None)
                    messages = []
                    if sys_prompt:
                        messages.append({"role": "system", "content": sys_prompt})
                    messages.append({
                        "role": "user",
                        "content": f"The screenshot has been analysed and the following was observed:\n\n{objective_desc}\n\nRespond to the user in your usual manner.",
                    })

                    llm_resp = self.bot.client.chat.completions.create(  # type: ignore
                        model=self.bot.config.main_model,
                        messages=messages,
                        max_tokens=512,
                        temperature=0.65,
                    )
                    final_reply = (llm_resp.choices[0].message.content or "").strip()
                except Exception as exc:
                    logger.error(f"Secondary Jarvis craft failed: {exc}")

            # Display final reply
            if self.bot is not None and hasattr(self.bot, "colors"):
                try:
                    self.bot.colors.print_ai_response(f"\n{self.bot.config.bot_name}: {final_reply}\n")
                except Exception:
                    print(f"\n{self.bot.config.bot_name}: {final_reply}\n")
                # Speak via Jarvis TTS
                try:
                    self.bot._speak_via_tts(final_reply)
                except Exception:
                    pass
            else:
                print(f"Jarvis: {final_reply}")

            logger.info(f"Screenshot final Jarvis reply: {final_reply}")
            
            # Delete the screenshot after processing
            try:
                os.remove(image_path)
                logger.info(f"Deleted screenshot: {image_path}")
            except Exception as e:
                logger.warning(f"Failed to delete screenshot {image_path}: {e}")
            
        except Exception as e:
            logger.error(f"Failed to process screenshot: {e}")
            # Try to delete the screenshot even if processing failed
            try:
                os.remove(image_path)
                logger.info(f"Deleted screenshot after failed processing: {image_path}")
            except Exception as del_e:
                logger.warning(f"Failed to delete screenshot {image_path}: {del_e}")

    def _monitor_tmp_directory(self):
        """Monitor /tmp/ directory for new Spectacle screenshots"""
        import time
        from pathlib import Path
        
        tmp_dir = Path("/tmp")
        known_files = set()
        
        while True:
            try:
                # Look for Spectacle screenshot files
                for file in tmp_dir.glob("spectacle-*.png"):
                    if file not in known_files and file.is_file():
                        logger.info(f"Found new screenshot: {file}")
                        self._process_screenshot(str(file))
                        known_files.add(file)
                        
                        # Clean up old files from known_files set
                        known_files = {f for f in known_files if f.exists()}
                        
            except Exception as e:
                logger.error(f"Error monitoring /tmp/: {e}")
            
            time.sleep(1)  # Check every second

    def run(self):
        """Start the DBus handler and screenshot monitoring"""
        # Start screenshot monitoring in a separate thread
        import threading
        monitor_thread = threading.Thread(target=self._monitor_tmp_directory, daemon=True)
        monitor_thread.start()
        
        # Run the DBus main loop
        from gi.repository import GLib  # type: ignore
        self.loop = GLib.MainLoop()
        self.loop.run()
    
    def stop(self):
        """Stop the DBus service"""
        logger.info("Stopping JARVIS DBus service")
        if self.loop.is_running():
            self.loop.quit()

    @dbus.service.method("org.jarvis.Service", in_signature="s", out_signature="s")
    def Query(self, user_input: str) -> str:
        """Handle text-based queries from external clients (e.g. KDE widget).

        Args:
            user_input: The prompt or command entered by the user.

        Returns:
            The assistant's response (or an error string if something went wrong).
        """
        logger.info(f"DBus Query received: {user_input}")

        # If no Jarvis bot instance is attached, we cannot process the query.
        if self.bot is None:
            logger.warning("Jarvis bot instance not available – returning placeholder response")
            return "Jarvis is not running or DBus handler was initialised without bot reference."

        try:
            import asyncio

            # If the bot has an active asyncio loop, execute inside that loop for full context
            if getattr(self.bot, "loop", None) and self.bot.loop.is_running():  # type: ignore[attr-defined]
                future = asyncio.run_coroutine_threadsafe(self.bot.generate_response(user_input), self.bot.loop)  # type: ignore[attr-defined]
                response = future.result(timeout=120)  # Wait up to 2 minutes
            else:
                # Fall back to running in a temporary loop (should not normally happen)
                response = asyncio.run(self.bot.generate_response(user_input))

            logger.info("Query processed successfully via Jarvis bot")
            return response or "(no response)"

        except Exception as exc:
            logger.error(f"Error while processing DBus Query: {exc}")
            return f"Error processing query: {exc}"

def is_dbus_available() -> bool:
    """Check if DBus support is available"""
    return True # DBus is now always available with the new import

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    def process_screenshot(filename: str):
        print(f"Processing screenshot: {filename}")
    
    # Initialize and run service
    service = DBusHandler() # No config needed for this example
    try:
        service.run()
    except KeyboardInterrupt:
        service.stop()