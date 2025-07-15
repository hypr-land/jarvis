#!/usr/bin/env python3


import os
import subprocess
import logging
import dbus
from pathlib import Path

class KDEHotkeySetup:
    """Setup global hotkeys for JARVIS in KDE environment"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config_dir = Path.home() / '.config'
        self.shortcuts_file = self.config_dir / 'kglobalshortcutsrc'
        
    def detect_kde_version(self):
        """Detect KDE Plasma version"""
        try:
            result = subprocess.run(['plasmashell', '--version'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                version = result.stdout.strip()
                self.logger.info(f"Detected KDE version: {version}")
                return version
            return None
        except Exception as e:
            self.logger.error(f"Failed to detect KDE version: {e}")
            return None
    
    def setup_custom_shortcut_kde(self):
        """Set up custom shortcut using KDE's kwriteconfig"""
        try:
            # KDE version detection
            kde_version = self.detect_kde_version()
            
            # Use kwriteconfig5 or kwriteconfig6 based on KDE version
            if kde_version and '6' in kde_version:
                kwriteconfig_cmd = 'kwriteconfig6'
            else:
                kwriteconfig_cmd = 'kwriteconfig5'
            
            # Create custom shortcut entry
            group_name = 'jarvis.desktop'
            
            commands = [
                # Create the shortcut entry
                [kwriteconfig_cmd, '--file', 'kglobalshortcutsrc', '--group', group_name, 
                 '--key', '_k_friendly_name', 'JARVIS Screenshot Analysis'],
                
                # Set the command to execute
                [kwriteconfig_cmd, '--file', 'kglobalshortcutsrc', '--group', group_name,
                 '--key', 'TriggerScreenshot', 'Meta+Shift+C,none,Trigger JARVIS Screenshot Analysis'],
                
                # Set the action
                [kwriteconfig_cmd, '--file', 'kglobalshortcutsrc', '--group', group_name,
                 '--key', '_launch', 'dbus-send --session --type=method_call --dest=org.jarvis.Service /org/jarvis/Service org.jarvis.Service.TriggerScreenshotAnalysis'],
            ]
            
            for cmd in commands:
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    if result.returncode == 0:
                        self.logger.info(f"Successfully executed: {' '.join(cmd)}")
                    else:
                        self.logger.warning(f"Command failed: {' '.join(cmd)}")
                        self.logger.warning(f"Error: {result.stderr}")
                except Exception as e:
                    self.logger.error(f"Failed to execute {cmd}: {e}")
            
            # Reload kglobalaccel
            self.reload_kglobalaccel()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to setup KDE custom shortcut: {e}")
            return False
    
    def setup_via_dbus_kglobalaccel(self):
        """Setup hotkey using KDE's KGlobalAccel DBus interface"""
        try:
            # Connect to session bus
            bus = dbus.SessionBus()
            
            # Get KGlobalAccel service
            kglobalaccel = bus.get_object('org.kde.kglobalaccel', '/kglobalaccel')
            
            # Define the shortcut
            component_name = 'jarvis'
            action_name = 'TriggerScreenshot'
            friendly_name = 'JARVIS Screenshot Analysis'
            shortcut = 'Meta+Shift+C'
            
            # Register the shortcut
            # The exact method signature may vary by KDE version
            try:
                # Try method 1 - newer KDE versions
                kglobalaccel.setShortcut(
                    component_name,
                    action_name,
                    friendly_name,
                    [shortcut],
                    dbus_interface='org.kde.KGlobalAccel'
                )
                self.logger.info("Registered shortcut using method 1")
                
            except Exception as e1:
                try:
                    # Try method 2 - older KDE versions
                    kglobalaccel.registerShortcut(
                        component_name,
                        action_name,
                        friendly_name,
                        shortcut,
                        dbus_interface='org.kde.KGlobalAccel'
                    )
                    self.logger.info("Registered shortcut using method 2")
                    
                except Exception as e2:
                    self.logger.error(f"Both DBus methods failed: {e1}, {e2}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to setup via DBus: {e}")
            return False
    
    def setup_via_system_settings(self):
        """Setup via KDE System Settings (manual method)"""
        try:
            # Launch System Settings to Custom Shortcuts
            subprocess.Popen(['systemsettings5', 'kcm_keys'])
            
            print("\nKDE System Settings opened!")
            print("To set up the hotkey manually:")
            print("1. Go to 'Shortcuts' -> 'Custom Shortcuts'")
            print("2. Click 'Edit' -> 'New' -> 'Global Shortcut' -> 'Command/URL'")
            print("3. Set Name: 'JARVIS Screenshot Analysis'")
            print("4. Set Command: 'dbus-send --session --type=method_call --dest=org.jarvis.Service /org/jarvis/Service org.jarvis.Service.TriggerScreenshotAnalysis'")
            print("5. Set Trigger: Click 'None' and press Super+Shift+C")
            print("6. Click 'Apply'")
            print("\nPress Enter when done...")
            input()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to open System Settings: {e}")
            return False
    
    def reload_kglobalaccel(self):
        """Reload KGlobalAccel to pick up changes"""
        try:
            # Method 1: Restart kglobalaccel
            subprocess.run(['kquitapp5', 'kglobalaccel'], capture_output=True)
            subprocess.run(['kstart5', 'kglobalaccel'], capture_output=True)
            
            # Method 2: Send DBus signal to reload
            bus = dbus.SessionBus()
            kglobalaccel = bus.get_object('org.kde.kglobalaccel', '/kglobalaccel')
            kglobalaccel.reloadConfig(dbus_interface='org.kde.KGlobalAccel')
            
            self.logger.info("KGlobalAccel reloaded")
            
        except Exception as e:
            self.logger.info(f"Reload attempt completed (some methods may have failed): {e}")
    
    def create_desktop_file(self):
        """Create a .desktop file for the application"""
        try:
            desktop_content = """[Desktop Entry]
Type=Application
Name=JARVIS Screenshot Analysis
Comment=AI-powered screenshot analysis
Exec=dbus-send --session --type=method_call --dest=org.jarvis.Service /org/jarvis/Service org.jarvis.Service.TriggerScreenshotAnalysis
Icon=camera-photo
Terminal=false
Categories=Graphics;Photography;
"""
            
            desktop_file = Path.home() / '.local/share/applications/jarvis-screenshot.desktop'
            desktop_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(desktop_file, 'w') as f:
                f.write(desktop_content)
            
            # Make it executable
            os.chmod(desktop_file, 0o755)
            
            self.logger.info(f"Created desktop file: {desktop_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create desktop file: {e}")
            return False
    
    def setup_all_methods(self):
        """Try all methods to setup the hotkey"""
        print("Setting up JARVIS hotkey for KDE...")
        print("=" * 50)
        
        # Method 1: Try DBus KGlobalAccel
        print("\n1. Trying DBus KGlobalAccel method...")
        if self.setup_via_dbus_kglobalaccel():
            print("✓ DBus method succeeded!")
            return True
        else:
            print("✗ DBus method failed")
        
        # Method 2: Try kwriteconfig
        print("\n2. Trying kwriteconfig method...")
        if self.setup_custom_shortcut_kde():
            print("✓ kwriteconfig method succeeded!")
            return True
        else:
            print("✗ kwriteconfig method failed")
        
        # Method 3: Manual setup
        print("\n3. Opening System Settings for manual setup...")
        self.setup_via_system_settings()
        
        # Create desktop file regardless
        self.create_desktop_file()
        
        return False


def main():
    """Main setup function"""
    logging.basicConfig(level=logging.INFO)
    
    setup = KDEHotkeySetup()
    
    # First check if we're actually on KDE
    kde_version = setup.detect_kde_version()
    if not kde_version:
        print("Warning: KDE Plasma not detected!")
        print("This script is designed for KDE environments.")
        print("Continue anyway? (y/n): ", end='')
        if input().lower() != 'y':
            return
    
    # Check if DBus service might be running
    try:
        result = subprocess.run([
            'dbus-send', '--session', '--print-reply', 
            '--dest=org.jarvis.Service', '/org/jarvis/Service', 
            'org.jarvis.Service.TestService'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ JARVIS DBus service is running")
        else:
            print("✗ JARVIS DBus service not detected")
            print("Make sure JARVIS is running before setting up hotkeys")
            print("Continue anyway? (y/n): ", end='')
            if input().lower() != 'y':
                return
    except Exception as e:
        print(f"Could not check DBus service: {e}")
    
    # Setup hotkeys
    setup.setup_all_methods()
    
    print("\n" + "=" * 50)
    print("Setup complete!")
    print("\nTo test the hotkey:")
    print("1. Press Super+Shift+C")
    print("2. Or run manually: dbus-send --session --type=method_call --dest=org.jarvis.Service /org/jarvis/Service org.jarvis.Service.TriggerScreenshotAnalysis")
    print("\nIf the hotkey doesn't work, try logging out and back in.")


if __name__ == "__main__":
    main()
