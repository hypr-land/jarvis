#!/bin/bash

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p ~/.local/share/applications/
cp "$PROJECT_DIR/jarvis-screenshot.desktop" ~/.local/share/applications/
sed -i "s|/home/apollo/Documents/Development/J|$PROJECT_DIR|" ~/.local/share/applications/jarvis-screenshot.desktop
chmod +x ~/.local/share/applications/jarvis-screenshot.desktop

if command -v kquitapp5 >/dev/null 2>&1; then
    kquitapp5 kglobalaccel && sleep 2 && /usr/lib/kf5/kglobalaccel5 &
fi

if [ -f "$HOME/.zshrc" ]; then
  RC_FILE="$HOME/.zshrc"
else
  RC_FILE="$HOME/.bashrc"
fi

add_or_update_key() {
  local key_name="$1"
  local key_value="$2"
  if grep -q "export ${key_name}=" "$RC_FILE"; then
    sed -i "s|export ${key_name}=.*|export ${key_name}=\"${key_value}\"|" "$RC_FILE"
  else
    echo "export ${key_name}=\"${key_value}\"" >> "$RC_FILE"
  fi
}

read -p "Enter your GROQ_API_KEY (leave blank to skip): " INPUT_GROQ_KEY
if [ -n "$INPUT_GROQ_KEY" ]; then
  add_or_update_key "GROQ_API_KEY" "$INPUT_GROQ_KEY"
  echo "GROQ_API_KEY saved to $RC_FILE"
fi

read -p "Enter your PICOVOICE_ACCESS_KEY (leave blank to skip): " INPUT_PV_KEY
if [ -n "$INPUT_PV_KEY" ]; then
  add_or_update_key "PICOVOICE_ACCESS_KEY" "$INPUT_PV_KEY"
  echo "PICOVOICE_ACCESS_KEY saved to $RC_FILE"
fi

echo "Getting dependencies..."
pip install -r requirements.txt

echo -e "\nInstallation complete. Open a new terminal session or run 'source $RC_FILE' to apply environment variables."
echo "Shortcut installed. You may need to log out and back in for the changes to take effect."
echo "You can also configure the shortcut in System Settings > Shortcuts > Custom Shortcuts"
