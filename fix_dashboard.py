#!/usr/bin/env python3
"""Fix the broken emoji and add the Responsible AI tab"""

import re

# Read the file
with open("dashboard/app.py", encoding="utf-8") as f:
    content = f.read()

# Fix the tabs line - replace the broken emoji and add Responsible AI tab
old_tabs = r'"📊 Data Exploration", "📈 Risk Analysis", "⚖️ Fairness", "🌊 Drift"'
new_tabs = '"📊 Data Exploration", "📈 Risk Analysis", "🛡️ Responsible AI", "⚖️ Fairness", "🌊 Drift"'

# Replace the broken character pattern
content = re.sub(r'"[^"]*Data Exploration"', '"📊 Data Exploration"', content)

# Add the Responsible AI tab to the array if not already there
if "🛡️ Responsible AI" not in content:
    content = content.replace(
        '"📊 Data Exploration", "📈 Risk Analysis", "⚖️ Fairness"',
        '"📊 Data Exploration", "📈 Risk Analysis", "🛡️ Responsible AI", "⚖️ Fairness"',
    )

# Write the fixed content back
with open("dashboard/app.py", "w", encoding="utf-8") as f:
    f.write(content)

print("Fixed the dashboard file!")
