#!/bin/bash
# Quick KB Export - View knowledge base in Finder

# Create export directory on your Mac
EXPORT_DIR="$HOME/Desktop/zen-kb-export-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$EXPORT_DIR"

echo "📂 Exporting Knowledge Base to: $EXPORT_DIR"

# Export the entire KB from Docker to your Mac
docker cp zen-mcp-server:/tmp/zen-context-kb "$EXPORT_DIR/"

# Count entries
ENTRY_COUNT=$(find "$EXPORT_DIR/zen-context-kb" -name "*.json" -type f | wc -l | tr -d ' ')
echo "✅ Exported $ENTRY_COUNT knowledge entries"

# Open in Finder
open "$EXPORT_DIR/zen-context-kb"

echo "🎉 Knowledge base opened in Finder!"
echo "   You can now browse all entries using your Mac's file browser"