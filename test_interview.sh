#!/bin/bash
echo "Testing interview start endpoint..."
echo ""

response=$(curl -X POST http://127.0.0.1:1111/api/start_interview \
  -H "Content-Type: application/json" \
  -d '{"language": "java", "mode": "coding_only"}' \
  -s)

status=$(echo "$response" | python3 -c "import sys, json; print(json.load(sys.stdin).get('error', 'SUCCESS'))")

if [ "$status" = "SUCCESS" ]; then
    echo "✓ Interview started successfully!"
    echo "$response" | python3 -m json.tool
else
    echo "✗ Interview failed to start"
    echo "Error: $status"
fi
