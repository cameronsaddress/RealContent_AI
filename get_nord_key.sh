#!/bin/bash
echo "----------------------------------------------------------------"
echo "NordVPN WireGuard Key Fetcher"
echo "----------------------------------------------------------------"
echo "1. Go to: https://my.nordaccount.com/dashboard/nordvpn/token/"
echo "2. Click 'Generate new token'"
echo "3. Select 'Forever' (or any duration) and create it."
echo "----------------------------------------------------------------"
echo "Paste that token below and press Enter:"
read -r TOKEN

if [ -z "$TOKEN" ]; then
    echo "Error: Token cannot be empty."
    exit 1
fi

echo ""
echo "Fetching WireGuard Key..."
RESPONSE=$(curl -s -u "token:$TOKEN" https://api.nordvpn.com/v1/users/services/credentials)

# Check for unauthorized or error
if [[ "$RESPONSE" == *"Unauthorized"* ]]; then
    echo "Error: Invalid Token or API Error."
    exit 1
fi

KEY=$(echo "$RESPONSE" | jq -r '.[] | select(.service == "nordlynx") | .private_key')

if [ -z "$KEY" ] || [ "$KEY" == "null" ]; then
    echo "Error: Could not find 'nordlynx' (WireGuard) key in response."
    echo "Raw response: $RESPONSE"
    exit 1
fi

echo "----------------------------------------------------------------"
echo "SUCCESS! Your WireGuard Private Key is:"
echo ""
echo "WIREGUARD_PRIVATE_KEY=$KEY"
echo ""
echo "----------------------------------------------------------------"
echo "Next Step: Copy the line above and paste it into your .env file:"
echo "nano .env"
