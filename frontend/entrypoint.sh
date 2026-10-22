#!/bin/sh
set -eu

BACKEND_URL="${BACKEND_URL%/}"
export BACKEND_URL
envsubst '${BACKEND_URL}' < /etc/nginx/countwell.conf.template > /etc/nginx/conf.d/default.conf

# Encode deployment configuration as data, even when its value contains quotes.
encoded_api_url=$(printf '%s' "${API_BASE_URL:-}" | base64 | tr -d '\n')
printf 'globalThis.COUNTWELL_CONFIG = { apiBaseUrl: new TextDecoder().decode(Uint8Array.from(atob("%s"), c => c.charCodeAt(0))) };\n' "$encoded_api_url" > /usr/share/nginx/html/config.js
