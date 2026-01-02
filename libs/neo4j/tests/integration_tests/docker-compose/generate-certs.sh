#!/bin/bash
# Generate self-signed certificates for the OAuth2 proxy

mkdir -p certs

# Generate private key and certificate
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout certs/server.key \
    -out certs/server.crt \
    -subj "/CN=oauth2-server" \
    -addext "subjectAltName=DNS:oauth2-server,DNS:localhost"

echo "Certificates generated in certs/ directory"

