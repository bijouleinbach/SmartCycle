import os

# The python.org macOS installer ships without trusted SSL certificates, which
# breaks downloads (TrashNet, pretrained weights) with CERTIFICATE_VERIFY_FAILED.
# Point Python at certifi's bundle unless the user already configured one.
try:
    import certifi

    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
except ImportError:
    pass
