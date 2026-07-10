# Security Notes

## Current Controls

- No secrets are required by the app.
- CSV uploads are bounded by request size and row-count limits.
- Prediction input must match the published schema.
- JSON responses include conservative browser security headers.
- Audit logs store aggregate prediction metadata, not raw uploaded files.

## Production Recommendations

- Add authentication before exposing real company financial data.
- Put the app behind HTTPS-only hosting.
- Rotate logs and move audit events to managed storage for long-running deployments.
- Add rate limiting at the host or proxy layer.
- Avoid uploading confidential raw financial records to public demo deployments.
