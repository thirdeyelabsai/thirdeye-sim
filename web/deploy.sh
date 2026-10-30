#!/bin/sh
# Build the site (landing page, game, realism check) and deploy it to Vercel. The static build is copied into
# .deploy, which holds the link to the Vercel project (made once with `vercel link` there); .vercelignore there
# keeps env files out.
# The page carries noindex until launch. Run export.py first if the simulator or the map changed.
set -e
cd "$(dirname "$0")"
npm run build
rsync -a --delete --exclude .vercel --exclude .vercelignore --exclude .gitignore --exclude ".env*" dist/ .deploy/
cd .deploy
for attempt in 1 2 3; do   # (the upload sometimes ends in "fetch failed": try again)
    if vercel deploy --prod --yes; then exit 0; fi
    sleep 5
done
exit 1
